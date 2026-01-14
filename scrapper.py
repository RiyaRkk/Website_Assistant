"""
Improved site crawler / scraper that upserts pages into MongoDB.
Now supports per-bot crawling via bot_id.
"""

import time
import logging
import argparse
import re
from collections import deque
from urllib.parse import urlparse, urljoin, urldefrag
from datetime import datetime
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import html2text
import urllib.robotparser
from pymongo import MongoClient, ASCENDING
import os

load_dotenv()

try:
    from requests_html import HTMLSession
    REQUESTS_HTML_AVAILABLE = True
except Exception:
    REQUESTS_HTML_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

USER_AGENT = "Mozilla/5.0 (compatible; CustomCrawler/1.0; +https://example.com/bot)"
REQUESTS_TIMEOUT = 20
POLITE_DELAY = 1.0  # seconds between requests
MAX_PAGES_DEFAULT = 500

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DBNAME = "chat_bot_db"
MONGO_COLLECTION = "pages"
DEMO_COLLECTION = "demo_pages"


client = MongoClient(MONGO_URI)
db = client[MONGO_DBNAME]
pages_col = db[MONGO_COLLECTION]
demo_pages_col = db[DEMO_COLLECTION]

try:
    pages_col.create_index([("url", ASCENDING), ("bot_id", ASCENDING)], unique=True)
except Exception:
    pass

try:
    demo_pages_col.create_index([("url", ASCENDING), ("bot_id", ASCENDING)], unique=True)
except Exception:
    pass

_html2text = html2text.HTML2Text()
_html2text.ignore_links = True
_html2text.body_width = 0

def normalize_url(base: str, link: str) -> str:
    if not link:
        return None
    joined = urljoin(base, link)
    clean, _ = urldefrag(joined)
    return clean.strip()

def same_domain(url: str, domain: str) -> bool:
    try:
        if not domain:
            return True
        return urlparse(url).netloc.endswith(domain)
    except Exception:
        return False

def load_robots_txt(start_url: str):
    parsed = urlparse(start_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        logging.warning("Could not fetch robots.txt; proceeding.")
    return rp

def get_sitemap_urls(start_url: str) -> list:
    parsed = urlparse(start_url)
    sitemap_urls = []
    candidates = [f"{parsed.scheme}://{parsed.netloc}/sitemap.xml",
                  f"{parsed.scheme}://{parsed.netloc}/sitemap_index.xml"]
    headers = {"User-Agent": USER_AGENT}
    for s in candidates:
        try:
            r = requests.get(s, headers=headers, timeout=REQUESTS_TIMEOUT)
            if r.status_code == 200 and 'xml' in r.headers.get('Content-Type', ''):
                locs = re.findall(r"<loc>(.*?)</loc>", r.text, flags=re.IGNORECASE)
                if locs:
                    sitemap_urls.extend(locs)
        except Exception:
            pass
    return sitemap_urls

def fetch_html(url: str, render_js: bool = False):
    headers = {"User-Agent": USER_AGENT}
    if render_js and REQUESTS_HTML_AVAILABLE:
        session = HTMLSession()
        try:
            r = session.get(url, timeout=REQUESTS_TIMEOUT, headers=headers)
            r.html.render(timeout=30, sleep=1)
            return r.html.raw_html.decode("utf-8", errors="ignore"), r.status_code
        except Exception as e:
            logging.debug(f"JS render failed for {url}: {e}")
            try:
                r = requests.get(url, headers=headers, timeout=REQUESTS_TIMEOUT)
                return r.text, r.status_code
            except Exception:
                return None, None
    else:
        try:
            r = requests.get(url, headers=headers, timeout=REQUESTS_TIMEOUT)
            return r.text, r.status_code
        except Exception as e:
            logging.debug(f"Request failed for {url}: {e}")
            return None, None

def extract_text_and_title(html: str):
    if not html:
        return "", ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for s in soup(["script", "style", "noscript"]):
            s.decompose()
        title_tag = soup.title.string.strip() if soup.title and soup.title.string else ""
        text = _html2text.handle(str(soup)).strip()
        text = re.sub(r'\n\s*\n+', '\n\n', text).strip()
        return text, title_tag
    except Exception:
        return html2text.html2text(html) if html else "", ""

def save_page_to_mongo(url: str, html: str, status_code: int, domain: str, bot_id: str, force: bool = False):
    text, title = extract_text_and_title(html)
    doc = {
        "url": url,
        "bot_id": bot_id,
        "domain": domain,
        "status_code": status_code,
        "fetched_at": datetime.utcnow(),
        "html": html,
        "text": text,
        "title": title
    }
    try:
        demo_pages_col.update_one(
            {"url": url, "bot_id": bot_id},
            {"$set": doc},
            upsert=True
        )
        return True
    except Exception as e:
        logging.warning(f"Mongo upsert failed for {url}: {e}")
        return False
    
def save_demopage_to_mongo(url: str, html: str, status_code: int, domain: str, bot_id: str, force: bool = False):
    text, title = extract_text_and_title(html)
    doc = {
        "url": url,
        "bot_id": bot_id,
        "domain": domain,
        "status_code": status_code,
        "fetched_at": datetime.utcnow(),
        "html": html,
        "text": text,
        "title": title
    }
    try:
        
        pages_col.update_one(
            {"url": url, "bot_id": bot_id},
            {"$set": doc},
            upsert=True
        )
        return True
    except Exception as e:
        logging.warning(f"Mongo upsert failed for {url}: {e}")
        return False

def crawl_site(start_url: str,
               dest_domain: str = None,
               max_pages: int = 200,
               render_js: bool = False,
               politeness: float = 1.0,
               force: bool = False,
               bot_id: str | None = None):
    """
    Crawl a site and save pages tagged with bot_id.
    """
    if not bot_id:
        raise ValueError("bot_id is required for crawl_site in multi-tenant mode")

    parsed = urlparse(start_url)
    domain = dest_domain or parsed.netloc
    rp = load_robots_txt(start_url)

    to_visit = deque([start_url])
    visited = set()
    saved = []

    sitemap_urls = get_sitemap_urls(start_url)
    if sitemap_urls:
        logging.info(f"Seeding from sitemap with {len(sitemap_urls)} entries")
        for s in sitemap_urls:
            if same_domain(s, domain):
                to_visit.append(s)

    while to_visit and len(visited) < max_pages:
        url = to_visit.popleft()
        if not url or url in visited:
            continue

        # try:
        #     if rp and not rp.can_fetch(USER_AGENT, url):
        #         logging.debug(f"Robots disallow: {url}")
        #         visited.add(url)
        #         continue
        # except Exception:
        #     pass

        if not same_domain(url, domain):
            visited.add(url)
            continue

        if not force:
            existing = pages_col.find_one({"url": url, "bot_id": bot_id}, {"fetched_at": 1})
            if existing:
                logging.debug(f"Skipping already-saved URL (use force=True to refetch): {url}")
                visited.add(url)
                continue

        logging.info(f"Crawling ({len(visited)+1}/{max_pages}) [bot={bot_id}]: {url}")
        html, status = fetch_html(url, render_js=render_js)
        if html is None:
            visited.add(url)
            time.sleep(politeness)
            continue

        ok = save_demopage_to_mongo(url, html, status or 200, domain, bot_id, force=force)
        if ok:
            saved.append(url)

        try:
            soup = BeautifulSoup(html, "html.parser")
            anchors = [a.get("href") for a in soup.find_all("a", href=True)]
            for link in anchors:
                norm = normalize_url(url, link)
                if norm and same_domain(norm, domain) and norm not in visited and norm not in to_visit:
                    if re.search(r"\.(jpg|jpeg|png|gif|svg|pdf|zip|rar|mp4|mp3|woff|woff2|ttf)$", norm, flags=re.IGNORECASE):
                        continue
                    to_visit.append(norm)
        except Exception:
            pass

        visited.add(url)
        time.sleep(politeness)

    logging.info(f"Crawl finished for bot_id={bot_id}. Visited: {len(visited)} pages, Saved: {len(saved)}")
    return saved

def demo_crawl_site(start_url: str,
               dest_domain: str = None,
               max_pages: int = 200,
               render_js: bool = False,
               politeness: float = 1.0,
               force: bool = False,
               bot_id: str | None = None):
    """
    Crawl a site and save pages tagged with bot_id.
    """
    if not bot_id:
        raise ValueError("bot_id is required for crawl_site in multi-tenant mode")

    parsed = urlparse(start_url)
    domain = dest_domain or parsed.netloc
    rp = load_robots_txt(start_url)

    to_visit = deque([start_url])
    visited = set()
    saved = []

    sitemap_urls = get_sitemap_urls(start_url)
    if sitemap_urls:
        logging.info(f"Seeding from sitemap with {len(sitemap_urls)} entries")
        for s in sitemap_urls:
            if same_domain(s, domain):
                to_visit.append(s)

    while to_visit and len(visited) < max_pages:
        url = to_visit.popleft()
        if not url or url in visited:
            continue

        # try:
        #     if rp and not rp.can_fetch(USER_AGENT, url):
        #         logging.debug(f"Robots disallow: {url}")
        #         visited.add(url)
        #         continue
        # except Exception:
        #     pass

        if not same_domain(url, domain):
            visited.add(url)
            continue

        if not force:
            existing = demo_pages_col.find_one({"url": url}, {"fetched_at": 1})
            if existing:
                logging.debug(f"Skipping already-saved URL (use force=True to refetch): {url}")
                visited.add(url)
                continue

        logging.info(f"Crawling ({len(visited)+1}/{max_pages}) [bot={bot_id}]: {url}")
        html, status = fetch_html(url, render_js=render_js)
        if html is None:
            visited.add(url)
            time.sleep(politeness)
            continue

        ok = save_demopage_to_mongo(url, html, status or 200, domain, bot_id, force=force)
        if ok:
            saved.append(url)

        try:
            soup = BeautifulSoup(html, "html.parser")
            anchors = [a.get("href") for a in soup.find_all("a", href=True)]
            for link in anchors:
                norm = normalize_url(url, link)
                if norm and same_domain(norm, domain) and norm not in visited and norm not in to_visit:
                    if re.search(r"\.(jpg|jpeg|png|gif|svg|pdf|zip|rar|mp4|mp3|woff|woff2|ttf)$", norm, flags=re.IGNORECASE):
                        continue
                    to_visit.append(norm)
        except Exception:
            pass

        visited.add(url)
        time.sleep(politeness)

    logging.info(f"Crawl finished for bot_id={bot_id}. Visited: {len(visited)} pages, Saved: {len(saved)}")
    return saved

def main():
    parser = argparse.ArgumentParser(description="Site crawler that upserts pages into MongoDB (per-bot).")
    parser.add_argument("--start", "-s", required=True, help="Start (seed) URL, e.g. https://example.com")
    parser.add_argument("--bot-id", "-b", required=True, help="Bot ID to tag pages with")
    parser.add_argument("--max-pages", "-m", type=int, default=100000, help="Max pages to crawl")
    parser.add_argument("--render-js", action="store_true", help="Attempt JS rendering")
    parser.add_argument("--politeness", type=float, default=1.0, help="Delay between requests in seconds")
    parser.add_argument("--force", action="store_true", help="Refetch and overwrite pages even if present in Mongo")
    parser.add_argument("--domain", help="Optional: force a specific domain (netloc) to restrict crawling")
    args = parser.parse_args()

    if args.render_js and not REQUESTS_HTML_AVAILABLE:
        logging.warning("requests_html not available; --render-js ignored.")
        args.render_js = False

    # saved = crawl_site(
    #     start_url=args.start,
    #     dest_domain=args.domain,
    #     max_pages=args.max_pages,
    #     render_js=args.render_js,
    #     politeness=args.politeness,
    #     force=args.force,
    #     bot_id=args.bot_id
    # )

    saved_demo = demo_crawl_site(
        start_url=args.start,
        dest_domain=args.domain,
        max_pages=args.max_pages,
        render_js=args.render_js,
        politeness=args.politeness,
        force=args.force,
        bot_id=args.bot_id
    )
    logging.info(f"Saved {len(saved_demo)} pages for bot {args.bot_id}. Example: {saved_demo[:10]}")
    
if __name__ == "__main__":
    main()



# """
# Improved site crawler / scraper that upserts pages into MongoDB.
# Uses Selenium instead of BS4/requests for JS-heavy websites.
# Logic remains the same.
# """

# import time
# import logging
# import argparse
# import re
# from collections import deque
# from urllib.parse import urlparse, urljoin, urldefrag
# from datetime import datetime

# import html2text
# import urllib.robotparser

# from pymongo import MongoClient, ASCENDING

# # -------- Selenium --------
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.common.exceptions import WebDriverException
# from webdriver_manager.chrome import ChromeDriverManager

# logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# USER_AGENT = "Mozilla/5.0 (compatible; CustomCrawler/1.0; +https://example.com/bot)"
# POLITE_DELAY = 1.0
# # MONGO_URI = "mongodb://localhost:27017/"
# MONGO_URI = "mongodb+srv://rajathda:Sp99l0ZXuszijwar@cluster0.x9apgd4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# MONGO_DBNAME = "chat_bot_db"
# MONGO_COLLECTION = "pages"

# client = MongoClient(MONGO_URI)
# db = client[MONGO_DBNAME]
# pages_col = db[MONGO_COLLECTION]

# try:
#     pages_col.create_index([("url", ASCENDING), ("bot_id", ASCENDING)], unique=True)
# except Exception:
#     pass

# _html2text = html2text.HTML2Text()
# _html2text.ignore_links = True
# _html2text.body_width = 0


# # ----------------- Helpers -----------------
# def normalize_url(base: str, link: str) -> str:
#     if not link:
#         return None
#     joined = urljoin(base, link)
#     clean, _ = urldefrag(joined)
#     return clean.strip()


# def same_domain(url: str, domain: str) -> bool:
#     try:
#         if not domain:
#             return True
#         return urlparse(url).netloc.endswith(domain)
#     except Exception:
#         return False


# def load_robots_txt(start_url: str):
#     parsed = urlparse(start_url)
#     robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
#     rp = urllib.robotparser.RobotFileParser()
#     try:
#         rp.set_url(robots_url)
#         rp.read()
#     except Exception:
#         pass
#     return rp


# # ----------------- Selenium -----------------
# # def init_driver():
# #     opts = Options()
# #     opts.add_argument("--headless=new")
# #     opts.add_argument("--disable-gpu")
# #     opts.add_argument("--no-sandbox")
# #     opts.add_argument(f"user-agent={USER_AGENT}")
# #     driver = webdriver.Chrome(
# #         service=webdriver.chrome.service.Service(ChromeDriverManager().install()),
# #         options=opts
# #     )
# #     driver.set_page_load_timeout(30)
# #     return driver

# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options

# def init_driver():
#     opts = Options()
#     opts.add_argument("--headless=new")
#     opts.add_argument("--no-sandbox")
#     opts.add_argument("--disable-dev-shm-usage")
#     opts.add_argument("--disable-gpu")
#     opts.add_argument("--window-size=1920,1080")
#     opts.add_argument(f"user-agent={USER_AGENT}")
#     opts.binary_location = "/usr/bin/chromium"   # 👈 chromium path in Docker

#     # Use the chromedriver that webdriver-manager downloaded (still fine)
#     driver = webdriver.Chrome(options=opts)
#     driver.set_page_load_timeout(30)

#     return driver


# def fetch_html_and_links(driver, url: str):
#     try:
#         driver.get(url)
#         time.sleep(2)  # allow JS to load
#         html = driver.page_source
#         anchors = driver.find_elements(By.XPATH, "//a[@href]")
#         links = [a.get_attribute("href") for a in anchors]
#         return html, links
#     except WebDriverException as e:
#         logging.debug(f"Selenium failed for {url}: {e}")
#         return None, []


# # ----------------- Content extraction -----------------
# def extract_text_and_title(html: str):
#     if not html:
#         return "", ""
#     text = _html2text.handle(html).strip()
#     text = re.sub(r'\n\s*\n+', '\n\n', text)
#     title_match = re.search(r"<title>(.*?)</title>", html, re.I | re.S)
#     title = title_match.group(1).strip() if title_match else ""
#     return text, title


# def save_page_to_mongo(url, html, domain, bot_id):
#     text, title = extract_text_and_title(html)
#     doc = {
#         "url": url,
#         "bot_id": bot_id,
#         "domain": domain,
#         "status_code": 200,
#         "fetched_at": datetime.utcnow(),
#         "html": html,
#         "text": text,
#         "title": title
#     }
#     pages_col.update_one(
#         {"url": url, "bot_id": bot_id},
#         {"$set": doc},
#         upsert=True
#     )


# # ----------------- Main crawler -----------------
# def crawl_site(start_url: str,
#                dest_domain: str = None,
#                max_pages: int = 200,
#                render_js: bool = False,   # 👈 REQUIRED
#                politeness: float = 1.0,
#                force: bool = False,
#                bot_id: str | None = None):


#     if not bot_id:
#         raise ValueError("bot_id is required")

#     parsed = urlparse(start_url)
#     domain = dest_domain or parsed.netloc

#     driver = init_driver()

#     to_visit = deque([start_url])
#     visited = set()
#     saved = []

#     while to_visit and len(visited) < max_pages:
#         url = to_visit.popleft()
#         if not url or url in visited:
#             continue

#         if not same_domain(url, domain):
#             visited.add(url)
#             continue

#         if not force:
#             if pages_col.find_one({"url": url, "bot_id": bot_id}):
#                 visited.add(url)
#                 continue

#         logging.info(f"Crawling ({len(visited)+1}/{max_pages}) [bot={bot_id}]: {url}")

#         html, links = fetch_html_and_links(driver, url)
#         if not html:
#             visited.add(url)
#             continue

#         save_page_to_mongo(url, html, domain, bot_id)
#         saved.append(url)

#         for link in links:
#             norm = normalize_url(url, link)
#             if not norm:
#                 continue
#             if not same_domain(norm, domain):
#                 continue
#             if norm in visited or norm in to_visit:
#                 continue
#             if re.search(r"\.(jpg|png|pdf|zip|mp4|mp3|woff|ttf)$", norm, re.I):
#                 continue
#             to_visit.append(norm)

#         visited.add(url)
#         time.sleep(politeness)

#     driver.quit()
#     logging.info(f"Crawl finished for bot_id={bot_id}. Visited: {len(visited)} pages, Saved: {len(saved)}")
#     return saved


# # ----------------- CLI -----------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--start", required=True)
#     parser.add_argument("--bot-id", required=True)
#     parser.add_argument("--max-pages", type=int, default=200)
#     parser.add_argument("--force", action="store_true")
#     parser.add_argument("--domain")
#     args = parser.parse_args()

#     crawl_site(
#         start_url=args.start,
#         dest_domain=args.domain,
#         max_pages=args.max_pages,
#         force=args.force,
#         bot_id=args.bot_id
#     )


# # if __name__ == "__main__":
# #     main()
