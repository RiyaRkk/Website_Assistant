from flask import Flask, render_template, jsonify, request
from pathlib import Path
import os, re, json, time, uuid
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
# import google.genai as genai
from pymongo import MongoClient
import gridfs
from datetime import datetime
from temppy.collector import record_low_confidence
from llama_index.core.prompts import PromptTemplate
from io import BytesIO
import jwt
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = "fyeyueguygxiu"
# ---------- CONFIG ----------
MONGO_URI=os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["chat_bot_db"]
app.config["SECRET_KEY"] = "gygjhxksbxksdxkjsd"

chat_collection = db["chats"]              # logging of Q/A
conversations = db["conversations"]        # persistent memory per session+bot
bots_collection = db["bots"]               # bot definitions
fs = gridfs.GridFS(db)

OUTDIR = Path("embedded")
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 10
GEMINI_MODEL = "gemini-2.5-flash"

# 👉 DO NOT hardcode keys in code – use env var
api_key= os.getenv("GEMINI_API_KEY")

MAX_HISTORY_TURNS = 120
MAX_HISTORY_CHARS = 35000
MAX_CONTEXT_CHARS = 50000

CONFIDENCE_THRESHOLD = 0.35

# ---------- EMBEDDER ----------
embedder = SentenceTransformer(MODEL_NAME)

def embed_query(q: str) -> np.ndarray:
    vec = embedder.encode([q], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vec)
    return vec

# ---------- helpers ----------
def _row_text(meta_obj):
    if isinstance(meta_obj, dict):
        return meta_obj.get("text") or meta_obj.get("preview") or meta_obj.get("source_title") or ""
    try:
        return str(meta_obj)
    except Exception:
        return ""

def load_embeddings_from_bundle(mongo_uri: str,
                                mongo_db: str,
                                bundle_coll: str = "bundles",
                                bundle_id: str = "all_embeddings"):
    """
    Load embeddings, ids and meta from a single-bundle document stored in MongoDB.
    Returns dict: ok, EMB_MATRIX, IDS, META, CORPUS_TEXTS, error
    """
    client_local = None
    try:
        client_local = MongoClient(mongo_uri)
        db_local = client_local[mongo_db]
        coll = db_local[bundle_coll]

        doc = coll.find_one({"_id": bundle_id})
        if not doc:
            return {"ok": False, "error": f"bundle {bundle_id} not found in {mongo_db}.{bundle_coll}"}

        data_bytes = None

        if "bundle_bytes" in doc and doc.get("bundle_bytes") is not None:
            data_bytes = bytes(doc["bundle_bytes"])
        elif doc.get("gridfs_id"):
            try:
                fs_local = gridfs.GridFS(db_local)
                gridfs_id = doc["gridfs_id"]
                f = fs_local.get(gridfs_id)
                data_bytes = f.read()
            except Exception as e:
                return {"ok": False, "error": f"Error reading GridFS id {doc.get('gridfs_id')}: {e}"}
        else:
            return {"ok": False, "error": "No bundle_bytes or gridfs_id found in bundle document"}

        if not data_bytes:
            return {"ok": False, "error": "Bundle contained no bytes"}

        try:
            bio = BytesIO(data_bytes)
            npz = np.load(bio, allow_pickle=True)
        except Exception as e:
            return {"ok": False, "error": f"Failed to load npz from bytes: {e}"}

        if "embeddings" not in npz:
            return {"ok": False, "error": "npz missing 'embeddings' array"}
        embeddings = np.asarray(npz["embeddings"], dtype="float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        ids = []
        meta = []
        if "ids" in npz:
            try:
                raw_ids = npz["ids"]
                ids = [str(x) for x in raw_ids.tolist()]
            except Exception:
                ids = [str(x) for x in npz["ids"]]
        if "meta" in npz:
            raw_meta = npz["meta"]
            try:
                meta = raw_meta.tolist()
            except Exception:
                meta = [m for m in raw_meta]

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        EMB_MATRIX = embeddings / norms

        META_FROM_MONGO = meta
        IDS_FROM_MONGO = ids[: EMB_MATRIX.shape[0]]
        CORPUS_TEXTS = [_row_text(m) for m in META_FROM_MONGO[: EMB_MATRIX.shape[0]]]

        return {
            "ok": True,
            "EMB_MATRIX": EMB_MATRIX,
            "IDS": IDS_FROM_MONGO,
            "META": META_FROM_MONGO,
            "CORPUS_TEXTS": CORPUS_TEXTS
        }

    except Exception as e:
        return {"ok": False, "error": f"unexpected error: {e}"}
    finally:
        try:
            if client_local:
                client_local.close()
        except Exception:
            pass

# ---------- PER-BOT EMBEDDING CACHE ----------
BOT_EMB_CACHE = {}   # bot_id -> dict(EMB_MATRIX, IDS, META, CORPUS_TEXTS)

def ensure_bot_embeddings_loaded(bot_id: str):
    """
    Load embeddings for a bot_id from Mongo 'bundles' (bundle_id=f'emb_{bot_id}')
    and cache them in memory.
    """
    if bot_id in BOT_EMB_CACHE:
        return BOT_EMB_CACHE[bot_id]

    bundle_id = f"emb_{bot_id}"
    res = load_embeddings_from_bundle(MONGO_URI, db.name, bundle_coll="bundles", bundle_id=bundle_id)
    if not res.get("ok"):
        raise RuntimeError(f"No embeddings found for this bot (bot_id={bot_id}): {res.get('error')}")

    BOT_EMB_CACHE[bot_id] = {
        "EMB_MATRIX": res["EMB_MATRIX"],
        "IDS": res["IDS"],
        "META": res["META"],
        "CORPUS_TEXTS": res["CORPUS_TEXTS"],
    }
    return BOT_EMB_CACHE[bot_id]

# ---------- RETRIEVAL ----------
def retrieve_context(bot_id: str, q: str, k: int = TOP_K):
    """
    Retrieve top-k chunks for a given bot_id using its embeddings.
    """
    bot_data = ensure_bot_embeddings_loaded(bot_id)
    EMB_MATRIX = bot_data["EMB_MATRIX"]
    IDS_FROM_MONGO = bot_data["IDS"]
    CORPUS_TEXTS = bot_data["CORPUS_TEXTS"]

    vec = embed_query(q)
    qv = vec[0].astype("float32")
    sims = EMB_MATRIX.dot(qv)

    results = []
    if k >= sims.shape[0]:
        top_idx = np.argsort(-sims)
    else:
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

    for idx in top_idx:
        score = float(sims[idx])
        text = CORPUS_TEXTS[idx]
        results.append({"id": IDS_FROM_MONGO[idx], "score": score, "text": text, "idx": int(idx)})

    return results

def build_context(results, max_chars: int = MAX_CONTEXT_CHARS):
    blocks = []
    total = 0
    seen = set()
    for r in results:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        block = f"[{r['id']}] {r['text']}"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)

# ---------- MEMORY ----------
def _now_ts():
    return datetime.utcnow().isoformat()

def create_session_id():
    return str(uuid.uuid4())  
def get_history(session_id: str, bot_id: str):
    doc = conversations.find_one({"session_id": session_id, "bot_id": bot_id})
    if not doc:
        return []
    return doc.get("history", [])

def append_history(session_id: str, bot_id: str, role: str, content: str):
    now = _now_ts()
    conversations.update_one(
        {"session_id": session_id, "bot_id": bot_id},
        {
            "$setOnInsert": {"session_id": session_id, "bot_id": bot_id},
            "$push": {"history": {"role": role, "content": content, "ts": now}}
        },
        upsert=True
    )
    conversations.update_one(
        {"session_id": session_id, "bot_id": bot_id},
        {"$push": {"history": {"$each": [], "$slice": -MAX_HISTORY_TURNS}}}
    )

def trim_history_for_prompt(history):
    if not history:
        return history
    total = sum(len(h.get("content", "")) for h in history)
    if total <= MAX_HISTORY_CHARS and len(history) <= MAX_HISTORY_TURNS:
        return history[-MAX_HISTORY_TURNS:]
    trimmed = []
    total = 0
    for h in reversed(history):
        c = h.get("content") or ""
        if total + len(c) > MAX_HISTORY_CHARS and len(trimmed) >= 1:
            break
        trimmed.append(h)
        total += len(c)
        if len(trimmed) >= MAX_HISTORY_TURNS:
            break
    return list(reversed(trimmed))

def get_previous_user_question_from_history(history):
    if not history:
        return ""
    for h in reversed(history):
        if h.get("role") == "user" and h.get("content") and h.get("content").strip():
            return h.get("content").strip()
    return ""

def is_greeting(text: str):
    if not text:
        return False
    t = text.lower().strip()
    GREETINGS = {"hi", "hello", "hey", "hii", "hii!", "hiya",
                 "good morning", "good afternoon", "good evening"}
    return t in GREETINGS

# ---------- Prompt Template ----------
qa_template = PromptTemplate(
"""
You are an assistant trained strictly on the provided context.

SYSTEM RULES:
- If a question is outside scope, politely decline.
- Give detailed information answer.
- Before answering, think step-by-step.
- Tone: friendly, warm, professional, empathetic.
- Start friendly and name the user if known.
- Verify with one short question only when truly required.
- Never tell the user to wait or that you will do something later.
- Behave as humanly as possible. 
- Once greeted, do not repeat greetings. 

CONTEXT:
{context_str}

QUESTION:
{query_str}

ANSWER:
"""
)

def ask_gemini_with_history(question: str, context: str, history: list, previous_question: str):
    hist_trimmed = trim_history_for_prompt(history)
    hist_blocks = []
    for h in hist_trimmed:
        role = h.get("role", "user")
        hist_blocks.append(f"{role.upper()}: {h.get('content')}")
    history_text = "\n".join(hist_blocks) if hist_blocks else "(no prior conversation)"

    prev_q_text = previous_question.strip() if previous_question else "(none)"

    context_str = (
        f"Previous user question: {prev_q_text}\n\n"
        f"History summary:\n{history_text}\n\n"
        f"Retrieved context:\n{context}"
    )

    prompt = qa_template.format(context_str=context_str, query_str=question)

    r = genai.GenerativeModel(GEMINI_MODEL).generate_content(prompt)
    txt = (getattr(r, "text", "") or "").strip()

    m = re.search(r"\{.*\}", txt, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0)), txt
        except Exception:
            pass

    used = sorted(set(re.findall(r"\[([^\]]+)\]", txt)))
    return {"answer": txt, "sources": used}, txt

# ---------- ROUTES ----------
@app.route("/")
def home():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))
    # generic entry; frontend will inspect URL/query to decide mode
    return render_template("landing_page.html")

@app.route("/admin/dashboard")
def admin_dashboard():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))
    data=list(bots_collection.find())
    return render_template("admin_dashboard.html",data=data)

@app.route("/bot/<bot_id>")
def bot_page(bot_id):
    """
    Public-facing URL for a specific bot.
    We serve the same chat.html - frontend reads bot_id from the path.
    """
    bot = bots_collection.find_one({"bot_id":bot_id})
    name = bot.get("name")
    tagline = bot.get("tagline")
    return render_template("chat.html",name =name,tagline = tagline)




# ---- BOT MANAGEMENT ----
def create_bot_record(start_url: str, name: str, tagline:str | None = None):
    bot_id = str(uuid.uuid4())
    bot_doc = {
        "bot_id": bot_id,
        "name": name or f"Bot for {start_url}",
        "start_url": start_url,
        "created_at": datetime.utcnow(),
        "tagline":tagline,
        "status": "created"
    }
    bots_collection.insert_one(bot_doc)
    return bot_doc

from scrapper import crawl_site
from chunker import run_chunking_for_bot
from vec_emb import run_embedding_for_bot
from flask import Flask, render_template, jsonify, request, url_for

@app.route("/create-bot", methods=["POST"])
def create_bot():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))
    """
    Client-facing endpoint:
      body: { "start_url": "...", "name": "Optional", "max_pages": int, "render_js": bool, "force": bool }
    Runs crawl -> chunk -> embedding synchronously for that URL.
    """
    body = request.json or {}
    start_url = body.get("start_url")
    if not start_url:
        return jsonify({"error": "start_url is required"}), 400

    name = body.get("name")
    tagline = body.get("tagline")
    max_pages = int(body.get("max_pages", 1000000))
    render_js = bool(body.get("render_js", False))
    force = bool(body.get("force", False))

    bot_doc = create_bot_record(start_url, name, tagline)
    bot_id = bot_doc["bot_id"]

    # UPDATE status
    bots_collection.update_one({"bot_id": bot_id}, {"$set": {"status": "crawling"}})

    # 1) crawl pages for this bot
    saved_pages = crawl_site(
        start_url=start_url,
        dest_domain=None,
        max_pages=max_pages,
        render_js=render_js,
        politeness=1.0,
        force=force,
        bot_id=bot_id
    )

    # 2) chunk
    bots_collection.update_one({"bot_id": bot_id}, {"$set": {"status": "chunking"}})
    run_chunking_for_bot(bot_id)

    # 3) embed
    bots_collection.update_one({"bot_id": bot_id}, {"$set": {"status": "embedding"}})
    run_embedding_for_bot(bot_id)

    bots_collection.update_one(
        {"bot_id": bot_id},
        {"$set": {"status": "ready", "last_trained_at": datetime.utcnow()}}
    )

    # clear cached embeddings if any
    BOT_EMB_CACHE.pop(bot_id, None)

    # 🔗 generate shareable link for this bot (pretty URL: /bot/<bot_id>)
    share_url = url_for("bot_page", bot_id=bot_id, _external=True)

    return jsonify({
        "status": "ok",
        "bot_id": bot_id,
        "saved_pages": len(saved_pages),
        "message": "Bot created and trained successfully.",
        "share_url": share_url
    })

@app.route("/bots/<bot_id>", methods=["GET"])
def get_bot(bot_id):
    bot = bots_collection.find_one({"bot_id": bot_id}, {"_id": 0})
    if not bot:
        return jsonify({"error": "Bot not found"}), 404
    return jsonify(bot)

# ---- CHAT ----
@app.route("/chat", methods=["POST"])
def chat():
    payload = request.json or {}
    qa_input = payload.get("message", "")
    session_id = payload.get("session_id")
    bot_id = payload.get("bot_id")  # 🔥 REQUIRED for multi-bot

    if not bot_id:
        return jsonify({"error": "bot_id is required"}), 400

    bot = bots_collection.find_one({"bot_id": bot_id})
    if not bot:
        return jsonify({"error": "Invalid bot_id"}), 404

    if not session_id:
        session_id = create_session_id()
    if not qa_input or not qa_input.strip():
        return jsonify({"response": "Please ask a question.", "session_id": session_id, "bot_id": bot_id})

    if is_greeting(qa_input):
        answer = "Hi, how are you? How can I help you today?"
        append_history(session_id, bot_id, "user", qa_input)
        append_history(session_id, bot_id, "assistant", answer)
        chat_collection.insert_one({
            "session_id": session_id,
            "bot_id": bot_id,
            "FAQ": qa_input,
            "answer": answer,
            "timestamp": datetime.utcnow()
        })
        return jsonify({"response": answer, "session_id": session_id, "bot_id": bot_id})

    # 1) history for this bot+session
    history = get_history(session_id, bot_id)
    previous_user_question = get_previous_user_question_from_history(history)

    # 2) append current user message
    append_history(session_id, bot_id, "user", qa_input)
    history = get_history(session_id, bot_id)

    # 3) retrieval for this bot
    try:
        retrieved = retrieve_context(bot_id, qa_input)
        context = build_context(retrieved, max_chars=MAX_CONTEXT_CHARS)
    except Exception as e:
        context = ""
        retrieved = []
        print("Retrieval error:", e)

    # 4) LLM answer
    try:
        answer_obj, raw_text = ask_gemini_with_history(qa_input, context, history, previous_user_question)
        answer = answer_obj.get("answer") if isinstance(answer_obj, dict) else str(raw_text)
        if not answer:
            answer = raw_text
    except Exception as e:
        answer = f"Error: {str(e)}"
        answer_obj = {"answer": answer, "sources": []}

    # 5) confidence logic
    try:
        top_score = max((r.get("score", 0.0) for r in retrieved), default=0.0)
        scores_list = sorted([r.get("score", 0.0) for r in retrieved], reverse=True)[:3]
        avg_top3 = float(np.mean(scores_list)) if scores_list else 0.0
    except Exception:
        top_score = 0.0
        avg_top3 = 0.0

    normalized_answer = (answer or "").strip()
    sources = []
    if isinstance(answer_obj, dict):
        sources = answer_obj.get("sources", []) or []

    low_confidence = False
    if normalized_answer.startswith("I don't know based on the provided information.") or normalized_answer.startswith("I don't have an answer"):
        low_confidence = True
    else:
        if sources:
            low_confidence = False
        else:
            if top_score >= CONFIDENCE_THRESHOLD or avg_top3 >= CONFIDENCE_THRESHOLD:
                low_confidence = False
            else:
                if len(normalized_answer) < 60:
                    low_confidence = True
                else:
                    low_confidence = False

    if low_confidence:
        record_low_confidence(qa_input, normalized_answer, session_id)
        contact_msg = (
            f"\n\nI couldn't find a confident answer in my knowledge base. "
            f"For official help, please feel free to get connected with an official."
            " They can provide authoritative assistance."
        )
        if normalized_answer.startswith("I don't know based on the provided information.") or normalized_answer.startswith("I don't have an answer"):
            answer = (
                "I don't know based on the provided information."
                + " If you need an authoritative answer, please feel free to get connected with an official."
            )
        else:
            answer = (answer or "").rstrip() + contact_msg

        if isinstance(answer_obj, dict):
            answer_obj["low_confidence"] = True
            answer_obj["top_retrieval_score"] = top_score
            answer_obj["avg_top3_retrieval_score"] = avg_top3
    else:
        if isinstance(answer_obj, dict):
            answer_obj["low_confidence"] = False
            answer_obj["top_retrieval_score"] = top_score
            answer_obj["avg_top3_retrieval_score"] = avg_top3

    # 6) save assistant message
    append_history(session_id, bot_id, "assistant", answer)

    # 7) log
    chat_collection.insert_one({
        "session_id": session_id,
        "bot_id": bot_id,
        "FAQ": qa_input,
        "answer": answer,
        "timestamp": datetime.utcnow()
    })

    resp = {"response": answer, "session_id": session_id, "bot_id": bot_id}
    if isinstance(answer_obj, dict):
        resp["sources"] = answer_obj.get("sources", [])
        resp["low_confidence"] = answer_obj.get("low_confidence", False)
        resp["top_retrieval_score"] = answer_obj.get("top_retrieval_score", 0.0)
        resp["avg_top3_retrieval_score"] = answer_obj.get("avg_top3_retrieval_score", 0.0)
        if answer_obj.get("assumptions"):
            resp["assumptions"] = answer_obj["assumptions"]
        if answer_obj.get("missing_info"):
            resp["missing_info"] = answer_obj["missing_info"]

    return jsonify(resp)

# ---- ADMIN PIPELINE ENDPOINTS (per-bot) ----
# @app.route("/run-crawler", methods=["POST"])
# def run_crawler():
#     if not session.get("admin_logged_in"):
#         return redirect(url_for("admin_login"))    
#     body = request.json or {}
#     start_url = body.get("start")
#     bot_id = body.get("bot_id")

#     if not start_url:
#         return jsonify({"error": "start URL is required"}), 400
#     if not bot_id:
#         return jsonify({"error": "bot_id is required"}), 400

#     max_pages = body.get("max_pages", 200)
#     force = body.get("force", False)
#     render_js = body.get("render_js", False)

#     saved = crawl_site(
#         start_url=start_url,
#         max_pages=int(max_pages),
#         force=bool(force),
#         render_js=bool(render_js),
#         bot_id=bot_id,
#         dest_domain=None,
#         politeness=1.0
#     )

#     BOT_EMB_CACHE.pop(bot_id, None)
#     return jsonify({"status": "ok", "saved_pages": saved})

# @app.route("/run-embedding", methods=["POST"])
# def run_embedding_api():
#     if not session.get("admin_logged_in"):
#         return redirect(url_for("admin_login"))
#     body = request.json or {}
#     bot_id = body.get("bot_id")
#     if not bot_id:
#         return jsonify({"error": "bot_id is required"}), 400
#     try:
#         run_embedding_for_bot(bot_id)
#         BOT_EMB_CACHE.pop(bot_id, None)
#         return jsonify({"status": "ok"}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route("/run-chunking", methods=["POST"])
# def run_chunking_api():
#     if not session.get("admin_logged_in"):
#         return redirect(url_for("admin_login"))
#     body = request.json or {}
#     bot_id = body.get("bot_id")
#     if not bot_id:
#         return jsonify({"error": "bot_id is required"}), 400
#     try:
#         run_chunking_for_bot(bot_id)
#         BOT_EMB_CACHE.pop(bot_id, None)
#         return jsonify({"status": "ok"}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500



from flask import render_template, request, redirect, url_for, session, flash

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # 🔒 Hardcoded admin credentials (can move to env/db later)
        if username == "admin" and password == "admin123":
            session["admin_logged_in"] = True
            return redirect(url_for("admin_pipeline"))
        else:
            flash("Invalid credentials", "error")

    return render_template("login.html")



@app.route("/admin/pipeline")
def admin_pipeline():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))

    return render_template("chat.html")



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
