import os
import re
import json
import sys
from pathlib import Path
from hashlib import sha1
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError
# MONGO_URI = "mongodb://localhost:27017/"

MONGO_URI = os.getenv("MONGO_URI")

MONGO_DBNAME = "chat_bot_db"
SOURCE_COLLECTION = "pages"
TARGET_COLLECTION = "chunks"
DEMO_COLLECTION = "demo_chunks"

OUT_JSONL = "chunks_dedup.jsonl"

TARGET_WORDS = 500
OVERLAP_WORDS = 100

SIMHASH_BITS = 64
HAMMING_THRESHOLD = 3
BUCKET_PREFIX_BITS = 16

LIMIT_SOURCE_DOCS = None

_ws_re = re.compile(r'\s+')

def normalize_whitespace(text: str) -> str:
    return _ws_re.sub(' ', text).strip()

def words(text: str):
    if not text:
        return []
    return text.split()

def _tokenize_for_simhash(text: str):
    t = text.lower()
    t = re.sub(r'[^\w\s]', ' ', t)
    tokens = [tok for tok in t.split() if len(tok) > 1]
    return tokens if tokens else [""]

def _int_hash(x: str) -> int:
    from hashlib import sha1 as _sha1
    h = _sha1(x.encode('utf-8')).hexdigest()
    return int(h, 16) & ((1 << SIMHASH_BITS) - 1)

def simhash(text: str) -> int:
    v = [0] * SIMHASH_BITS
    tokens = _tokenize_for_simhash(text)
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    for token, weight in freq.items():
        h = _int_hash(token)
        for i in range(SIMHASH_BITS):
            bit = (h >> i) & 1
            v[i] += weight if bit else -weight
    fingerprint = 0
    for i in range(SIMHASH_BITS):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint

def hamming_distance(a: int, b: int) -> int:
    x = a ^ b
    return x.bit_count()

def simhash_bucket_prefix(h: int, prefix_bits: int):
    return h & ((1 << prefix_bits) - 1)

def split_paragraphs_keep_headings(md: str):
    parts = re.split(r'\n{2,}', md)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out.append(p)
    return out

def make_chunks_from_paragraphs(paragraphs, target_words, overlap_words):
    chunks = []
    buffer_words = []
    buffer_texts = []
    buffer_word_count = 0

    def flush_buffer():
        nonlocal buffer_words, buffer_texts, buffer_word_count
        if buffer_word_count == 0:
            return
        chunk_text = ' '.join(buffer_texts)
        chunks.append(chunk_text)
        if overlap_words > 0:
            last = buffer_words[-overlap_words:] if len(buffer_words) >= overlap_words else buffer_words[:]
            last_text = ' '.join(last)
            buffer_words = last
            buffer_texts = [last_text] if last_text else []
            buffer_word_count = len(buffer_words)
        else:
            buffer_words = []
            buffer_texts = []
            buffer_word_count = 0

    for p in paragraphs:
        p_norm = normalize_whitespace(p)
        p_words = words(p_norm)
        p_count = len(p_words)
        if p_count == 0:
            continue

        if p_count >= target_words:
            flush_buffer()
            start = 0
            while start < p_count:
                end = min(start + target_words, p_count)
                chunk_text = ' '.join(p_words[start:end])
                chunks.append(chunk_text)
                if end == p_count:
                    last = p_words[max(0, end - overlap_words):end]
                    buffer_words = last
                    buffer_texts = [' '.join(last)] if last else []
                    buffer_word_count = len(buffer_words)
                    break
                start = end - overlap_words
            continue

        if buffer_word_count + p_count <= target_words:
            buffer_words.extend(p_words)
            buffer_texts.append(p_norm)
            buffer_word_count += p_count
        else:
            flush_buffer()
            buffer_words = p_words[:]
            buffer_texts = [p_norm]
            buffer_word_count = p_count

    flush_buffer()
    return chunks

def get_mongo_client(uri: str):
    client = MongoClient(uri)
    return client

def get_collection(client: MongoClient, dbname: str, collname: str):
    db = client[dbname]
    return db[collname]

def ensure_index_on_target(coll):
    try:
        coll.create_index([("meta.id", ASCENDING)], unique=True)
        coll.create_index([("bot_id", ASCENDING), ("meta.id", ASCENDING)])
    except Exception as e:
        print("Warning: could not create unique index on meta.id:", e, file=sys.stderr)

def extract_text_from_doc(doc: dict, candidates=None) -> Optional[str]:
    if candidates is None:
        candidates = ["content", "text", "body", "markdown", "raw", "html"]
    for f in candidates:
        v = doc.get(f)
        if isinstance(v, str) and v.strip():
            return v
    pieces = []
    for f in ["title", "summary", "excerpt"]:
        if isinstance(doc.get(f), str):
            pieces.append(doc.get(f))
    if pieces:
        return "\n\n".join(pieces)
    return None

def process_from_mongo(client_uri, dbname, source_collname, target_collname, out_jsonl=None, limit=None, bot_id: str | None = None):
    if not bot_id:
        raise ValueError("bot_id is required for process_from_mongo in multi-tenant mode")

    client = get_mongo_client(client_uri)
    src_coll = get_collection(client, dbname, source_collname)
    tgt_coll = get_collection(client, dbname, target_collname)
    ensure_index_on_target(tgt_coll)

    seen_exact = set()
    seen_simhash_buckets = defaultdict(list)

    total_source_docs = 0
    total_paragraphs = 0
    kept_paragraphs = 0
    total_chunks = 0
    inserted_count = 0
    updated_count = 0
    failed_count = 0

    fh = None
    if out_jsonl:
        fh = open(out_jsonl, "w", encoding="utf-8")

    session = None
    cursor = None
    try:
        session = client.start_session()
        filter_query = {"bot_id": bot_id}

        is_cloud = "mongodb+srv://" in client_uri.lower()
        cursor = src_coll.find(
            filter_query,
            session=session,
            no_cursor_timeout=not is_cloud
            )

        if limit:
            cursor = cursor.limit(limit)

        for doc in cursor:
            total_source_docs += 1
            doc_id = str(doc.get("_id"))
            title = doc.get("title") or doc.get("name") or None

            text_content = extract_text_from_doc(doc)
            if not text_content:
                print(f"Skipping source doc {doc_id}: no suitable text field found.")
                continue

            paragraphs = split_paragraphs_keep_headings(text_content)
            total_paragraphs += len(paragraphs)

            kept = []
            for p in paragraphs:
                p_norm = normalize_whitespace(p)
                if not p_norm:
                    continue

                k = sha1(p_norm.encode('utf-8')).hexdigest()
                if k in seen_exact:
                    continue

                h = simhash(p_norm)
                prefix = simhash_bucket_prefix(h, BUCKET_PREFIX_BITS)
                found_near = False
                for h2 in seen_simhash_buckets[prefix]:
                    if hamming_distance(h, h2) <= HAMMING_THRESHOLD:
                        found_near = True
                        break
                if found_near:
                    continue

                seen_exact.add(k)
                seen_simhash_buckets[prefix].append(h)
                kept.append(p_norm)

            kept_paragraphs += len(kept)

            chunks = make_chunks_from_paragraphs(kept, TARGET_WORDS, OVERLAP_WORDS)
            for idx, text in enumerate(chunks):
                raw_id = f"{doc_id}::chunk::{idx}::bot::{bot_id}"
                chunk_id = sha1(raw_id.encode("utf-8")).hexdigest()

                inserted_at_dt = datetime.now(timezone.utc)

                meta = {
                    "id": chunk_id,
                    "bot_id": bot_id,
                    "source_id": str(doc.get("_id")),
                    "source_collection": source_collname,
                    "source_title": title,
                    "chunk_index": idx,
                    "word_count": len(words(text)),
                    "preview": text[:200]
                }

                out = {
                    "bot_id": bot_id,
                    "text": text,
                    "meta": meta,
                    "inserted_at": inserted_at_dt
                }

                if fh is not None:
                    out_for_json = {
                        "bot_id": bot_id,
                        "text": out["text"],
                        "meta": out["meta"],
                        "inserted_at": inserted_at_dt.isoformat()
                    }
                    fh.write(json.dumps(out_for_json, ensure_ascii=False) + "\n")

                try:
                    result = tgt_coll.update_one(
                        {"meta.id": meta["id"], "bot_id": bot_id},
                        {"$set": out},
                        upsert=True
                    )
                    if getattr(result, "upserted_id", None) is not None:
                        inserted_count += 1
                        print(f"Inserted chunk: {meta['id']} (bot: {bot_id}, src: {doc_id} idx: {meta['chunk_index']})")
                    else:
                        updated_count += 1
                        print(f"Updated chunk:  {meta['id']} (bot: {bot_id}, src: {doc_id} idx: {meta['chunk_index']})")
                except PyMongoError as e:
                    failed_count += 1
                    print(f"Mongo upsert FAILED for {meta['id']}: {repr(e)}", file=sys.stderr)

            total_chunks += len(chunks)

    finally:
        try:
            if cursor is not None:
                cursor.close()
        except Exception:
            pass


    print("\n=== Summary ===")
    print(f"Bot ID: {bot_id}")
    print(f"Source collection: {source_collname}")
    print(f"Docs processed: {total_source_docs}")
    print(f"Total paragraphs seen: {total_paragraphs}")
    print(f"Paragraphs kept after dedupe: {kept_paragraphs}")
    print(f"Chunks created/written: {total_chunks}")
    print(f"JSONL output: {out_jsonl}")
    print("\nMongoDB summary:")
    print(f"  Inserted (new docs): {inserted_count}")
    print(f"  Updated (existing docs): {updated_count}")
    print(f"  Failed upserts: {failed_count}")
    try:
        total_in_db = tgt_coll.count_documents({"bot_id": bot_id})
        print(f"  Total documents in target collection now for bot {bot_id}: {total_in_db}")
    except Exception:
        pass

def run_chunking_for_bot(bot_id: str):
    return process_from_mongo(
        client_uri=MONGO_URI,
        dbname=MONGO_DBNAME,
        source_collname=SOURCE_COLLECTION,
        target_collname=TARGET_COLLECTION,
        out_jsonl=OUT_JSONL,
        limit=LIMIT_SOURCE_DOCS,
        bot_id=bot_id
    )

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python chunker.py <bot_id>")
#         sys.exit(1)
#     bot_id_cli = sys.argv[1]
#     run_chunking_for_bot(bot_id_cli)