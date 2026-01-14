# collector.py
from pymongo import MongoClient
from datetime import datetime, timedelta
import numpy as np
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

from temppy.utils import normalize_text, embed_text

# --- Config: adjust MONGO_URI as needed ---


MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["chat_bot_db"]
faq_topics = db["faq_topics"]   # owner "sheet" collection

# similarity threshold for semantic grouping (cosine because vectors normalized)
SIMILARITY_THRESHOLD = 0.80

def _now():
    return datetime.utcnow().isoformat()

def _examples_push(doc, q_text, limit=8):
    ex = doc.get("examples", [])
    ex.insert(0, {"q": q_text, "ts": _now()})
    ex = ex[:limit]
    return ex

def find_similar_topic_by_embedding(vec, candidates):
    """candidates: iterable of dicts with 'representative_embedding'. Return best candidate id or None."""
    best = None
    best_score = -1.0
    for c in candidates:
        rep = c.get("representative_embedding")
        if not rep:
            continue
        repv = np.array(rep, dtype="float32")
        # vectors are expected to be normalized; compute inner product
        score = float(np.dot(vec, repv))
        if score > best_score:
            best_score = score
            best = c
    if best_score >= SIMILARITY_THRESHOLD:
        return best, best_score
    return None, best_score

def assign_to_topic(question: str, attempted_answer: str = None, session_id: str = None):
    """Main function: group or create topic and increment counter."""
    q_raw = question or ""
    norm = normalize_text(q_raw)
    now = _now()

    # 1) try exact normalized match
    doc = faq_topics.find_one({"norm_question": norm})
    if doc:
        faq_topics.update_one({"_id": doc["_id"]}, {
            "$inc": {"count_total": 1},
            "$set": {"last_seen": now},
            "$push": {"examples": {"q": q_raw, "ts": now}}
        })
        return str(doc["_id"])

    # 2) semantic match: load recent candidate docs (limit 200 for speed)
    vec = embed_text(q_raw)  # normalized vector
    candidates = list(faq_topics.find({}, projection={"representative_embedding": 1, "norm_question": 1}).limit(500))
    match, score = find_similar_topic_by_embedding(vec, candidates)
    if match:
        faq_topics.update_one({"_id": match["_id"]}, {
            "$inc": {"count_total": 1},
            "$set": {"last_seen": now},
            "$push": {"examples": {"q": q_raw, "ts": now}}
        })
        return str(match["_id"])

    # 3) create new topic
    new_doc = {
        "topic_id": str(uuid.uuid4()),
        "norm_question": norm,
        "canonical_question": q_raw,
        "examples": [{"q": q_raw, "ts": now}],
        "count_total": 1,
        "first_seen": now,
        "last_seen": now,
        "status": "pending",          # owner edits later, sets to "published"
        "representative_embedding": vec.tolist(),
        "attempted_answers": [{"text": attempted_answer, "ts": now}] if attempted_answer else [],
    }
    res = faq_topics.insert_one(new_doc)
    return str(res.inserted_id)

def record_low_confidence(question: str, attempted_answer: str = None, session_id: str = None):
    """Call this from your chat flow when low_confidence is detected."""
    try:
        tid = assign_to_topic(question, attempted_answer, session_id)
        return tid
    except Exception as e:
        print("collector error:", e)
        return None
