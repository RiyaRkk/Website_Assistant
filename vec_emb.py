from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError
import gridfs
from io import BytesIO
import os
from dotenv import load_dotenv
load_dotenv()
# OUTDIR = Path("embedded")
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DBNAME = "chat_bot_db"
MONGO_CHUNKS_COLLECTION = "chunks"
MONGO_SINGLE_COLLECTION = "bundles"

FETCH_LIMIT = None

def load_chunks_from_mongo(bot_id: str,
                           uri: str = MONGO_URI,
                           dbname: str = MONGO_DBNAME,
                           collname: str = MONGO_CHUNKS_COLLECTION,
                           limit: int = FETCH_LIMIT):
    client = MongoClient(uri)
    db = client[dbname]
    coll = db[collname]

    total = coll.count_documents({"bot_id": bot_id})
    if total == 0:
        raise RuntimeError(f"No documents found for bot_id={bot_id} in {dbname}.{collname} - aborting.")

    cursor = coll.find({"bot_id": bot_id}, {"text": 1, "meta": 1, "_id": 0})
    if limit:
        cursor = cursor.limit(limit)

    items = []
    for doc in cursor:
        text = doc.get("text")
        meta = doc.get("meta") or {}
        if not text and isinstance(meta, dict) and meta.get("text"):
            text = meta.get("text")
        if not text:
            parts = [f"{k}: {v}" for k, v in (meta.items() if isinstance(meta, dict) else []) if isinstance(v, str) and v.strip()]
            text = " | ".join(parts) if parts else ""
        if isinstance(meta, dict):
            if "id" not in meta:
                inferred_id = meta.get("path") or meta.get("source") or f"mongo_chunk_{len(items)}"
                meta["id"] = str(inferred_id)
            meta["text"] = text
            meta["bot_id"] = bot_id
        else:
            meta = {"id": f"mongo_chunk_{len(items)}", "text": text, "bot_id": bot_id}
        items.append({"text": text, "meta": meta})

    print(f"Loaded {len(items)} chunks for bot {bot_id} from {dbname}.{collname} (total in coll for bot: {total}).")
    client.close()
    return items

def build_texts(chunks):
    texts = []
    for item in chunks:
        if isinstance(item, dict) and "text" in item:
            text = item["text"]
        elif isinstance(item, dict):
            parts = [f"{k}: {v}" for k, v in item.items() if isinstance(v, str) and v.strip()]
            text = " | ".join(parts) if parts else ""
        else:
            text = str(item)
        texts.append(text)
    print(f"Prepared {len(texts)} text chunks for embedding.")
    return texts

def build_ids_meta(chunks, texts, bot_id: str):
    ids = []
    meta = []
    seen = set()
    for i, item in enumerate(chunks):
        cid = None
        if isinstance(item, dict):
            meta_obj = item.get("meta", {})
            cid = meta_obj.get("id") or item.get("id") or item.get("path")
        if not cid:
            cid = f"{bot_id}_chunk_{i}"
        cid = str(cid)
        if cid in seen:
            cid = f"{cid}_{i}"
        seen.add(cid)
        ids.append(cid)

        if isinstance(item, dict) and isinstance(item.get("meta"), dict) and item.get("meta"):
            entry_meta = dict(item["meta"])
        elif isinstance(item, dict):
            entry_meta = dict(item)
        else:
            entry_meta = {"raw": str(item)}

        entry_meta["id"] = cid
        entry_meta["text"] = texts[i]
        entry_meta["bot_id"] = bot_id
        meta.append(entry_meta)
    return ids, meta

def embed_texts(texts, model_name=MODEL_NAME, batch_size=BATCH_SIZE):
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    if not isinstance(emb, np.ndarray):
        emb = np.array(emb)
    return emb.astype("float32")

def normalize_embeddings(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    return emb.astype("float32")

def save_single_bundle( emb: np.ndarray, ids: list, meta: list,
                       bot_id: str,
                       mongo_uri: str = MONGO_URI, mongo_db: str = MONGO_DBNAME,
                       bundle_id: str | None = None):
    if bundle_id is None:
        bundle_id = f"emb_{bot_id}"

    # outdir.mkdir(parents=True, exist_ok=True)
    # bundle_path = outdir / f"{bundle_id}.npz"

    # np.savez_compressed(bundle_path, embeddings=emb, ids=np.array(ids, dtype=object), meta=np.array(meta, dtype=object))
    # print(f"Saved single compressed bundle -> {bundle_path}")

    bio = BytesIO()
    np.savez_compressed(bio, embeddings=emb, ids=np.array(ids, dtype=object), meta=np.array(meta, dtype=object))
    bio.seek(0)
    data_bytes = bio.read()

    client = MongoClient(mongo_uri)
    db = client[mongo_db]
    bundles = db[MONGO_SINGLE_COLLECTION]

    approx_doc_overhead = 1024
    if len(data_bytes) + approx_doc_overhead < (16 * 1024 * 1024):
        try:
            bundles.replace_one(
                {"_id": bundle_id},
                {
                    "_id": bundle_id,
                    "bot_id": bot_id,
                    "created_at": datetime.utcnow(),
                    "bundle_bytes": data_bytes,
                    "ids": ids,
                    "meta": meta,
                    "count": len(ids),
                },
                upsert=True,
            )
            print(f"Inserted/updated single bundle document in {mongo_db}.{MONGO_SINGLE_COLLECTION} (inline) for bot {bot_id}.")
        except PyMongoError as e:
            print("Failed to store inline bundle document in MongoDB:", repr(e))
    else:
        try:
            fs = gridfs.GridFS(db)
            existing = bundles.find_one({"_id": bundle_id})
            if existing and existing.get("gridfs_id"):
                try:
                    fs.delete(existing["gridfs_id"])
                except Exception:
                    pass


            gridfs_id = fs.put(data_bytes, uploaded_at=datetime.utcnow())
            bundles.replace_one(
                {"_id": bundle_id},
                {
                    "_id": bundle_id,
                    "bot_id": bot_id,
                    "created_at": datetime.utcnow(),
                    "gridfs_id": gridfs_id,
                    "ids": ids,
                    "meta": meta,
                    "count": len(ids),
                    "note": "bundle stored in GridFS due to size"
                },
                upsert=True,
            )
            print(f"Bundle stored into GridFS (id: {gridfs_id}) for bot {bot_id}.")
        except PyMongoError as e:
            print("Failed to store bundle in GridFS / MongoDB:", repr(e))
    client.close()

def main_for_bot(bot_id: str):
    chunks = load_chunks_from_mongo(bot_id=bot_id)
    texts = build_texts(chunks)
    ids, meta = build_ids_meta(chunks, texts, bot_id)

    embeddings = embed_texts(texts)
    print("Embeddings generated. shape:", embeddings.shape)

    emb_norm = normalize_embeddings(embeddings)
    print("Embeddings normalized.")

    save_single_bundle( emb_norm, ids, meta, bot_id=bot_id)

def run_embedding_for_bot(bot_id: str):
    return main_for_bot(bot_id)

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) < 2:
#         print("Usage: python vec_emb.py <bot_id>")
#         sys.exit(1)
#     bot_id_cli = sys.argv[1]
#     main_for_bot(bot_id_cli)
