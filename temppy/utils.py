# utils.py
import os, json, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from datetime import datetime

# --- Config: adjust as needed ---
OUTDIR = Path("embedded")
OUTDIR.mkdir(exist_ok=True, parents=True)
MODEL_NAME = "all-MiniLM-L6-v2"   # use your HF model name
FAISS_INDEX_PATH = OUTDIR / "faiss.index"
META_PATH = OUTDIR / "meta.json"
IDS_PATH = OUTDIR / "ids.json"

# --- Embedding model (singleton) ---
embedder = SentenceTransformer(MODEL_NAME)

def normalize_text(text: str) -> str:
    t = (text or "").strip().lower()
    t = __import__("re").sub(r"[^\w\s]", "", t)
    t = __import__("re").sub(r"\s+", " ", t)
    return t

def embed_text(texts):
    """Accept single str or list[str]. Return np.ndarray float32 L2-normalized."""
    single = False
    if isinstance(texts, str):
        texts = [texts]; single = True
    vecs = embedder.encode(texts, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vecs)
    return vecs[0] if single else vecs

def load_meta_ids():
    """Return (META:list, IDS:list) - if files missing, return empty lists."""
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            META = json.load(f)
    except Exception:
        META = []
    try:
        with open(IDS_PATH, "r", encoding="utf-8") as f:
            IDS = json.load(f)
    except Exception:
        IDS = []
    return META, IDS

def save_meta_ids(META, IDS):
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(META, f, ensure_ascii=False, indent=2)
    with open(IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(IDS, f, ensure_ascii=False, indent=2)

def load_faiss_index(d=384):
    """Try to load index; if missing, create a flat index (d should match embed dim)."""
    if FAISS_INDEX_PATH.exists():
        try:
            return faiss.read_index(str(FAISS_INDEX_PATH))
        except Exception as e:
            print("faiss read error:", e)
    # fallback: empty Flat index
    index = faiss.IndexFlatIP(d)  # normalized vectors -> inner product ~ cosine
    return index

def save_faiss_index(index):
    try:
        faiss.write_index(index, str(FAISS_INDEX_PATH))
    except Exception as e:
        print("faiss write error:", e)

def add_vectors_to_faiss(index, vectors):
    """Vectors: np.ndarray (N,d). Adds to index and returns new total count."""
    if vectors is None or len(vectors) == 0:
        return index.ntotal
    try:
        index.add(vectors)
    except Exception as e:
        print("faiss add error:", e)
    return index.ntotal

def now_iso():
    return datetime.utcnow().isoformat()
