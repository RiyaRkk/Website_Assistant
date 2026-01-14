# fast_query.py — reuse-loaded model for many queries
from pathlib import Path
import json, numpy as np, faiss, torch
from sentence_transformers import SentenceTransformer

OUTDIR = Path("embedded")
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

# load FAISS + metadata once
index = faiss.read_index(str(OUTDIR / "faiss.index"))
with (OUTDIR / "ids.json").open("r", encoding="utf-8") as f: IDS = json.load(f)
with (OUTDIR / "meta.json").open("r", encoding="utf-8") as f: META = json.load(f)

# load embedding model once (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)

def embed_query(q: str) -> np.ndarray:
    v = model.encode([q], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(v)  # must match how you built the index
    return v

def retrieve(q: str, k: int = TOP_K):
    qv = embed_query(q)
    print(qv)
    D, I = index.search(qv, k)
    results = []
    for s, idx in zip(D[0], I[0]):
        m = META[idx]
        text = m.get("text") or (m.get("question","") + " " + m.get("answer",""))
        results.append({"score": float(s), "id": IDS[idx], "text": text})
    print(D, I)
    print(results)
    return results

if __name__ == "__main__":
    print("Ask a question (type 'exit' to quit):")
    while True:
        q = input("> ").strip()
        if not q or q.lower() == "exit":
            break
        hits = retrieve(q)
        for r in hits:
            print(f"{r['score']:.4f} | {r['id']} | {r['text'][:140]}")
