# build_faiss_index.py  — build-only
from pathlib import Path
import numpy as np
import json
import faiss

OUTDIR = Path("embedded")
FAISS_INDEX_FILE = OUTDIR / "faiss.index"

# load saved arrays/metadata
embeddings = np.load(OUTDIR / "embeddings.npy")
with (OUTDIR / "ids.json").open("r", encoding="utf-8") as f:
    ids = json.load(f)
with (OUTDIR / "meta.json").open("r", encoding="utf-8") as f:
    meta = json.load(f)

print(f"Loaded embeddings: {embeddings.shape}, ids: {len(ids)}, meta: {len(meta)}")

# build FAISS (cosine: vectors must be L2-normalized already)
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
faiss.write_index(index, str(FAISS_INDEX_FILE))
print(f"✅ FAISS index built and saved at {FAISS_INDEX_FILE.resolve()}")
#print(embeddings[37])
