from __future__ import annotations

import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load_index_and_meta():
    index = faiss.read_index(str(DATA_DIR / "jobs.index"))
    metas = []
    with open(DATA_DIR / "jobs_meta.jsonl", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return index, metas


def load_embed_model():
    model_name = (DATA_DIR / "embed_model.txt").read_text().strip()
    return SentenceTransformer(model_name)


def embed_query(model, text: str):
    return model.encode([f"query: {text}"], normalize_embeddings=True)


def search_chunks(query: str, k: int = 40):
    index, metas = load_index_and_meta()
    model = load_embed_model()
    qv = embed_query(model, query)
    scores, ids = index.search(np.array(qv, dtype="float32"), k)

    out = []
    for s, i in zip(scores[0], ids[0]):
        if i >= 0:
            m = metas[i].copy()
            m["score"] = float(s)
            out.append(m)
    return out


def group_by_job(chunks, top_jobs: int, location_filter: str):
    jobs = {}
    for c in chunks:
        loc = c.get("location", "").lower()

        if location_filter == "Berlin":
            if "berlin" not in loc:
                continue
        elif location_filter == "Berlin + Remote":
            if not ("berlin" in loc or "remote" in loc):
                continue

        jf = c["job_file"]
        if jf not in jobs:
            jobs[jf] = {
                "job_file": jf,
                "job_title": c["job_title"],
                "location": c["location"],
                "score": c["score"],
                "chunks": [],
            }

        if len(jobs[jf]["chunks"]) < 3:
            jobs[jf]["chunks"].append({"text": c["text"], "score": c["score"]})

    return sorted(jobs.values(), key=lambda x: x["score"], reverse=True)[:top_jobs]
