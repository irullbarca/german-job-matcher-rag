from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .loaders import load_any, guess_job_title
from .text_utils import split_into_chunks

ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = ROOT / "jobs"
DATA_DIR = ROOT / "data"

EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"


@dataclass
class ChunkMeta:
    chunk_id: int
    job_file: str
    job_title: str
    location: str
    text: str


def extract_location(text: str) -> str:
    for line in text.splitlines():
        if line.lower().startswith("location:"):
            return line.split(":", 1)[1].strip()
    return "Unknown"


def embed_passages(model: SentenceTransformer, passages: List[str]) -> np.ndarray:
    passages = [f"passage: {p}" for p in passages]
    emb = model.encode(passages, normalize_embeddings=True, show_progress_bar=True)
    return np.array(emb, dtype="float32")


def main():
    DATA_DIR.mkdir(exist_ok=True)

    job_files = list(JOBS_DIR.glob("*"))
    model = SentenceTransformer(EMBED_MODEL_NAME)

    all_chunks = []
    metas = []
    cid = 0

    for jf in job_files:
        raw = load_any(jf)
        title = guess_job_title(raw) or jf.stem
        location = extract_location(raw)

        chunks = split_into_chunks(raw)

        for ch in chunks:
            all_chunks.append(ch)
            metas.append(
                ChunkMeta(
                    chunk_id=cid,
                    job_file=jf.name,
                    job_title=title,
                    location=location,
                    text=ch,
                )
            )
            cid += 1

    vectors = embed_passages(model, all_chunks)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(DATA_DIR / "jobs.index"))
    with open(DATA_DIR / "jobs_meta.jsonl", "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")

    (DATA_DIR / "embed_model.txt").write_text(EMBED_MODEL_NAME)
    print("✅ Index rebuilt with location metadata")


if __name__ == "__main__":
    main()
