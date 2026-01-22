from __future__ import annotations
import re
from typing import List

def clean_text(text: str) -> str:
    if not text:
        return ""
    # Normalize whitespace
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_into_chunks(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    Simple character-based chunking that keeps paragraph boundaries when possible.
    chunk_size/overlap are characters (works well for a v1).
    """
    text = clean_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf = ""

    def flush_buf():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for p in paragraphs:
        if len(buf) + len(p) + 2 <= chunk_size:
            buf = (buf + "\n\n" + p).strip() if buf else p
        else:
            flush_buf()
            if len(p) <= chunk_size:
                buf = p
            else:
                # If a paragraph is too large, hard-split it
                start = 0
                while start < len(p):
                    end = min(start + chunk_size, len(p))
                    chunks.append(p[start:end].strip())
                    start = max(end - overlap, start + 1)
                buf = ""

    flush_buf()

    # Add overlap between adjacent chunks
    if overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, c in enumerate(chunks):
            if i == 0:
                overlapped.append(c)
            else:
                prev = overlapped[-1]
                take = prev[-overlap:] if len(prev) > overlap else prev
                overlapped.append((take + "\n" + c).strip())
        chunks = overlapped

    return [c for c in chunks if c]
