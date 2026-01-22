from __future__ import annotations
from pathlib import Path
from typing import Optional
from pypdf import PdfReader
from docx import Document

from .text_utils import clean_text

def load_txt(path: Path) -> str:
    return clean_text(path.read_text(encoding="utf-8", errors="ignore"))

def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return clean_text("\n".join(parts))

def load_docx(path: Path) -> str:
    doc = Document(str(path))
    parts = []
    for p in doc.paragraphs:
        parts.append(p.text)
    return clean_text("\n".join(parts))

def load_any(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt":
        return load_txt(path)
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)
    # Fallback: try reading as text
    return clean_text(path.read_text(encoding="utf-8", errors="ignore"))

def guess_job_title(text: str) -> Optional[str]:
    """
    Tries to extract job title from lines like:
    'Job Title: ...' or first non-empty line.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:20]:
        if ln.lower().startswith("job title:"):
            return ln.split(":", 1)[1].strip() or None
        if ln.lower().startswith("titel:"):
            return ln.split(":", 1)[1].strip() or None
    if lines:
        # Often first line is title
        return lines[0][:120]
    return None
