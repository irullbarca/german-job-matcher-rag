from __future__ import annotations
import re
from typing import Dict, List
from .text_utils import clean_text

SECTION_HINTS = [
    "skills", "technical skills", "kompetenzen", "kenntnisse",
    "experience", "professional experience", "berufserfahrung",
    "projects", "projekte", "education", "ausbildung", "zertifikate", "certifications"
]

def extract_resume_query(resume_text: str, max_chars: int = 1800) -> str:
    """
    Build a compact query emphasizing skills/tools and role keywords.
    This improves retrieval quality vs embedding the full resume.
    """
    t = clean_text(resume_text)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]

    # Keep likely-signal lines (skills/tools/role summary)
    keep: List[str] = []
    for ln in lines:
        low = ln.lower()
        if any(h in low for h in SECTION_HINTS):
            keep.append(ln)
            continue
        # keep dense skill lines (commas / hyphens)
        if ("," in ln and len(ln) <= 160) or ("-" in ln and len(ln) <= 160):
            keep.append(ln)

    # Fallback: include first ~30 lines if we didn’t find much
    if len(" ".join(keep)) < 400:
        keep = lines[:30]

    query = " ".join(keep)
    query = re.sub(r"\s+", " ", query).strip()
    return query[:max_chars]
