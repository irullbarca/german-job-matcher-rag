from __future__ import annotations
import re
from typing import List

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())

def keyword_coverage(resume_text: str, job_keywords: List[str]) -> float:
    if not job_keywords:
        return 0.0

    res = normalize(resume_text)
    hits = 0

    for kw in job_keywords:
        if normalize(kw) in res:
            hits += 1

    return round((hits / len(job_keywords)) * 100, 2)
