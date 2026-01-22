
from __future__ import annotations

import re
from typing import Dict, Any, List

from .retrieve import search_chunks, group_by_job
from .llm import ollama_generate, safe_json_loads
from .ats_score import keyword_coverage  # ensure src/ats_score.py exists

DEFAULT_MODEL = "llama3.1:8b"


def simple_keywords_from_text(text: str) -> list[str]:
    """
    Simple keyword extractor from job evidence (grounded).
    """
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\+\#\.\-/]{1,25}", text)

    stop = {
        "Job", "Title", "Location", "Employment", "Type", "Company", "Description",
        "Responsibilities", "Requirements", "Nice", "Have", "Salary", "Keywords",
        "Germany", "Remote"
    }

    out: list[str] = []
    for t in tokens:
        if t in stop:
            continue
        if len(t) < 2:
            continue
        out.append(t)

    # unique while preserving order (case-insensitive)
    seen = set()
    uniq: list[str] = []
    for t in out:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            uniq.append(t)

    return uniq[:80]


def build_prompt(
    resume_text: str,
    job_results: List[dict],
    job_keywords: list[str],
    kw_coverage: float,
) -> str:
    context_blocks = []
    for j in job_results:
        ev = "\n\n".join(
            [f"- Evidence (score={c['score']:.3f}):\n{c['text']}" for c in j.get("chunks", [])]
        )
        context_blocks.append(
            f"JOB FILE: {j.get('job_file','')}\n"
            f"JOB TITLE: {j.get('job_title','')}\n"
            f"LOCATION: {j.get('location','Unknown')}\n"
            f"TOP RETRIEVED EVIDENCE:\n{ev}"
        )

    context = "\n\n---\n\n".join(context_blocks)
    kw_line = ", ".join(job_keywords) if job_keywords else ""
    n_jobs = len(job_results)

    return f"""
You are an expert German job-market recruiter and ATS specialist.

TASK:
Given a candidate resume and retrieved German job description evidence, produce a structured matching report.

RULES:
- Use ONLY the provided resume + job evidence.
- Be realistic: do not invent experience.
- Output MUST be valid JSON only (no markdown, no commentary).
- Score is 0..100. Provide short, specific bullets.
- Use JOB KEYWORDS list as the primary source for "missing_keywords". Do not invent unrelated tools.
- Consider KEYWORD COVERAGE in your score (higher coverage generally means better match).
- You MUST return exactly {n_jobs} items in best_job_matches (one per JOB FILE in the evidence).
- Do NOT omit any job. If a job is weak, assign a low match_score and explain briefly.

JOB KEYWORDS (extracted from retrieved evidence):
{kw_line}

KEYWORD COVERAGE (job keywords found in resume):
{kw_coverage}%

RESUME:
{resume_text}

RETRIEVED JOB EVIDENCE:
{context}

OUTPUT JSON SCHEMA:
{{
  "overall_summary": "string",
  "best_job_matches": [
    {{
      "job_file": "string",
      "job_title": "string",
      "match_score": number,
      "why_match": ["bullet", "..."],
      "skill_gaps": ["bullet", "..."],
      "missing_keywords": ["keyword", "..."],
      "tailored_cv_suggestions": ["bullet", "..."]
    }}
  ]
}}
""".strip()


def match_resume_to_jobs(
    resume_text: str,
    top_jobs: int = 5,
    model: str = DEFAULT_MODEL,
    location_filter: str = "Berlin + Remote",
) -> Dict[str, Any]:
    """
    End-to-end:
    - retrieve relevant job chunks with embeddings + FAISS
    - group into job-level results with optional location filter (Berlin/Berlin+Remote/All)
    - extract grounded keywords from evidence
    - compute deterministic keyword coverage score
    - call local Ollama model to generate JSON match report
    - return retrieved evidence for UI transparency
    """
    # Increase k for better diversity (more distinct job files)
    chunks = search_chunks(resume_text, k=max(40, top_jobs * 12))

    # NOTE: this expects your UPDATED retrieve.py group_by_job signature:
    # group_by_job(chunks, top_jobs: int, location_filter: str)
    jobs = group_by_job(chunks, top_jobs=top_jobs, location_filter=location_filter)

    # Grounded keywords extracted from retrieved evidence
    evidence_text = "\n".join(c["text"] for j in jobs for c in j.get("chunks", []))
    job_keywords = simple_keywords_from_text(evidence_text)

    # Deterministic keyword coverage % (ATS-style)
    kw_coverage = keyword_coverage(resume_text, job_keywords)

    prompt = build_prompt(resume_text, jobs, job_keywords, kw_coverage)
    raw = ollama_generate(model=model, prompt=prompt, temperature=0.2)

    try:
        data = safe_json_loads(raw)
    except Exception:
        data = {
            "overall_summary": "Model did not return valid JSON. See raw_output.",
            "best_job_matches": [],
            "raw_output": raw[:4000],
        }

    # Attach RAG transparency fields (used by UI)
    if isinstance(data, dict):
        data.setdefault("keyword_coverage_percent", kw_coverage)
        data.setdefault("retrieved_job_keywords", job_keywords)
        data.setdefault("retrieved_jobs", jobs)
        data.setdefault("retrieved_job_files", [j.get("job_file") for j in jobs])
        data.setdefault("location_filter", location_filter)

    # Ensure job_file/job_title exist even if model omits them
    if (
        isinstance(data, dict)
        and "best_job_matches" in data
        and isinstance(data["best_job_matches"], list)
    ):
        for i, item in enumerate(data["best_job_matches"]):
            if i < len(jobs) and isinstance(item, dict):
                item.setdefault("job_file", jobs[i].get("job_file"))
                item.setdefault("job_title", jobs[i].get("job_title"))

        # If LLM returned fewer items than expected, pad with placeholders
        expected = len(jobs)
        got = len(data["best_job_matches"])
        if got < expected:
            for i in range(got, expected):
                data["best_job_matches"].append({
                    "job_file": jobs[i].get("job_file"),
                    "job_title": jobs[i].get("job_title"),
                    "match_score": 0,
                    "why_match": ["Model returned insufficient structured output for this job."],
                    "skill_gaps": [],
                    "missing_keywords": [],
                    "tailored_cv_suggestions": [],
                })

    return data
