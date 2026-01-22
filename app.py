# app.py
from __future__ import annotations

import json
from pathlib import Path
import requests
import streamlit as st

from src.loaders import load_any
from src.matcher import match_resume_to_jobs, DEFAULT_MODEL

ROOT = Path(__file__).resolve().parent
RESUMES_DIR = ROOT / "resumes"
DATA_DIR = ROOT / "data"

OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"


def is_ollama_up() -> bool:
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def list_sample_resumes() -> list[Path]:
    if not RESUMES_DIR.exists():
        return []
    return sorted([p for p in RESUMES_DIR.glob("*") if p.suffix.lower() in [".txt", ".pdf", ".docx"]])


def pretty_percent(v: float) -> float:
    try:
        v = float(v)
    except Exception:
        return 0.0
    return max(0.0, min(100.0, v))


st.set_page_config(page_title="German RAG Job Matcher", page_icon="🇩🇪", layout="wide")

# Header
st.markdown(
    """
    <div style="display:flex;align-items:center;justify-content:space-between;">
      <div>
        <h1 style="margin-bottom:0;">🇩🇪 Resume ↔ Job Matcher</h1>
        <div style="opacity:0.8;margin-top:4px;">
          RAG (FAISS + sentence-transformers) + Local LLM (Ollama) — no OpenAI
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# Sidebar
with st.sidebar:
    st.header("Run settings")

    ollama_ok = is_ollama_up()
    if ollama_ok:
        st.success("Ollama: running")
    else:
        st.error("Ollama: not reachable")
        st.caption("Start Ollama, then retry. Example: `ollama pull llama3.1:8b`")

    model = st.text_input("Ollama model", value=DEFAULT_MODEL)
    top_jobs = st.slider("Jobs to evaluate", min_value=3, max_value=10, value=5, step=1)

    location_filter = st.selectbox(
        "Job location",
        ["Berlin + Remote", "Berlin", "All"],
        index=0,
        help="Filters jobs before the LLM analysis.",
    )

    st.divider()
    st.subheader("Data status")
    index_exists = (DATA_DIR / "jobs.index").exists()
    meta_exists = (DATA_DIR / "jobs_meta.jsonl").exists()
    if index_exists and meta_exists:
        st.success("Job index: ready")
    else:
        st.warning("Job index: missing")
        st.caption("Run: `python -m src.index_jobs`")

    st.divider()
    st.caption("Tip: more job files = better retrieval variety.")

# Input section
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1) Resume input")
    uploaded = st.file_uploader("Upload resume (TXT / PDF / DOCX)", type=["txt", "pdf", "docx"])

    sample_files = list_sample_resumes()
    sample_choice = None
    if sample_files:
        sample_choice = st.selectbox(
            "…or select a sample resume from /resumes",
            options=["(none)"] + [p.name for p in sample_files],
            index=0,
        )

with col_right:
    st.subheader("2) Controls")
    run_btn = st.button("🔎 Find matching jobs", use_container_width=True, type="primary")
    st.caption("Runs retrieval (RAG) + local LLM analysis.")

# Load resume text
resume_text = ""
resume_source = ""

if uploaded is not None:
    temp_path = DATA_DIR / f"_uploaded_resume{Path(uploaded.name).suffix.lower()}"
    temp_path.parent.mkdir(exist_ok=True)
    temp_path.write_bytes(uploaded.getvalue())
    resume_text = load_any(temp_path)
    resume_source = f"Uploaded: {uploaded.name}"
elif sample_choice and sample_choice != "(none)":
    p = RESUMES_DIR / sample_choice
    resume_text = load_any(p)
    resume_source = f"Sample: {sample_choice}"

# Preview only on click
if resume_text.strip():
    st.write("")
    if st.button("👁️ Preview resume"):
        st.subheader("Resume preview")
        tabs = st.tabs(["Preview", "Raw text"])
        with tabs[0]:
            st.info(resume_source)
            lines = resume_text.splitlines()
            st.write("\n".join(lines[:120]) + ("\n…" if len(lines) > 120 else ""))
        with tabs[1]:
            st.text(resume_text[:9000])
else:
    st.info("Upload a resume or pick a sample to continue.")

st.write("")
st.divider()
st.write("")

# Run matching
if run_btn:
    if not resume_text.strip():
        st.error("No resume loaded. Upload a resume or choose a sample.")
        st.stop()
    if not ollama_ok:
        st.error("Ollama is not reachable. Start Ollama and retry.")
        st.stop()
    if not ((DATA_DIR / "jobs.index").exists() and (DATA_DIR / "jobs_meta.jsonl").exists()):
        st.error("Job index not found. Run: `python -m src.index_jobs`")
        st.stop()

    with st.spinner("Running RAG retrieval + LLM analysis..."):
        result = match_resume_to_jobs(
            resume_text,
            top_jobs=top_jobs,
            model=model,
            location_filter=location_filter,
        )

    st.subheader("3) Results")
    overall = result.get("overall_summary", "")
    if overall:
        st.write(overall)

    # Downloads + metrics
    st.write("")
    dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 1])
    with dl_col1:
        st.download_button(
            "⬇️ Download JSON report",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name="match_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with dl_col2:
        cov = result.get("keyword_coverage_percent", None)
        st.metric("Keyword coverage (global)", f"{pretty_percent(cov):.2f}%" if cov is not None else "—")
    with dl_col3:
        st.metric("Location filter", result.get("location_filter", location_filter))

    matches = result.get("best_job_matches", [])
    retrieved_jobs = result.get("retrieved_jobs", [])

    # Lookup: job_file -> retrieved evidence
    evidence_by_file = {j.get("job_file"): j for j in retrieved_jobs}

    if not matches:
        st.warning("No matches returned (or JSON parsing failed).")
        if "raw_output" in result:
            with st.expander("Raw model output (debug)"):
                st.text(result["raw_output"])
        st.stop()

    # Cards + evidence viewer
    for i, m in enumerate(matches, start=1):
        title = m.get("job_title") or m.get("job_file") or f"Job {i}"
        job_file = m.get("job_file", "—")
        llm_score = pretty_percent(m.get("match_score", 0))

        st.markdown(f"### {i}. {title}")
        st.caption(f"Source file: `{job_file}`")

        st.progress(llm_score / 100.0)

        met1, met2, met3, met4 = st.columns(4)
        met1.metric("LLM Match", f"{llm_score:.0f}/100")
        met2.metric("Keyword coverage", f"{pretty_percent(result.get('keyword_coverage_percent', 0)):.2f}%")

        ev_job = evidence_by_file.get(job_file, {})
        met3.metric("Evidence chunks", str(len(ev_job.get("chunks", []))))
        met4.metric("Location", ev_job.get("location", "—"))

        t1, t2, t3, t4, t5 = st.tabs(
            ["Why match", "Skill gaps", "Missing keywords", "CV suggestions", "🔍 RAG evidence"]
        )

        with t1:
            why = m.get("why_match", [])
            if why:
                for b in why:
                    st.write(f"- {b}")
            else:
                st.write("—")

        with t2:
            gaps = m.get("skill_gaps", [])
            if gaps:
                for b in gaps:
                    st.write(f"- {b}")
            else:
                st.write("—")

        with t3:
            kws = m.get("missing_keywords", [])
            if kws:
                st.write(", ".join(kws))
            else:
                st.write("—")

        with t4:
            sug = m.get("tailored_cv_suggestions", [])
            if sug:
                for b in sug:
                    st.write(f"- {b}")
            else:
                st.write("—")

        with t5:
            if not ev_job:
                st.write("No evidence found for this job.")
            else:
                st.write("Exact job-description chunks retrieved by FAISS (RAG):")
                for k, c in enumerate(ev_job.get("chunks", []), start=1):
                    score = c.get("score", 0.0)
                    st.markdown(f"**Chunk {k} — similarity: {score:.3f}**")
                    st.text((c.get("text", "") or "")[:3000])
                    st.write("")

        st.divider()

    st.success("Done ✅")
