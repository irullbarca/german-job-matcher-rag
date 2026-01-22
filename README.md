# 🇩🇪 German Resume ↔ Job Matcher  
### RAG (FAISS) + Local LLM (Ollama) | Berlin & EU Focus

A production-style **Retrieval-Augmented Generation (RAG)** system that matches resumes with German job descriptions using **semantic search, explainable retrieval, and ATS-style scoring** — built to reflect **real hiring workflows** in Berlin and across Europe.

> 🔒 Fully local execution (no OpenAI / no external LLM APIs)  
> 🔍 Transparent RAG with evidence inspection  
> 📊 Deterministic + LLM-assisted scoring  

---

## 🚀 Why This Project Exists

Many LLM job-matching demos behave like black boxes:  
they generate scores without showing *why*.

This project focuses on:
- **Explainability** (inspect retrieved job evidence)
- **Deterministic signals** (keyword coverage, retrieval scores)
- **Privacy-first design** (local embeddings + local LLM)
- **Real-world constraints** of the German / EU job market

The result is a tool that feels like an **internal recruiting assistant**, not a demo.

---

## ✨ Key Features

- 🔍 **Semantic Resume ↔ Job Matching** using FAISS
- 🤖 **Local LLM Reasoning** via Ollama (LLaMA / Mistral / Qwen)
- 🧠 **True RAG Pipeline** (retrieval → evidence → reasoning)
- 📊 **ATS-Style Keyword Coverage** (deterministic)
- 🔎 **RAG Evidence Viewer** (inspect exact job chunks + similarity)
- 📍 **Berlin / Berlin+Remote / All location filtering**
- 🖥️ **Recruiter-friendly Streamlit UI**
- 📄 JSON report export

---

## 🧱 System Architecture
```text
Resume
↓
Embedding (sentence-transformers)
↓
FAISS Vector Search
↓
Job Chunks (with similarity scores)
↓
Explainable RAG Context
↓
Local LLM (Ollama)
↓
Match Score + Skill Gaps + CV Suggestions
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-----|-----------|
| LLM | Ollama (local) |
| Embeddings | sentence-transformers (multilingual) |
| Vector DB | FAISS |
| Backend | Python |
| UI | Streamlit |
| Parsing | PDF / DOCX / TXT loaders |

---

## 📁 Project Structure

```text
german-job-matcher-rag/
│
├── app.py 
├── requirements.txt
├── README.md
│
├── src/
│ ├── matcher.py 
│ ├── retrieve.py
│ ├── index_jobs.py 
│ ├── loaders.py 
│ ├── text_utils.py 
│ ├── ats_score.py 
│ └── llm.py 
│
├── jobs/ # Job descriptions
├── resumes/ # Sample resumes
└── data/ # Generated FAISS index (gitignored)

```
---

## ⚙️ How to Run Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---
### 2️⃣ Install & start Ollama
```sh
ollama pull llama3.1:8b
```
--- 

### 3️⃣ Build the job index
```sh
python -m src.index_jobs
```
---
### 4️⃣ Run the app
```sh
streamlit run app.py
```
---
### 📊 What the Output Looks Like

For each job:
- Match score (0–100)
- Why it matches (bullet points)
- Skill gaps
- Missing ATS keywords
- Tailored CV improvement suggestions
- Exact job description chunks used during retrieval
---

### 📜 Disclaimer

This project is for educational and informational purposes only.
It does not guarantee hiring outcomes and should not be used as an automated decision system.
---

### Screenshots

<img width="1917" height="873" alt="image" src="https://github.com/user-attachments/assets/ee3033b7-f26a-41c1-9a3c-9bc405175976" />
<img width="1915" height="877" alt="image" src="https://github.com/user-attachments/assets/ce8af502-e6ea-4bf7-b2ea-3516bb4c3e4b" />

---

___Thank You___


