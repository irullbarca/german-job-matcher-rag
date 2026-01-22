from __future__ import annotations
import json
import requests
from typing import Any, Dict

OLLAMA_URL = "http://localhost:11434/api/generate"

def ollama_generate(model: str, prompt: str, temperature: float = 0.2) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Try to extract JSON from model output. Handles extra text.
    """
    text = text.strip()
    # Find first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        return json.loads(candidate)
    return json.loads(text)
