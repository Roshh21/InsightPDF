import requests
from typing import List
from ..config import HF_API_KEY, LLM_MODEL_NAME

API_URL = f"https://api-inference.huggingface.co/models/{LLM_MODEL_NAME}"

def _call_llm(prompt: str, max_tokens: int = 512) -> str:
  headers = {"Authorization": f"Bearer {HF_API_KEY}"}
  payload = {
    "inputs": prompt,
    "parameters": {
      "max_new_tokens": max_tokens,
      "temperature": 0.4,
      "top_p": 0.9,
    },
  }
  resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
  resp.raise_for_status()
  data = resp.json()
  if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
    return data[0]["generated_text"].replace(prompt, "").strip()
  if isinstance(data, dict) and "generated_text" in data:
    return data["generated_text"].strip()
  return str(data)
