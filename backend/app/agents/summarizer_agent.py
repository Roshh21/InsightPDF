from typing import List, Dict
from ..services.llm_orchestrator import _call_llm

def build_summary_prompt(doc_type: str, context: str) -> str:
  base = f"""
You are InsightPDF, an assistant that writes concise, structured bullet-point summaries.

Document type: {doc_type}

Based ONLY on the provided content, produce a summary in clear sections.
Output format (plain text):

Section Title:
- point
- point

Use 3–6 sections, 3–6 bullets per section, be faithful to the text, and avoid speculation.

Content:
{context[:6000]}
"""
  return base

def summarize_document(doc_type: str, representative_text: str) -> Dict:
  prompt = build_summary_prompt(doc_type, representative_text)
  summary_text = _call_llm(prompt, max_tokens=600)

  sections: List[Dict] = []
  current = None
  for line in summary_text.splitlines():
    line = line.strip()
    if not line:
      continue
    if line.endswith(":") and len(line) < 80:
      if current:
        sections.append(current)
      current = {"title": line[:-1].strip(), "points": []}
    elif line.startswith("-") and current:
      current["points"].append(line.lstrip("-• ").strip())
  if current:
    sections.append(current)

  if not sections:
    sections = [
      {"title": "Summary", "points": [summary_text.strip()[:400]]}
    ]
  return {"documentType": doc_type, "sections": sections}
