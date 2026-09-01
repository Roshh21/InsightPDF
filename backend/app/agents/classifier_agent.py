from ..services.llm_orchestrator import _call_llm

DOC_TYPES = [
  "research paper",
  "novel or literature",
  "study material or notes",
  "technical documentation",
  "business report",
  "other",
]

def classify_document(text_sample: str) -> str:
  prompt = f"""
You are a document classifier.

Given the following excerpt of a PDF, classify it into exactly one of these types:
{", ".join(DOC_TYPES)}.

Return only the type text, nothing else.

Excerpt:
{text_sample[:2500]}
"""
  result = _call_llm(prompt, max_tokens=32)
  for t in DOC_TYPES:
    if t.lower() in result.lower():
      return t
  return "other"
