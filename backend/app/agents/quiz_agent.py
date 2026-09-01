from typing import Dict
from ..services.llm_orchestrator import _call_llm
from .retrieval_agent import retrieve_context

def generate_quiz(doc_id: str, num_questions: int = 5) -> Dict:
  context_chunks = retrieve_context(doc_id, "overall content summary", k=8)
  context = "\n\n".join(context_chunks)

  prompt = f"""
You are a quiz generator. Based ONLY on the PDF content below, create a short quiz.

Content:
{context[:6000]}

Instructions:
- Generate {num_questions} questions.
- Mix of MCQ, True/False, and short-answer.
- Make questions clear and unambiguous.
- Provide an answer for each question.
- Output in strict JSON with this shape:

{{
  "questions": [
    {{
      "type": "mcq" | "true_false" | "short",
      "prompt": "question text",
      "options": ["A", "B", "C", "D"]  // only for mcq, otherwise []
      "answer": "the correct answer or explanation"
    }}
  ]
}}

Return ONLY valid JSON, no extra text.
"""
  raw = _call_llm(prompt, max_tokens=700)

  import json
  try:
    quiz = json.loads(raw)
  except Exception:
    quiz = {
      "questions": [
        {
          "type": "short",
          "prompt": "Summarize the main idea of the document in 2–3 sentences.",
          "options": [],
          "answer": "",
        }
      ]
    }
  return quiz