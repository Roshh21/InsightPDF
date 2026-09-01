from ..services.llm_orchestrator import _call_llm
from .retrieval_agent import retrieve_context

def answer_question(doc_id: str, question: str) -> str:
  context_chunks = retrieve_context(doc_id, question, k=5)
  context = "\n\n".join(context_chunks)

  guardrail = f"""
You ONLY answer questions using the provided PDF context.
If the question is not about this PDF or cannot be answered from the context,
respond exactly with: "Please ask a question based on the uploaded PDF."

Context from PDF:
{context}

User question: {question}

Answer:
"""
  return _call_llm(guardrail, max_tokens=256)
