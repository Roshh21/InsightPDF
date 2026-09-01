from typing import List
from ..database.chroma_store import query_chunks

def retrieve_context(doc_id: str, question: str, k: int = 5) -> List[str]:
  return query_chunks(doc_id, question, k=k)
