from typing import Tuple
from .pdf_processor import extract_text_from_pdf, chunk_text
from ..database.chroma_store import add_chunks
from ..agents.classifier_agent import classify_document
from ..agents.summarizer_agent import summarize_document

def ingest_pdf(doc_id: str, pdf_bytes: bytes):
  full_text = extract_text_from_pdf(pdf_bytes)
  chunks = chunk_text(full_text)
  metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]
  add_chunks(doc_id, chunks, metadatas)

  sample = "\n\n".join(chunks[:5])
  doc_type = classify_document(sample)
  summary = summarize_document(doc_type, sample)
  summary["documentType"] = doc_type
  summary["docId"] = doc_id
  return summary, full_text
