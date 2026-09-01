from io import BytesIO
from typing import List
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
  reader = PdfReader(BytesIO(pdf_bytes))
  parts = []
  for page in reader.pages:
    text = page.extract_text() or ""
    parts.append(text)
  return "\n".join(parts)

def chunk_text(text: str) -> List[str]:
  splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " "],
  )
  chunks = splitter.split_text(text)
  return chunks