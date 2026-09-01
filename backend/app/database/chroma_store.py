import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict
from ..config import CHROMA_DIR
from ..services.embedding_service import embed_texts

_client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings())
_COLLECTION_PREFIX = "insightpdf_"

def _collection_name(doc_id: str) -> str:
  return f"{_COLLECTION_PREFIX}{doc_id}"

def get_collection(doc_id: str):
  return _client.get_or_create_collection(
    name=_collection_name(doc_id),
    embedding_function=embed_texts,
  )

def add_chunks(doc_id: str, chunks: List[str], metadatas: List[Dict]):
  col = get_collection(doc_id)
  ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
  col.add(ids=ids, documents=chunks, metadatas=metadatas)

def query_chunks(doc_id: str, query: str, k: int = 5):
  col = get_collection(doc_id)
  res = col.query(query_texts=[query], n_results=k)
  docs = res.get("documents", [[]])[0]
  return docs
