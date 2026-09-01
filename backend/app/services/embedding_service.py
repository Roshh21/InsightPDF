from sentence_transformers import SentenceTransformer
from typing import List
from ..config import EMBEDDING_MODEL_NAME

_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
  global _embedding_model
  if _embedding_model is None:
    _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
  return _embedding_model

def embed_texts(texts: List[str]):
  model = get_embedding_model()
  return model.encode(texts).tolist()
