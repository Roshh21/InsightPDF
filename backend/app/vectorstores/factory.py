from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.embeddings.factory import get_embedding_provider
from app.vectorstores.base import VectorStore


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    if settings.VECTOR_STORE == "chroma":
        from app.vectorstores.chroma_store import ChromaStore

        return ChromaStore(persist_directory=settings.CHROMA_DIR)
    if settings.VECTOR_STORE == "qdrant":
        from app.vectorstores.qdrant_store import QdrantStore

        dim_hint = 384
        try:
            dim_hint = get_embedding_provider().dimension
        except Exception:
            pass
        return QdrantStore(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            path=settings.QDRANT_PATH,
            dimension_hint=dim_hint,
        )
    raise ValueError(f"Unsupported vector store: {settings.VECTOR_STORE}")
