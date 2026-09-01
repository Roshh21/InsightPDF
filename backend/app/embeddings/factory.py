from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.huggingface_embeddings import HuggingFaceEmbeddingProvider


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    if settings.EMBEDDING_PROVIDER == "huggingface":
        return HuggingFaceEmbeddingProvider(model_name=settings.EMBEDDING_MODEL)
    raise ValueError(f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}")
