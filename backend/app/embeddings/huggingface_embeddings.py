from __future__ import annotations

import threading

from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger
from app.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Local sentence-transformers embeddings. No network calls at inference
    time once the model is cached, and no API key required."""

    name = "huggingface"

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()
        self._dimension: int | None = None

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer

                        logger.info("Loading embedding model %s", self.model_name)
                        self._model = SentenceTransformer(self.model_name)
                        self._dimension = self._model.get_sentence_embedding_dimension()
                    except Exception as exc:  # pragma: no cover
                        raise EmbeddingError(f"Failed to load embedding model: {exc}") from exc
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        try:
            vectors = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
            return [v.tolist() for v in vectors]
        except Exception as exc:
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        model = self._load()
        try:
            vector = model.encode([text], show_progress_bar=False, normalize_embeddings=True)[0]
            return vector.tolist()
        except Exception as exc:
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load()
        return self._dimension  # type: ignore[return-value]
