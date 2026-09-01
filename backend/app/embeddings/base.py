from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstraction over embedding backends.

    Kept entirely separate from the LLM provider abstraction (app/llm) --
    you can swap the chat/completion model without touching how documents
    are embedded, and vice versa.
    """

    name: str = "base"

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunk texts for indexing."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for retrieval."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality, needed by vector stores that pre-declare schema (Qdrant)."""
