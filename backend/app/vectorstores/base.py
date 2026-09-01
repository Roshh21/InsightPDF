from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Chunk:
    """A single indexable unit of a document."""

    id: str
    workspace_id: str
    document_id: str
    text: str
    page: int
    section: Optional[str] = None
    chunk_index: int = 0
    document_type: Optional[str] = None
    document_name: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "document_id": self.document_id,
            "page": self.page,
            "section": self.section or "",
            "chunk_index": self.chunk_index,
            "document_type": self.document_type or "generic",
            "document_name": self.document_name or "",
            **self.extra,
        }


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_name: str
    page: int
    section: Optional[str]
    text: str
    score: float  # similarity score, higher = more relevant (0..1 normalized where possible)


class VectorStore(ABC):
    """
    Abstraction over the vector database. Isolation is per-workspace (and
    filterable by document within a workspace) so that uploading a new
    document never clears or disturbs existing embeddings -- the opposite
    of the original single-global-collection implementation.
    """

    name: str = "base"

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        workspace_id: str,
        document_ids: Optional[list[str]] = None,
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        ...

    @abstractmethod
    def delete_document(self, workspace_id: str, document_id: str) -> None:
        ...

    @abstractmethod
    def delete_workspace(self, workspace_id: str) -> None:
        ...

    @abstractmethod
    def health_check(self) -> bool:
        ...
