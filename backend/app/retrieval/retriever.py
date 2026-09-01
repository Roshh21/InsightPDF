from __future__ import annotations

from dataclasses import dataclass

from app.embeddings.base import EmbeddingProvider
from app.vectorstores.base import RetrievedChunk, VectorStore


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    query: str
    avg_score: float

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0


class Retriever:
    """Thin orchestration layer over VectorStore + EmbeddingProvider.
    This is what makes retrieval swappable between Chroma/Qdrant without
    the agent/tools caring which backend is active."""

    def __init__(self, vector_store: VectorStore, embedding_provider: EmbeddingProvider):
        self.vector_store = vector_store
        self.embeddings = embedding_provider

    def retrieve(
        self,
        query: str,
        *,
        workspace_id: str,
        document_ids: list[str] | None = None,
        top_k: int = 8,
    ) -> RetrievalResult:
        query_embedding = self.embeddings.embed_query(query)
        chunks = self.vector_store.search(
            query_embedding, workspace_id=workspace_id, document_ids=document_ids, top_k=top_k
        )
        avg_score = sum(c.score for c in chunks) / len(chunks) if chunks else 0.0
        return RetrievalResult(chunks=chunks, query=query, avg_score=avg_score)
