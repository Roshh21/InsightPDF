from __future__ import annotations

from typing import Optional

from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.vectorstores.base import Chunk, RetrievedChunk, VectorStore

logger = get_logger(__name__)


def _collection_name(workspace_id: str) -> str:
    safe = workspace_id.replace("-", "")[:40]
    return f"ws_{safe}"


class QdrantStore(VectorStore):
    """
    Qdrant-backed vector store. Works both against a local/self-hosted
    instance (QDRANT_URL) and an embedded on-disk instance (QDRANT_PATH) --
    and the same code path works unmodified against Qdrant Cloud (a managed
    deployment) by supplying QDRANT_URL + QDRANT_API_KEY.

    One collection per workspace, mirroring ChromaStore, so retrieval/business
    logic in app/retrieval does not need to know which backend is active.
    """

    name = "qdrant"

    def __init__(self, url: Optional[str], api_key: Optional[str], path: str, dimension_hint: int = 384):
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover
            raise VectorStoreError("qdrant-client is not installed") from exc

        self._qdrant_client_module = __import__("qdrant_client")
        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            self._client = QdrantClient(path=path)
        self._dimension_hint = dimension_hint
        self._ensured: set[str] = set()

    def _ensure_collection(self, workspace_id: str, dimension: int) -> None:
        name = _collection_name(workspace_id)
        if name in self._ensured:
            return
        from qdrant_client.models import Distance, VectorParams

        existing = [c.name for c in self._client.get_collections().collections]
        if name not in existing:
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
        self._ensured.add(name)

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        from qdrant_client.models import PointStruct

        by_workspace: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            by_workspace.setdefault(c.workspace_id, []).append(i)

        try:
            for workspace_id, idxs in by_workspace.items():
                dim = len(embeddings[idxs[0]])
                self._ensure_collection(workspace_id, dim)
                points = [
                    PointStruct(
                        id=chunks[i].id,
                        vector=embeddings[i],
                        payload={**chunks[i].metadata(), "text": chunks[i].text},
                    )
                    for i in idxs
                ]
                self._client.upsert(collection_name=_collection_name(workspace_id), points=points)
        except Exception as exc:
            raise VectorStoreError(f"Failed to index chunks in Qdrant: {exc}") from exc

    def search(
        self,
        query_embedding: list[float],
        workspace_id: str,
        document_ids: Optional[list[str]] = None,
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        query_filter = None
        if document_ids:
            query_filter = Filter(
                must=[FieldCondition(key="document_id", match=MatchAny(any=document_ids))]
            )
        try:
            name = _collection_name(workspace_id)
            collections = [c.name for c in self._client.get_collections().collections]
            if name not in collections:
                return []
            results = self._client.query_points(
                collection_name=name,
                query=query_embedding,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            ).points
        except Exception as exc:
            raise VectorStoreError(f"Qdrant search failed: {exc}") from exc

        out: list[RetrievedChunk] = []
        for r in results:
            payload = r.payload or {}
            out.append(
                RetrievedChunk(
                    chunk_id=str(r.id),
                    document_id=payload.get("document_id", ""),
                    document_name=payload.get("document_name", ""),
                    page=int(payload.get("page", 0)),
                    section=payload.get("section") or None,
                    text=payload.get("text", ""),
                    score=max(0.0, min(1.0, float(r.score))),
                )
            )
        return out

    def delete_document(self, workspace_id: str, document_id: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        try:
            self._client.delete(
                collection_name=_collection_name(workspace_id),
                points_selector=FilterSelector(
                    filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
                ),
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete document from Qdrant: {exc}") from exc

    def delete_workspace(self, workspace_id: str) -> None:
        try:
            self._client.delete_collection(collection_name=_collection_name(workspace_id))
        except Exception:
            logger.info("Workspace collection %s did not exist, nothing to delete", workspace_id)

    def health_check(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception as exc:
            logger.warning("Qdrant health check failed: %s", exc)
            return False
