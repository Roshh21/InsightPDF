from __future__ import annotations

from typing import Optional

from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.vectorstores.base import Chunk, RetrievedChunk, VectorStore

logger = get_logger(__name__)


def _collection_name(workspace_id: str) -> str:
    # Chroma requires collection names to be 3-63 chars, alnum/underscore/hyphen.
    safe = workspace_id.replace("-", "")[:40]
    return f"ws_{safe}"


class ChromaStore(VectorStore):
    """
    One Chroma collection per workspace. Documents within a workspace are
    additive (never cleared on new upload) and distinguished via the
    `document_id` metadata field, which every query/delete filters on.
    """

    name = "chroma"

    def __init__(self, persist_directory: str):
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:  # pragma: no cover
            raise VectorStoreError("chromadb is not installed") from exc

        self._chromadb = chromadb
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def _get_collection(self, workspace_id: str):
        return self._client.get_or_create_collection(
            name=_collection_name(workspace_id),
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise VectorStoreError("chunks and embeddings length mismatch")

        by_workspace: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            by_workspace.setdefault(c.workspace_id, []).append(i)

        try:
            for workspace_id, idxs in by_workspace.items():
                collection = self._get_collection(workspace_id)
                collection.upsert(
                    ids=[chunks[i].id for i in idxs],
                    embeddings=[embeddings[i] for i in idxs],
                    documents=[chunks[i].text for i in idxs],
                    metadatas=[chunks[i].metadata() for i in idxs],
                )
        except Exception as exc:
            raise VectorStoreError(f"Failed to index chunks in Chroma: {exc}") from exc

    def search(
        self,
        query_embedding: list[float],
        workspace_id: str,
        document_ids: Optional[list[str]] = None,
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        try:
            collection = self._get_collection(workspace_id)
            where = None
            if document_ids:
                where = {"document_id": {"$in": document_ids}} if len(document_ids) > 1 else {
                    "document_id": document_ids[0]
                }
            count = collection.count()
            if count == 0:
                return []
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, count),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Chroma search failed: {exc}") from exc

        out: list[RetrievedChunk] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for i in range(len(ids)):
            meta = metas[i] or {}
            # Cosine distance in Chroma's hnsw space is ~ (1 - cosine_similarity).
            score = max(0.0, min(1.0, 1.0 - float(dists[i])))
            out.append(
                RetrievedChunk(
                    chunk_id=ids[i],
                    document_id=meta.get("document_id", ""),
                    document_name=meta.get("document_name", ""),
                    page=int(meta.get("page", 0)),
                    section=meta.get("section") or None,
                    text=docs[i],
                    score=score,
                )
            )
        return out

    def delete_document(self, workspace_id: str, document_id: str) -> None:
        try:
            collection = self._get_collection(workspace_id)
            collection.delete(where={"document_id": document_id})
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete document from Chroma: {exc}") from exc

    def delete_workspace(self, workspace_id: str) -> None:
        try:
            self._client.delete_collection(name=_collection_name(workspace_id))
        except Exception:
            logger.info("Workspace collection %s did not exist, nothing to delete", workspace_id)

    def health_check(self) -> bool:
        try:
            self._client.heartbeat()
            return True
        except Exception as exc:
            logger.warning("Chroma health check failed: %s", exc)
            return False
