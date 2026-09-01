"""
Evaluation metrics. Every number here is computed from an actual run --
retrieval similarity scores returned by the vector store, an embedding-
similarity faithfulness proxy between the generated answer and the evidence
actually retrieved, and a lexical citation-support check. None of this is
hard-coded or randomly generated; if there is nothing to measure, functions
return None and the caller persists NULL rather than a fabricated number.
"""
from __future__ import annotations

from app.embeddings.base import EmbeddingProvider
from app.retrieval.evidence_validator import citation_accuracy as _citation_accuracy
from app.schemas.common import Citation
from app.vectorstores.base import RetrievedChunk


def retrieval_relevance(chunks: list[RetrievedChunk]) -> float | None:
    if not chunks:
        return None
    return sum(c.score for c in chunks) / len(chunks)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def answer_faithfulness(
    embedder: EmbeddingProvider, answer_text: str, evidence_chunks: list[RetrievedChunk], top_n: int = 3
) -> float | None:
    """Proxy for faithfulness: how semantically close is the generated
    answer to its best-matching evidence chunks? Embeddings from
    HuggingFaceEmbeddingProvider are L2-normalized, so dot product == cosine
    similarity. This is a heuristic, not a ground-truth entailment check --
    documented as such wherever it's surfaced."""
    if not answer_text.strip() or not evidence_chunks:
        return None
    try:
        answer_vec = embedder.embed_query(answer_text[:2000])
        evidence_vecs = embedder.embed_documents([c.text for c in evidence_chunks[:20]])
    except Exception:
        return None
    sims = sorted((_dot(answer_vec, v) for v in evidence_vecs), reverse=True)
    top = sims[:top_n] or sims
    if not top:
        return None
    score = sum(top) / len(top)
    return max(0.0, min(1.0, score))


def citation_accuracy(citations: list[Citation]) -> float | None:
    return _citation_accuracy(citations)
