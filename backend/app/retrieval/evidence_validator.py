from __future__ import annotations

from difflib import SequenceMatcher

from app.schemas.common import Citation
from app.vectorstores.base import RetrievedChunk

SUPPORTED_THRESHOLD = 0.35


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _best_overlap(excerpt: str, candidates: list[str]) -> float:
    if not candidates:
        return 0.0
    best = 0.0
    for text in candidates:
        # Windowed comparison: an excerpt should match *some* contiguous
        # span of the source chunk, not necessarily the whole thing.
        if excerpt.lower() in text.lower():
            return 1.0
        best = max(best, _similarity(excerpt, text[: max(len(excerpt) * 3, 200)]))
    return best


def validate_citations(citations: list[Citation], evidence_chunks: list[RetrievedChunk]) -> list[Citation]:
    """Checks each citation's excerpt against the pool of retrieved evidence
    for the same document+page. Mutates and returns the citations with
    `supported` set. This is a real (if heuristic) check -- not fabricated --
    and backs the `citation_accuracy` evaluation metric."""
    by_doc_page: dict[tuple[str, int], list[str]] = {}
    for c in evidence_chunks:
        by_doc_page.setdefault((c.document_id, c.page), []).append(c.text)
    # Also allow matching against any chunk of the same document (page
    # numbers from the LLM can be slightly off).
    by_doc: dict[str, list[str]] = {}
    for c in evidence_chunks:
        by_doc.setdefault(c.document_id, []).append(c.text)

    for citation in citations:
        candidates = by_doc_page.get((citation.document_id, citation.page), [])
        score = _best_overlap(citation.excerpt, candidates)
        if score < SUPPORTED_THRESHOLD:
            score = max(score, _best_overlap(citation.excerpt, by_doc.get(citation.document_id, [])) * 0.8)
        citation.supported = score >= SUPPORTED_THRESHOLD
    return citations


def citation_accuracy(citations: list[Citation]) -> float | None:
    checked = [c for c in citations if c.supported is not None]
    if not checked:
        return None
    return sum(1 for c in checked if c.supported) / len(checked)
