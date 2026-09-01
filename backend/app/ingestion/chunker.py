from __future__ import annotations

import re
import uuid

from app.classification.profiles import DocumentSection
from app.ingestion.pdf_parser import ParsedDocument
from app.vectorstores.base import Chunk

_PARA_SPLIT_RE = re.compile(r"\n\s*\n")


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily pack paragraphs into ~chunk_size windows with overlap.
    Falls back to hard character slicing for pathologically long paragraphs."""
    paragraphs = [p.strip() for p in _PARA_SPLIT_RE.split(text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(para), chunk_size - chunk_overlap):
                chunks.append(para[i : i + chunk_size])
            continue
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > chunk_size and current:
            chunks.append(current)
            # start next chunk with overlap tail of previous
            tail = current[-chunk_overlap:] if chunk_overlap else ""
            current = f"{tail}\n\n{para}" if tail else para
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _section_for_page(sections: list[DocumentSection], page: int) -> str | None:
    for s in sections:
        end = s.page_end or s.page_start
        if s.page_start <= page <= end:
            return s.title
    return None


def chunk_document(
    parsed: ParsedDocument,
    *,
    workspace_id: str,
    document_id: str,
    document_type: str,
    document_name: str,
    sections: list[DocumentSection] | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    min_chars: int = 30,
) -> list[Chunk]:
    sections = sections or []
    chunks: list[Chunk] = []
    chunk_index = 0
    for page in parsed.pages:
        if len(page.text.strip()) < min_chars:
            continue
        section_title = _section_for_page(sections, page.page_number)
        for piece in _split_text(page.text, chunk_size, chunk_overlap):
            if len(piece.strip()) < min_chars:
                continue
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    workspace_id=workspace_id,
                    document_id=document_id,
                    text=piece.strip(),
                    page=page.page_number,
                    section=section_title,
                    chunk_index=chunk_index,
                    document_type=document_type,
                    document_name=document_name,
                )
            )
            chunk_index += 1
    return chunks
