"""
Document ingestion pipeline.

Upload -> Validate -> Parse -> Extract pages -> Detect structure -> Classify
-> Extract metadata -> Extract tables -> Chunk -> Embed -> Index ->
Generate profile -> Expose capabilities.

Each stage is isolated so a failure produces a clear FAILED status with a
useful `error_message` instead of a raw stack trace bubbling to the user.
Runs as a FastAPI BackgroundTask, kicked off right after upload.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.classification.classifier import classify_document
from app.classification.profiles import DocumentProfile
from app.core.enums import ProcessingStatus
from app.core.exceptions import DocumentValidationError, InsightPDFError
from app.core.logging import get_logger
from app.database import session_scope
from app.embeddings.factory import get_embedding_provider
from app.extraction.table_extractor import detect_tables
from app.ingestion.chunker import chunk_document
from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.structure_detector import detect_sections
from app.llm.gateway import get_model_gateway
from app.models.document import Document
from app.vectorstores.factory import get_vector_store

logger = get_logger(__name__)


def _set_status(document_id: str, status: ProcessingStatus, error_message: str | None = None) -> None:
    with session_scope() as db:
        doc = db.get(Document, document_id)
        if doc is None:
            return
        doc.status = status.value
        doc.error_message = error_message


def process_document(document_id: str) -> None:
    logger.info("Starting ingestion pipeline for document %s", document_id)
    _set_status(document_id, ProcessingStatus.PROCESSING)

    with session_scope() as db:
        doc = db.get(Document, document_id)
        if doc is None:
            logger.error("Document %s vanished before processing", document_id)
            return
        file_path = doc.file_path
        workspace_id = doc.workspace_id
        original_filename = doc.original_filename

    try:
        # 0. Idempotency: clear any vectors from a previous attempt (first
        # run or a partial failure) before indexing anything new. Without
        # this, retrying a failed/partial document would append a second,
        # duplicate set of chunks under fresh chunk ids alongside the old
        # ones rather than replacing them.
        try:
            get_vector_store().delete_document(workspace_id, document_id)
        except Exception as exc:
            logger.warning("Could not clear prior vectors for document %s before (re)processing: %s", document_id, exc)

        # 1-3. Validate + parse + extract pages
        parsed = parse_pdf(file_path)

        # 4. Detect structure (cheap heuristic pass)
        sections = detect_sections(parsed)

        # 5. Classify + extract metadata/entities/topics (LLM, structured)
        gateway = get_model_gateway()
        profile: DocumentProfile = classify_document(gateway, parsed, original_filename)
        if sections:
            profile.sections = sections

        # 6. Extract tables where applicable
        profile.tables = detect_tables(file_path)
        if not profile.title:
            profile.title = original_filename.rsplit(".", 1)[0]

        # 7. Chunk
        chunks = chunk_document(
            parsed,
            workspace_id=workspace_id,
            document_id=document_id,
            document_type=profile.type.value,
            document_name=original_filename,
            sections=profile.sections,
        )
        if not chunks:
            raise DocumentValidationError(
                "No usable text chunks could be produced from this document",
                user_message=(
                    "This document was parsed, but no usable text content could be extracted from it "
                    "(pages may be mostly blank, formatting-only, or extremely short)."
                ),
            )

        # 8. Embed
        embedder = get_embedding_provider()
        texts = [c.text for c in chunks]
        vectors = embedder.embed_documents(texts)

        # 9. Index
        store = get_vector_store()
        store.add_chunks(chunks, vectors)

        # 10-11. Persist profile + capabilities, mark READY
        with session_scope() as db:
            doc = db.get(Document, document_id)
            if doc is None:
                return
            doc.document_type = profile.type.value
            doc.classification_confidence = profile.confidence
            doc.page_count = parsed.page_count
            doc.profile = profile.model_dump(mode="json")
            doc.capabilities = profile.capabilities
            doc.status = ProcessingStatus.READY.value
            doc.error_message = None
            doc.processed_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Finished ingestion for document %s: type=%s chunks=%d",
            document_id,
            profile.type.value,
            len(chunks),
        )

    except InsightPDFError as exc:
        logger.warning("Ingestion failed for document %s: %s", document_id, exc)
        _set_status(document_id, ProcessingStatus.FAILED, str(exc.user_message))
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Unexpected ingestion failure for document %s", document_id)
        _set_status(document_id, ProcessingStatus.FAILED, f"Unexpected error: {exc}")
