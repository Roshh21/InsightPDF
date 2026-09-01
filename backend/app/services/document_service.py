from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.enums import ProcessingStatus
from app.core.exceptions import DocumentNotFoundError, DocumentValidationError, EmptyDocumentError
from app.core.logging import get_logger
from app.models.document import Document
from app.vectorstores.factory import get_vector_store

logger = get_logger(__name__)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_filename(name: str) -> str:
    return _SAFE_NAME_RE.sub("_", name)[-150:]


def validate_upload(filename: str, content: bytes) -> None:
    settings = get_settings()
    if not filename.lower().endswith(".pdf"):
        raise DocumentValidationError(
            f"'{filename}' is not a .pdf file", user_message=f"'{filename}' was skipped: only .pdf files are supported."
        )
    if len(content) == 0:
        raise EmptyDocumentError(f"'{filename}' is empty", user_message=f"'{filename}' was skipped: the file is empty.")
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise DocumentValidationError(
            f"'{filename}' exceeds the {settings.MAX_UPLOAD_MB}MB upload limit",
            user_message=f"'{filename}' was skipped: it exceeds the {settings.MAX_UPLOAD_MB}MB upload limit.",
        )
    if content[:4] != b"%PDF":
        raise DocumentValidationError(
            f"'{filename}' does not look like a valid PDF file",
            user_message=f"'{filename}' was skipped: it doesn't look like a valid PDF file.",
        )


def save_upload(workspace_id: str, filename: str, content: bytes) -> tuple[str, str, str]:
    settings = get_settings()
    workspace_dir = Path(settings.UPLOAD_DIR) / workspace_id
    workspace_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}_{_safe_filename(filename)}"
    path = workspace_dir / stored_name
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    return stored_name, str(path), digest


def create_document(
    db: Session, *, workspace_id: str, original_filename: str, stored_filename: str, file_path: str, file_hash: str
) -> Document:
    doc = Document(
        workspace_id=workspace_id,
        filename=stored_filename,
        original_filename=original_filename,
        file_path=file_path,
        file_hash=file_hash,
        status=ProcessingStatus.UPLOADED.value,
    )
    db.add(doc)
    db.flush()
    return doc


def get_document(db: Session, document_id: str) -> Document:
    doc = db.get(Document, document_id)
    if doc is None:
        raise DocumentNotFoundError()
    return doc


def list_documents(db: Session, workspace_id: str) -> list[Document]:
    return list(
        db.execute(
            select(Document).where(Document.workspace_id == workspace_id).order_by(Document.created_at.desc())
        ).scalars().all()
    )


def resolve_documents(db: Session, workspace_id: str, document_ids: list[str] | None) -> list[Document]:
    """Empty/None document_ids means 'the whole workspace' -- resolved here
    to every READY document, never an unprocessed or failed one."""
    if document_ids:
        docs = [get_document(db, did) for did in document_ids]
        for d in docs:
            if d.workspace_id != workspace_id:
                raise DocumentValidationError("Document does not belong to this workspace")
        return docs
    return [d for d in list_documents(db, workspace_id) if d.status == ProcessingStatus.READY.value]


def retry_document(db: Session, document_id: str) -> Document:
    """Resets a document (typically FAILED, but allowed from any state) back
    to UPLOADED so the background pipeline picks it up again. Any vectors
    from a prior attempt are cleared idempotently by process_document itself
    before it re-indexes, so retrying never duplicates chunks."""
    doc = get_document(db, document_id)
    doc.status = ProcessingStatus.UPLOADED.value
    doc.error_message = None
    doc.retry_count = (doc.retry_count or 0) + 1
    db.flush()
    return doc


def delete_document(db: Session, document_id: str) -> None:
    doc = get_document(db, document_id)
    try:
        get_vector_store().delete_document(doc.workspace_id, doc.id)
    except Exception as exc:
        logger.warning("Failed to delete vectors for document %s: %s", document_id, exc)
    try:
        Path(doc.file_path).unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Failed to delete file for document %s: %s", document_id, exc)
    db.delete(doc)
