from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.core.exceptions import DocumentValidationError
from app.database import get_db
from app.ingestion.pipeline import process_document
from app.schemas.document import DocumentResponse, DocumentUploadResponse, UploadError
from app.services import document_service, workspace_service

router = APIRouter(tags=["documents"])


@router.post("/workspaces/{workspace_id}/documents", response_model=DocumentUploadResponse)
async def upload_documents(
    workspace_id: str,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    workspace_service.get_workspace(db, workspace_id)  # 404s if missing

    created: list = []
    errors: list[UploadError] = []
    for f in files:
        content = await f.read()
        try:
            document_service.validate_upload(f.filename, content)
        except DocumentValidationError as exc:
            # Independent per-file handling: one file failing validation
            # (wrong type, empty, too large, not a real PDF) must never
            # block or hide the others -- it's reported back per-filename
            # instead of collapsing into one generic upload error.
            errors.append(UploadError(filename=f.filename, message=exc.user_message))
            continue
        stored_name, path, digest = document_service.save_upload(workspace_id, f.filename, content)
        doc = document_service.create_document(
            db,
            workspace_id=workspace_id,
            original_filename=f.filename,
            stored_filename=stored_name,
            file_path=path,
            file_hash=digest,
        )
        created.append(doc)

    db.commit()
    for doc in created:
        db.refresh(doc)
        background_tasks.add_task(process_document, doc.id)

    if not created and errors:
        # Every single file failed validation -- surface it as a proper
        # error rather than a silent empty success.
        raise DocumentValidationError("; ".join(f"{e.filename}: {e.message}" for e in errors))

    return DocumentUploadResponse(documents=[DocumentResponse.model_validate(d) for d in created], errors=errors)


@router.get("/workspaces/{workspace_id}/documents", response_model=list[DocumentResponse])
def list_documents(workspace_id: str, db: Session = Depends(get_db)):
    workspace_service.get_workspace(db, workspace_id)
    return document_service.list_documents(db, workspace_id)


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str, db: Session = Depends(get_db)):
    return document_service.get_document(db, document_id)


@router.get("/documents/{document_id}/status")
def get_document_status(document_id: str, db: Session = Depends(get_db)):
    doc = document_service.get_document(db, document_id)
    return {"id": doc.id, "status": doc.status, "error_message": doc.error_message, "retry_count": doc.retry_count}


@router.post("/documents/{document_id}/retry", response_model=DocumentResponse)
def retry_document(document_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """A failed (or stuck) document must never be a dead end -- this resets
    it and re-queues the same independent, per-document pipeline used on
    first upload. Idempotent: safe to call repeatedly."""
    doc = document_service.retry_document(db, document_id)
    db.commit()
    db.refresh(doc)
    background_tasks.add_task(process_document, doc.id)
    return doc


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db)):
    document_service.delete_document(db, document_id)
    db.commit()
    return {"deleted": True}
