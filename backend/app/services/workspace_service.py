from __future__ import annotations

import shutil
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import WorkspaceNotFoundError
from app.core.logging import get_logger
from app.models.document import Document
from app.models.workspace import Workspace
from app.vectorstores.factory import get_vector_store

logger = get_logger(__name__)


def create_workspace(db: Session, name: str, description: str | None = None) -> Workspace:
    ws = Workspace(name=name, description=description)
    db.add(ws)
    db.flush()
    return ws


def list_workspaces(db: Session) -> list[Workspace]:
    return list(db.execute(select(Workspace).order_by(Workspace.created_at.desc())).scalars().all())


def get_workspace(db: Session, workspace_id: str) -> Workspace:
    ws = db.get(Workspace, workspace_id)
    if ws is None:
        raise WorkspaceNotFoundError()
    return ws


def workspace_stats(db: Session, workspace_id: str) -> dict:
    docs = list(db.execute(select(Document).where(Document.workspace_id == workspace_id)).scalars().all())
    by_status: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for d in docs:
        by_status[d.status] = by_status.get(d.status, 0) + 1
        if d.document_type:
            by_type[d.document_type] = by_type.get(d.document_type, 0) + 1
    return {"total_documents": len(docs), "by_status": by_status, "by_type": by_type}


def delete_workspace(db: Session, workspace_id: str) -> None:
    ws = get_workspace(db, workspace_id)
    settings = get_settings()

    try:
        get_vector_store().delete_workspace(workspace_id)
    except Exception as exc:
        logger.warning("Failed to delete vector store data for workspace %s: %s", workspace_id, exc)

    workspace_upload_dir = Path(settings.UPLOAD_DIR) / workspace_id
    if workspace_upload_dir.exists():
        shutil.rmtree(workspace_upload_dir, ignore_errors=True)

    db.delete(ws)
