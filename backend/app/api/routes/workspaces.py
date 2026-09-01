from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.workspace import (
    WorkspaceCreateRequest,
    WorkspaceDetailResponse,
    WorkspaceResponse,
    WorkspaceStatsResponse,
)
from app.services import workspace_service

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.post("", response_model=WorkspaceResponse)
def create_workspace(payload: WorkspaceCreateRequest, db: Session = Depends(get_db)):
    ws = workspace_service.create_workspace(db, name=payload.name, description=payload.description)
    db.commit()
    db.refresh(ws)
    return ws


@router.get("", response_model=list[WorkspaceResponse])
def list_workspaces(db: Session = Depends(get_db)):
    return workspace_service.list_workspaces(db)


@router.get("/{workspace_id}", response_model=WorkspaceDetailResponse)
def get_workspace(workspace_id: str, db: Session = Depends(get_db)):
    ws = workspace_service.get_workspace(db, workspace_id)
    stats = workspace_service.workspace_stats(db, workspace_id)
    return WorkspaceDetailResponse(
        id=ws.id,
        name=ws.name,
        description=ws.description,
        created_at=ws.created_at,
        stats=WorkspaceStatsResponse(**stats),
    )


@router.delete("/{workspace_id}")
def delete_workspace(workspace_id: str, db: Session = Depends(get_db)):
    workspace_service.delete_workspace(db, workspace_id)
    db.commit()
    return {"deleted": True}
