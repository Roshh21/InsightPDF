from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import DocumentType
from app.database import get_db
from app.models.tool_execution import ToolExecution
from app.schemas.tool import ToolExecuteRequest, ToolExecuteResponse, ToolExecutionHistoryItem
from app.services import workspace_service
from app.services.tool_service import execute_tool
from app.tools.registry import catalog, ensure_loaded

router = APIRouter(tags=["tools"])


@router.get("/tools/catalog")
def get_tool_catalog(document_type: Optional[DocumentType] = Query(default=None)):
    ensure_loaded()
    return catalog(document_type)


@router.post("/workspaces/{workspace_id}/tools/execute", response_model=ToolExecuteResponse)
def run_tool(workspace_id: str, payload: ToolExecuteRequest, db: Session = Depends(get_db)):
    workspace_service.get_workspace(db, workspace_id)
    result, execution = execute_tool(
        db,
        workspace_id=workspace_id,
        document_ids=payload.document_ids or None,
        tool_id=payload.tool_id,
        params=payload.params,
    )
    db.commit()
    return ToolExecuteResponse(
        execution_id=execution.id,
        tool_name=result.tool_name,
        result_kind=result.result_kind.value,
        title=result.title,
        content=result.content,
        citations=result.citations,
        warnings=result.warnings,
        latency_ms=execution.latency_ms,
        model_used=execution.model_used,
        provider_used=execution.provider_used,
        fallback_used=execution.fallback_used,
    )


@router.get("/workspaces/{workspace_id}/tools/executions", response_model=list[ToolExecutionHistoryItem])
def list_tool_executions(workspace_id: str, limit: int = 50, db: Session = Depends(get_db)):
    rows = db.execute(
        select(ToolExecution)
        .where(ToolExecution.workspace_id == workspace_id)
        .order_by(ToolExecution.created_at.desc())
        .limit(limit)
    ).scalars().all()
    return list(rows)


@router.get("/tools/executions/{execution_id}", response_model=ToolExecuteResponse)
def get_tool_execution(execution_id: str, db: Session = Depends(get_db)):
    execution = db.get(ToolExecution, execution_id)
    if execution is None:
        from app.core.exceptions import ToolNotFoundError

        raise ToolNotFoundError("Execution not found")
    return ToolExecuteResponse(
        execution_id=execution.id,
        tool_name=execution.tool_name,
        result_kind=execution.result_kind or "text",
        title=execution.tool_name,
        content=execution.result,
        citations=execution.citations or [],
        warnings=[],
        latency_ms=execution.latency_ms,
        model_used=execution.model_used,
        provider_used=execution.provider_used,
        fallback_used=execution.fallback_used,
    )
