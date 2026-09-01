from __future__ import annotations

import time
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.enums import ToolExecutionStatus
from app.core.exceptions import InsightPDFError, ToolExecutionError, ToolNotFoundError
from app.core.logging import get_logger
from app.embeddings.factory import get_embedding_provider
from app.evaluation.tracker import record_run_evaluation
from app.llm.gateway import get_model_gateway
from app.models.tool_execution import ToolExecution
from app.retrieval.retriever import Retriever
from app.schemas.common import ToolResultEnvelope
from app.services.document_service import resolve_documents
from app.services.model_usage_lookup import latest_successful_usage_since
from app.tools.base import ToolContext
from app.tools.registry import ensure_loaded, get_tool
from app.vectorstores.factory import get_vector_store

logger = get_logger(__name__)


def _text_for_eval(content) -> str:
    if isinstance(content, dict):
        for key in ("answer", "summary", "narrative"):
            if content.get(key):
                base = str(content[key])
                break
        else:
            base = ""
        sections = content.get("sections") or []
        extra = " ".join(s.get("content", "") for s in sections if isinstance(s, dict))
        return (base + " " + extra).strip()[:4000]
    return str(content)[:4000]


def execute_tool(
    db: Session,
    *,
    workspace_id: str,
    document_ids: list[str] | None,
    tool_id: str,
    params: dict,
) -> tuple[ToolResultEnvelope, ToolExecution]:
    ensure_loaded()
    tool = get_tool(tool_id)
    if tool is None:
        raise ToolNotFoundError(f"Unknown tool id: {tool_id}")

    documents = resolve_documents(db, workspace_id, document_ids)
    if not documents:
        raise ToolExecutionError(
            user_message="No processed (READY) documents are available in scope for this tool."
        )

    if tool.applicable_types is not None:
        allowed = {t.value for t in tool.applicable_types}
        applicable_docs = [d for d in documents if d.document_type in allowed]
        if not applicable_docs:
            raise ToolExecutionError(
                user_message=f"'{tool.name}' isn't applicable to the selected document type(s)."
            )
        documents = applicable_docs

    execution = ToolExecution(
        workspace_id=workspace_id,
        document_ids=[d.id for d in documents],
        tool_name=tool_id,
        params=params,
        status=ToolExecutionStatus.RUNNING.value,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    db.add(execution)
    db.commit()
    db.refresh(execution)

    gateway = get_model_gateway()
    retriever = Retriever(get_vector_store(), get_embedding_provider())
    ctx = ToolContext(
        workspace_id=workspace_id,
        document_ids=[d.id for d in documents],
        documents=documents,
        db=db,
        gateway=gateway,
        retriever=retriever,
        params=params,
    )

    start_monotonic = time.monotonic()
    start_dt = datetime.now(timezone.utc)
    try:
        result = tool.run(ctx)
    except InsightPDFError as exc:
        execution.status = ToolExecutionStatus.FAILED.value
        execution.error_message = str(exc.user_message)
        execution.completed_at = datetime.now(timezone.utc).isoformat()
        db.commit()
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Tool '%s' failed unexpectedly", tool_id)
        execution.status = ToolExecutionStatus.FAILED.value
        execution.error_message = str(exc)
        execution.completed_at = datetime.now(timezone.utc).isoformat()
        db.commit()
        raise ToolExecutionError(f"Tool '{tool_id}' failed: {exc}") from exc

    latency_ms = int((time.monotonic() - start_monotonic) * 1000)
    usage = latest_successful_usage_since(db, start_dt)

    execution.status = ToolExecutionStatus.SUCCEEDED.value
    execution.result_kind = result.result_kind.value
    execution.result = result.content if isinstance(result.content, dict) else {"value": result.content}
    execution.citations = [c.model_dump(mode="json") for c in result.citations]
    execution.latency_ms = latency_ms
    execution.completed_at = datetime.now(timezone.utc).isoformat()
    if usage:
        execution.model_used = usage.model
        execution.provider_used = usage.provider_used
        execution.fallback_used = usage.fallback_used
        execution.tokens_in = usage.tokens_in
        execution.tokens_out = usage.tokens_out
    db.flush()

    record_run_evaluation(
        db,
        run_type="tool",
        answer_text=_text_for_eval(result.content),
        evidence_chunks=list(ctx.last_evidence),
        citations=result.citations,
        total_latency_ms=latency_ms,
        tool_execution_id=execution.id,
    )
    db.commit()

    return result, execution
