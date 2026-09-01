from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.embeddings.factory import get_embedding_provider
from app.evaluation import metrics as m
from app.models.evaluation import EvaluationResult
from app.models.model_usage import ModelUsage
from app.models.tool_execution import ToolExecution
from app.schemas.common import Citation
from app.vectorstores.base import RetrievedChunk

logger = get_logger(__name__)


def record_run_evaluation(
    db: Session,
    *,
    run_type: str,
    answer_text: str,
    evidence_chunks: list[RetrievedChunk],
    citations: list[Citation],
    total_latency_ms: int,
    retrieval_latency_ms: Optional[int] = None,
    retries: int = 0,
    tool_execution_id: Optional[str] = None,
    chat_message_id: Optional[str] = None,
) -> Optional[EvaluationResult]:
    settings = get_settings()
    if not settings.EVALUATION_ENABLED:
        return None
    try:
        embedder = get_embedding_provider()
        result = EvaluationResult(
            tool_execution_id=tool_execution_id,
            chat_message_id=chat_message_id,
            run_type=run_type,
            retrieval_relevance=m.retrieval_relevance(evidence_chunks),
            answer_faithfulness=m.answer_faithfulness(embedder, answer_text, evidence_chunks),
            citation_accuracy=m.citation_accuracy(citations),
            retrieval_latency_ms=retrieval_latency_ms,
            total_latency_ms=total_latency_ms,
            retries=retries,
        )
        db.add(result)
        db.flush()
        return result
    except Exception as exc:  # observability must never break the user-facing request
        logger.warning("Failed to record evaluation: %s", exc)
        return None


@dataclass
class EvaluationSummary:
    retrieval_relevance_avg: Optional[float]
    answer_faithfulness_avg: Optional[float]
    citation_accuracy_avg: Optional[float]
    avg_latency_ms: Optional[float]
    total_runs: int
    primary_requests: int
    fallback_requests: int
    primary_request_pct: Optional[float]
    fallback_request_pct: Optional[float]
    provider_breakdown: dict[str, int]
    failed_runs: int
    total_tool_executions: int


def get_evaluation_summary(db: Session, limit_recent: int = 500) -> EvaluationSummary:
    """Aggregates real recorded rows. Returns None-valued fields (never a
    fabricated 0.0 or a made-up percentage) when there is genuinely no data
    yet -- the frontend renders "No data yet" in that case."""

    eval_rows = db.execute(
        select(EvaluationResult).order_by(EvaluationResult.created_at.desc()).limit(limit_recent)
    ).scalars().all()

    def _avg(values: list[float]) -> Optional[float]:
        vals = [v for v in values if v is not None]
        return sum(vals) / len(vals) if vals else None

    retrieval_avg = _avg([r.retrieval_relevance for r in eval_rows])
    faithfulness_avg = _avg([r.answer_faithfulness for r in eval_rows])
    citation_avg = _avg([r.citation_accuracy for r in eval_rows])
    latency_avg = _avg([r.total_latency_ms for r in eval_rows])

    usage_rows = db.execute(
        select(ModelUsage).order_by(ModelUsage.created_at.desc()).limit(limit_recent)
    ).scalars().all()
    successful_rows = [u for u in usage_rows if u.success]
    fallback_requests = sum(1 for u in successful_rows if u.fallback_used)
    primary_requests = len(successful_rows) - fallback_requests
    total_successful = len(successful_rows)
    failed_runs = sum(1 for u in usage_rows if not u.success)

    provider_breakdown: dict[str, int] = {}
    for u in successful_rows:
        provider_breakdown[u.provider_used] = provider_breakdown.get(u.provider_used, 0) + 1

    total_tool_executions = db.execute(select(func.count()).select_from(ToolExecution)).scalar_one()

    return EvaluationSummary(
        retrieval_relevance_avg=retrieval_avg,
        answer_faithfulness_avg=faithfulness_avg,
        citation_accuracy_avg=citation_avg,
        avg_latency_ms=latency_avg,
        total_runs=len(eval_rows),
        primary_requests=primary_requests,
        fallback_requests=fallback_requests,
        primary_request_pct=(primary_requests / total_successful * 100) if total_successful else None,
        fallback_request_pct=(fallback_requests / total_successful * 100) if total_successful else None,
        provider_breakdown=provider_breakdown,
        failed_runs=failed_runs,
        total_tool_executions=total_tool_executions,
    )
