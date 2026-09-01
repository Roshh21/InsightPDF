from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.evaluation.tracker import get_evaluation_summary
from app.schemas.evaluation import EvaluationSummaryResponse

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get("/summary", response_model=EvaluationSummaryResponse)
def evaluation_summary(db: Session = Depends(get_db)):
    summary = get_evaluation_summary(db)
    return EvaluationSummaryResponse(
        retrieval_relevance_avg=summary.retrieval_relevance_avg,
        answer_faithfulness_avg=summary.answer_faithfulness_avg,
        citation_accuracy_avg=summary.citation_accuracy_avg,
        avg_latency_ms=summary.avg_latency_ms,
        total_runs=summary.total_runs,
        primary_requests=summary.primary_requests,
        fallback_requests=summary.fallback_requests,
        primary_request_pct=summary.primary_request_pct,
        fallback_request_pct=summary.fallback_request_pct,
        provider_breakdown=summary.provider_breakdown,
        failed_runs=summary.failed_runs,
        total_tool_executions=summary.total_tool_executions,
        has_data=summary.total_runs > 0,
    )
