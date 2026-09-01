from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class EvaluationSummaryResponse(BaseModel):
    retrieval_relevance_avg: Optional[float] = None
    answer_faithfulness_avg: Optional[float] = None
    citation_accuracy_avg: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    total_runs: int = 0
    primary_requests: int = 0
    fallback_requests: int = 0
    primary_request_pct: Optional[float] = None
    fallback_request_pct: Optional[float] = None
    provider_breakdown: dict[str, int] = {}
    failed_runs: int = 0
    total_tool_executions: int = 0
    has_data: bool = False


class ProviderStatusResponse(BaseModel):
    slot: str
    provider: str
    configured: bool
    available: bool
    cooling_down: bool
    cooldown_remaining_s: float
    model_fast: str
    model_strong: str


class ModelStatusResponse(BaseModel):
    providers: list[ProviderStatusResponse]
    active_provider: Optional[str] = None
    vector_store: str
    embedding_model: str
