from __future__ import annotations

from typing import Any, Optional, TypedDict

from app.schemas.common import Citation
from app.vectorstores.base import RetrievedChunk


class AgentState(TypedDict, total=False):
    # --- inputs ---
    workspace_id: str
    document_ids: list[str]
    documents: list[Any]  # app.models.document.Document rows
    question: str
    conversation_context: str
    forced_tool_id: Optional[str]
    tool_params: dict[str, Any]
    db: Any  # the request-scoped SQLAlchemy Session, threaded through so tool
    # execution (when chat implicitly routes to a tool) shares one
    # transaction instead of opening a redundant nested session.

    # --- intent / routing ---
    intent: str  # "qa" | "tool"
    tool_id: Optional[str]
    tool_error: Optional[str]

    # --- agentic RAG loop ---
    rewritten_query: str
    attempts: int
    evidence: list[RetrievedChunk]
    retrieval_avg_score: float

    # --- outputs ---
    answer_text: str
    citations: list[Citation]
    tool_result: Optional[dict[str, Any]]
    result_kind: str
    warnings: list[str]
    model_used: Optional[str]
    provider_used: Optional[str]
    fallback_used: bool
    retrieval_latency_ms: int
    total_latency_ms: int
