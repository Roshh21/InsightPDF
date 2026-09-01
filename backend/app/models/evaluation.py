from __future__ import annotations

from typing import Optional

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class EvaluationResult(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    Real, computed evaluation metrics for a single agent/tool run. Never
    fabricated -- populated by app/evaluation/tracker.py from the actual
    retrieval scores, citation checks, and timings of that run.
    """

    __tablename__ = "evaluation_results"

    tool_execution_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("tool_executions.id", ondelete="CASCADE"), nullable=True, index=True
    )
    chat_message_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=True, index=True
    )
    run_type: Mapped[str] = mapped_column(String(30), nullable=False)  # "chat" | "tool"

    retrieval_relevance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    answer_faithfulness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    citation_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    retrieval_latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    retries: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
