from __future__ import annotations

from typing import Any, List, Optional

from sqlalchemy import JSON, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import ToolExecutionStatus
from app.database import Base
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class ToolExecution(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A single agent/tool run -- powers evaluation, observability and history."""

    __tablename__ = "tool_executions"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_ids: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True, default=list)

    tool_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    params: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    status: Mapped[str] = mapped_column(String(20), default=ToolExecutionStatus.PENDING.value, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    result_kind: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    result: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    citations: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(JSON, nullable=True, default=list)

    model_used: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    provider_used: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    fallback_used: Mapped[bool] = mapped_column(default=False, nullable=False)

    retrieval_attempts: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tokens_in: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    started_at: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    completed_at: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)

    evaluation_results: Mapped[List["EvaluationResult"]] = relationship(  # noqa: F821
        cascade="all, delete-orphan", passive_deletes=True
    )
