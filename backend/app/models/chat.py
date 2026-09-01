from __future__ import annotations

from typing import Any, List, Optional

from sqlalchemy import JSON, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class ChatSession(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "chat_sessions"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Which documents this session is scoped to. Empty/None = whole workspace.
    document_ids: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True, default=list)
    spoiler_level: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    workspace: Mapped["Workspace"] = relationship(back_populates="chat_sessions")  # noqa: F821
    messages: Mapped[List["ChatMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ChatMessage.created_at",
    )


class ChatMessage(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "chat_messages"

    session_id: Mapped[str] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    citations: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(JSON, nullable=True, default=list)
    tool_used: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    result_payload: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    model_used: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    provider_used: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(nullable=True)

    session: Mapped["ChatSession"] = relationship(back_populates="messages")
