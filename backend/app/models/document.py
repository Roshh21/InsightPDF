from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import DocumentType, ProcessingStatus
from app.database import Base
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class Document(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "documents"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)

    status: Mapped[str] = mapped_column(String(20), default=ProcessingStatus.UPLOADED.value, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    document_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    classification_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Structured document profile: sections, entities, topics, tables,
    # title/authors/metadata, etc. See classification/profiles.py for shape.
    profile: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # List[str] of tool ids this document supports, derived from profile.type
    capabilities: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True, default=list)

    processed_at: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    workspace: Mapped["Workspace"] = relationship(back_populates="documents")  # noqa: F821

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Document {self.id} {self.original_filename!r} status={self.status}>"
