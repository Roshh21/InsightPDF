from __future__ import annotations

from typing import Optional

from sqlalchemy import Boolean, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class ModelUsage(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One row per LLM call made through the ModelGateway. Real, recorded usage --
    this is what the evaluation dashboard's cloud/local ratio is computed from."""

    __tablename__ = "model_usage"

    task_type: Mapped[str] = mapped_column(String(50), nullable=False)
    provider_requested: Mapped[str] = mapped_column(String(20), nullable=False)
    provider_used: Mapped[str] = mapped_column(String(20), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)

    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    error_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tokens_in: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
