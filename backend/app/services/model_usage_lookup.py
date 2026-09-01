from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.model_usage import ModelUsage


def latest_successful_usage_since(db: Session, since: datetime) -> Optional[ModelUsage]:
    """The ModelGateway logs every provider call independently of which
    tool/chat turn triggered it (so it stays accurate even for multi-call
    tools). For UI/observability display on a single run, we attribute it
    to the most recent successful call made during that run's time window --
    a best-effort approximation, not a strict causal link."""
    return db.execute(
        select(ModelUsage)
        .where(ModelUsage.created_at >= since, ModelUsage.success.is_(True))
        .order_by(ModelUsage.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
