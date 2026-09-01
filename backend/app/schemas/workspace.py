from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class WorkspaceCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None


class WorkspaceResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class WorkspaceStatsResponse(BaseModel):
    total_documents: int
    by_status: dict[str, int]
    by_type: dict[str, int]


class WorkspaceDetailResponse(WorkspaceResponse):
    stats: WorkspaceStatsResponse
