from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, field_validator

from app.schemas.common import Citation


class ToolExecuteRequest(BaseModel):
    tool_id: str
    document_ids: list[str] = []
    params: dict[str, Any] = {}


class ToolExecuteResponse(BaseModel):
    execution_id: str
    tool_name: str
    result_kind: str
    title: Optional[str] = None
    content: Any
    citations: list[Citation] = []
    warnings: list[str] = []
    latency_ms: Optional[int] = None
    model_used: Optional[str] = None
    provider_used: Optional[str] = None
    fallback_used: bool = False

    @field_validator("citations", "warnings", mode="before")
    @classmethod
    def _none_to_list(cls, v):
        return v or []


class ToolExecutionHistoryItem(BaseModel):
    id: str
    tool_name: str
    status: str
    document_ids: list[str] = []
    latency_ms: Optional[int] = None
    model_used: Optional[str] = None
    provider_used: Optional[str] = None
    fallback_used: bool = False
    error_message: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}

    @field_validator("document_ids", mode="before")
    @classmethod
    def _none_to_list(cls, v):
        return v or []
