from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, field_validator

from app.schemas.common import Citation


def _none_to_list(v):
    return v or []


class ChatSessionCreateRequest(BaseModel):
    document_ids: list[str] = []
    spoiler_level: Optional[str] = None
    title: Optional[str] = None


class ChatSessionResponse(BaseModel):
    id: str
    workspace_id: str
    document_ids: list[str] = []
    spoiler_level: Optional[str] = None
    title: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}

    _v_document_ids = field_validator("document_ids", mode="before")(_none_to_list)


class ChatMessageRequest(BaseModel):
    session_id: Optional[str] = None
    document_ids: list[str] = []
    spoiler_level: Optional[str] = None
    message: str


class ChatMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    citations: list[Citation] = []
    tool_used: Optional[str] = None
    result_payload: Optional[dict[str, Any]] = None
    model_used: Optional[str] = None
    provider_used: Optional[str] = None
    latency_ms: Optional[int] = None
    created_at: datetime

    model_config = {"from_attributes": True}

    _v_citations = field_validator("citations", mode="before")(_none_to_list)


class ChatTurnResponse(BaseModel):
    session_id: str
    user_message: ChatMessageResponse
    assistant_message: ChatMessageResponse
