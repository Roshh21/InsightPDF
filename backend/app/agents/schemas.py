from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.common import Citation


class IntentDecision(BaseModel):
    intent: str = Field(description="'tool' if the message clearly requests one specific known capability, else 'qa'")
    tool_id: Optional[str] = Field(default=None, description="Must be one of the listed available tool ids, or null")
    reasoning: str = ""


class RagAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
