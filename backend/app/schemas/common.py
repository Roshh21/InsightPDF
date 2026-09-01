from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from app.core.enums import ResultKind


class Citation(BaseModel):
    document_id: str
    document_name: str
    page: int
    section: Optional[str] = None
    excerpt: str = Field(description="Short (<300 char) verbatim-ish excerpt supporting the claim")
    supported: Optional[bool] = Field(
        default=None, description="Filled in by evidence validation: does the excerpt actually appear in the source?"
    )


class ToolResultEnvelope(BaseModel):
    """Uniform wrapper every tool execution and agentic-RAG answer returns,
    so the frontend has one renderer contract regardless of which tool ran."""

    tool_name: str
    result_kind: ResultKind
    title: Optional[str] = None
    content: Any = Field(description="Shape depends on result_kind -- see frontend lib/types.ts")
    citations: list[Citation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
