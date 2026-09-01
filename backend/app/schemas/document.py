from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, field_validator


class DocumentResponse(BaseModel):
    id: str
    workspace_id: str
    original_filename: str
    status: str
    error_message: Optional[str] = None
    document_type: Optional[str] = None
    classification_confidence: Optional[float] = None
    page_count: Optional[int] = None
    capabilities: list[str] = []
    profile: Optional[dict[str, Any]] = None
    created_at: datetime
    processed_at: Optional[str] = None
    retry_count: int = 0

    model_config = {"from_attributes": True}

    @field_validator("capabilities", mode="before")
    @classmethod
    def _default_capabilities(cls, v):
        return v or []


class UploadError(BaseModel):
    filename: str
    message: str


class DocumentUploadResponse(BaseModel):
    documents: list[DocumentResponse]
    errors: list[UploadError] = []


class ToolCatalogEntry(BaseModel):
    id: str
    name: str
    description: str
    category: str
    result_kind: str
    requires_multi_document: bool
    icon: str
