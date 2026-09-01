from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.core.enums import DocumentType, ResultKind
from app.llm.gateway import ModelGateway
from app.retrieval.retriever import Retriever
from app.schemas.common import ToolResultEnvelope


@dataclass
class ToolContext:
    workspace_id: str
    document_ids: list[str]
    documents: list[Any]  # app.models.document.Document rows in scope, for names/types/profile access
    db: Session
    gateway: ModelGateway
    retriever: Retriever
    params: dict[str, Any] = field(default_factory=dict)
    conversation_context: str = ""
    # Populated by tools after retrieval so the caller (tool_service) can
    # feed real retrieval scores/text into the evaluation tracker without
    # re-running retrieval a second time.
    last_evidence: list[Any] = field(default_factory=list)

    def document_name(self, document_id: str) -> str:
        for d in self.documents:
            if d.id == document_id:
                return d.original_filename
        return document_id


class BaseTool(ABC):
    id: str
    name: str
    description: str
    category: str = "universal"
    result_kind: ResultKind = ResultKind.SECTIONS
    applicable_types: Optional[list[DocumentType]] = None  # None => applies to every type
    requires_multi_document: bool = False
    icon: str = "sparkles"

    @abstractmethod
    def run(self, ctx: ToolContext) -> ToolResultEnvelope:
        ...

    def catalog_entry(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "result_kind": self.result_kind.value,
            "requires_multi_document": self.requires_multi_document,
            "icon": self.icon,
        }
