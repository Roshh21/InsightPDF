"""
Capability / document-profile system.

This is the mechanism that drives "document type controls which tools are
exposed" WITHOUT hard-coded `if doc_type == research_paper` branching in the
UI or API: a document's `profile.capabilities` is just a list of tool ids,
computed once at classification time from `CAPABILITIES_BY_TYPE` below. The
frontend renders whichever tools from the catalog (GET /tools/catalog)
appear in that list. Adding a new document type or tool means editing this
map and the tool registry -- nothing else.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from app.core.enums import DocumentType

# Tools every document type gets, regardless of classification (section 5).
UNIVERSAL_CAPABILITIES: list[str] = [
    "summarize",
    "chat",
    "ask_question",
    "explain",
    "extract_key_info",
    "search_documents",
    "generate_quiz",
    "find_important_sections",
    "compare_documents",
]

RESEARCH_PAPER_CAPABILITIES: list[str] = [
    "analyze_dataset",
    "analyze_model",
    "analyze_architecture",
    "analyze_methodology",
    "extract_experimental_setup",
    "analyze_metrics",
    "extract_results",
    "analyze_limitations",
    "identify_research_gaps",
    "compare_papers",
    "find_common_techniques",
    "find_differences",
    "generate_literature_review",
    "generate_research_brief",
    "web_research",
]

LITERATURE_CAPABILITIES: list[str] = [
    "spoiler_free_summary",
    "full_summary",
    "chapter_summary",
    "character_analysis",
    "character_relationships",
    "theme_analysis",
    "plot_analysis",
    "important_events",
    "motifs_symbols",
    "review",
]

STUDY_MATERIAL_CAPABILITIES: list[str] = [
    "chapter_summary",
    "simplified_explanation",
    "notes_generation",
    "important_concepts",
    "definitions_extraction",
    "formula_extraction",
    "flashcards",
    "practice_questions",
    "important_questions",
    "question_paper_generator",
]

TECHNICAL_DOCUMENTATION_CAPABILITIES: list[str] = [
    "architecture_overview",
    "component_extraction",
    "api_extraction",
    "requirements_extraction",
    "dependency_analysis",
    "workflow_explanation",
    "configuration_extraction",
    "security_requirement_extraction",
    "implementation_checklist",
    "technical_summary",
]

BUSINESS_REPORT_CAPABILITIES: list[str] = [
    "executive_summary",
    "kpi_extraction",
    "metric_extraction",
    "trend_analysis",
    "risk_extraction",
    "action_items",
    "yoy_comparison",
    "multi_report_comparison",
    "generate_business_brief",
]

CAPABILITIES_BY_TYPE: dict[DocumentType, list[str]] = {
    DocumentType.RESEARCH_PAPER: RESEARCH_PAPER_CAPABILITIES,
    DocumentType.LITERATURE: LITERATURE_CAPABILITIES,
    DocumentType.STUDY_MATERIAL: STUDY_MATERIAL_CAPABILITIES,
    DocumentType.TECHNICAL_DOCUMENTATION: TECHNICAL_DOCUMENTATION_CAPABILITIES,
    DocumentType.BUSINESS_REPORT: BUSINESS_REPORT_CAPABILITIES,
    DocumentType.GENERIC: [],
}


def capabilities_for(doc_type: DocumentType) -> list[str]:
    specific = CAPABILITIES_BY_TYPE.get(doc_type, [])
    # Preserve order, de-duplicate.
    seen: set[str] = set()
    ordered = []
    for cap in [*UNIVERSAL_CAPABILITIES, *specific]:
        if cap not in seen:
            seen.add(cap)
            ordered.append(cap)
    return ordered


# --- Structured profile produced by the classifier -------------------------


class DocumentSection(BaseModel):
    title: str
    page_start: int
    page_end: Optional[int] = None
    level: int = 1


class DocumentEntity(BaseModel):
    name: str
    type: str = Field(description="e.g. person, organization, dataset, model, metric, api, concept")
    mentions: int = 1


class TableInfo(BaseModel):
    page: int
    caption: Optional[str] = None
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None


class DocumentProfile(BaseModel):
    """The structured output of the classification stage. Stored verbatim
    (as JSON) on `Document.profile`."""

    type: DocumentType
    confidence: float = Field(ge=0.0, le=1.0)
    title: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    summary_hint: Optional[str] = Field(
        default=None, description="One or two sentence description of what this document is"
    )
    sections: list[DocumentSection] = Field(default_factory=list)
    entities: list[DocumentEntity] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    tables: list[TableInfo] = Field(default_factory=list)
    key_metadata: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)

    def with_capabilities(self) -> "DocumentProfile":
        self.capabilities = capabilities_for(self.type)
        return self
