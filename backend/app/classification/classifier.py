from __future__ import annotations

from pydantic import BaseModel, Field

from app.classification.profiles import DocumentEntity, DocumentProfile
from app.core.enums import DocumentType, TaskType
from app.core.logging import get_logger
from app.ingestion.pdf_parser import ParsedDocument
from app.llm.gateway import ModelGateway
from app.llm.structured import generate_structured

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a document classification and understanding system for a document \
intelligence platform. Given a sample of extracted text from a PDF, determine:

1. The document type -- exactly one of: research_paper, literature, study_material, \
technical_documentation, business_report, generic.
   - research_paper: academic papers, preprints, conference/journal submissions.
   - literature: novels, short stories, plays, other narrative fiction/creative writing.
   - study_material: textbooks, lecture notes, revision notes, exam prep material.
   - technical_documentation: software/API/system docs, RFCs, architecture docs, manuals.
   - business_report: financial/business/market reports, earnings reports, consulting decks turned to text.
   - generic: anything that doesn't clearly fit the above.
2. Your confidence (0-1) in that classification.
3. Title and authors if identifiable.
4. A one-to-two sentence description of what the document is.
5. Up to 12 key topics/keywords.
6. Up to 15 important named entities (people, organizations, datasets, models, APIs, \
metrics, products -- whatever is most relevant to this document type) with an entity type label.

Be conservative with confidence: only go above 0.85 if the signal is unambiguous \
(e.g. an explicit "Abstract" section strongly implies research_paper).
Base everything ONLY on the provided text. Do not invent authors, titles, or entities."""


class _ClassifierOutput(BaseModel):
    type: DocumentType
    confidence: float = Field(ge=0.0, le=1.0)
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    summary_hint: str | None = None
    topics: list[str] = Field(default_factory=list)
    entities: list[DocumentEntity] = Field(default_factory=list)


def classify_document(gateway: ModelGateway, parsed: ParsedDocument, filename: str) -> DocumentProfile:
    # Sample from the start, a bit from the middle, and the end -- more
    # representative than just the first N characters for long documents.
    text = parsed.full_text
    if len(text) > 9000:
        sample = text[:5000] + "\n...\n" + text[len(text) // 2 : len(text) // 2 + 2000] + "\n...\n" + text[-2000:]
    else:
        sample = text

    user_prompt = f"Filename: {filename}\nPage count: {parsed.page_count}\n\nDocument text sample:\n{sample}"

    try:
        result = generate_structured(
            gateway,
            _ClassifierOutput,
            task_type=TaskType.CLASSIFICATION,
            system=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=1500,
        )
    except Exception as exc:
        logger.error("Classification failed, defaulting to generic: %s", exc)
        result = _ClassifierOutput(type=DocumentType.GENERIC, confidence=0.0, summary_hint="Classification unavailable.")

    profile = DocumentProfile(
        type=result.type,
        confidence=result.confidence,
        title=result.title,
        authors=result.authors,
        summary_hint=result.summary_hint,
        topics=result.topics,
        entities=result.entities,
    )
    return profile.with_capabilities()
