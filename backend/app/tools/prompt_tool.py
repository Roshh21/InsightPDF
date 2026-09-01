from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel

from app.core.enums import DocumentType, ResultKind, TaskType
from app.core.exceptions import InsightPDFError, ToolExecutionError
from app.llm.structured import generate_structured
from app.retrieval.evidence_validator import validate_citations
from app.schemas.common import Citation, ToolResultEnvelope
from app.tools.base import BaseTool, ToolContext
from app.tools.schemas import (
    ComparisonOutput,
    EntitiesOutput,
    FlashcardsOutput,
    ListOutput,
    QuestionPaperOutput,
    QuizOutput,
    SectionsOutput,
    TextOutput,
)
from app.vectorstores.base import RetrievedChunk

_SCHEMA_BY_KIND: dict[ResultKind, type[BaseModel]] = {
    ResultKind.SECTIONS: SectionsOutput,
    ResultKind.COMPARISON: ComparisonOutput,
    ResultKind.QUIZ: QuizOutput,
    ResultKind.QUESTION_PAPER: QuestionPaperOutput,
    ResultKind.FLASHCARDS: FlashcardsOutput,
    ResultKind.REPORT: SectionsOutput,
    ResultKind.TEXT: TextOutput,
    ResultKind.LIST: ListOutput,
    ResultKind.ENTITIES: EntitiesOutput,
}

_GROUNDING_INSTRUCTIONS = """
You are part of a document intelligence platform. You are given EVIDENCE excerpts pulled \
from the user's uploaded document(s) via retrieval. Each excerpt is tagged with its source, \
like: [document_id=<id> | <document name> | page=<n> | section=<name>].

Rules:
- Base your answer ONLY on the provided evidence. Do not invent facts, numbers, or citations \
that are not supported by the evidence.
- If the evidence is insufficient to fully address something, say so explicitly rather than \
guessing.
- Every citation you emit MUST use the exact `document_id` value shown in the evidence tags \
(not the document name) and a `page` number taken from that tag.
- Keep the `excerpt` field of each citation short (under ~250 characters) and close to the \
actual wording of the source so it can be verified.
"""


def _render_params(params: dict) -> str:
    if not params:
        return ""
    interesting = {k: v for k, v in params.items() if k != "query" and v not in (None, "", [])}
    if not interesting:
        return ""
    lines = "\n".join(f"- {k}: {v}" for k, v in interesting.items())
    return f"\nUser-specified configuration (respect these where applicable):\n{lines}\n"


def _format_evidence(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for c in chunks:
        section_part = f" | section={c.section}" if c.section else ""
        blocks.append(f"[document_id={c.document_id} | {c.document_name} | page={c.page}{section_part}]\n{c.text}")
    return "\n\n---\n\n".join(blocks)


@dataclass
class PromptToolSpec:
    id: str
    name: str
    description: str
    category: str
    system_prompt: str
    applicable_types: Optional[list[DocumentType]] = None
    result_kind: ResultKind = ResultKind.SECTIONS
    requires_multi_document: bool = False
    top_k: int = 10
    icon: str = "sparkles"
    task_type: TaskType = TaskType.GENERATION_SIMPLE
    default_query: Optional[str] = None
    max_tokens: int = 3000


class PromptTool(BaseTool):
    """Wraps a PromptToolSpec into a runnable BaseTool.

    This single class implements the vast majority of the platform's tools
    (summaries, extractions, analyses, comparisons, quizzes, ...): the
    document-type-specific *behavior* lives entirely in data (the prompt +
    schema + retrieval shape declared in each catalog file), not in
    duplicated Python code per tool.
    """

    def __init__(self, spec: PromptToolSpec):
        self.spec = spec
        self.id = spec.id
        self.name = spec.name
        self.description = spec.description
        self.category = spec.category
        self.applicable_types = spec.applicable_types
        self.result_kind = spec.result_kind
        self.requires_multi_document = spec.requires_multi_document
        self.icon = spec.icon

    def _retrieve(self, ctx: ToolContext) -> list[RetrievedChunk]:
        spec = self.spec
        query = ctx.params.get("query") or spec.default_query or f"{spec.name}: {spec.description}"

        if spec.requires_multi_document and len(ctx.document_ids) > 1:
            per_doc_k = max(3, spec.top_k // len(ctx.document_ids))
            merged: list[RetrievedChunk] = []
            for doc_id in ctx.document_ids:
                result = ctx.retriever.retrieve(
                    query, workspace_id=ctx.workspace_id, document_ids=[doc_id], top_k=per_doc_k
                )
                merged.extend(result.chunks)
            return merged

        result = ctx.retriever.retrieve(
            query, workspace_id=ctx.workspace_id, document_ids=ctx.document_ids or None, top_k=spec.top_k
        )
        return result.chunks

    def run(self, ctx: ToolContext) -> ToolResultEnvelope:
        spec = self.spec
        if spec.requires_multi_document and len(ctx.document_ids) < 2:
            raise ToolExecutionError(
                user_message=f"'{spec.name}' needs at least 2 documents selected to compare."
            )

        chunks = self._retrieve(ctx)
        ctx.last_evidence = chunks
        if not chunks:
            return ToolResultEnvelope(
                tool_name=spec.id,
                result_kind=spec.result_kind,
                title=spec.name,
                content={"summary": "No indexed content was found for the selected document(s).", "sections": []},
                citations=[],
                warnings=["No retrieval evidence available -- has the document finished processing?"],
            )

        evidence_block = _format_evidence(chunks)
        params_block = _render_params(ctx.params)
        doc_names = ", ".join(sorted({c.document_name for c in chunks}))

        user_prompt = (
            f"Task: {spec.description}\n"
            f"Document(s) in scope: {doc_names}\n"
            f"{params_block}\n"
            f"EVIDENCE:\n{evidence_block}\n\n"
            "Produce your response now, grounded strictly in the evidence above."
        )

        schema = _SCHEMA_BY_KIND[spec.result_kind]
        try:
            output = generate_structured(
                ctx.gateway,
                schema,
                task_type=spec.task_type,
                system=spec.system_prompt + _GROUNDING_INSTRUCTIONS,
                user_prompt=user_prompt,
                max_tokens=spec.max_tokens,
            )
        except InsightPDFError:
            # Preserve specific, actionable errors (all providers
            # unavailable, structured-output failure, etc.) instead of
            # flattening them into a generic tool-failure message.
            raise
        except Exception as exc:
            raise ToolExecutionError(f"{spec.id} generation failed: {exc}") from exc

        raw_citations: list[Citation] = getattr(output, "citations", [])
        for c in raw_citations:
            if not c.document_name:
                c.document_name = ctx.document_name(c.document_id)
        validated = validate_citations(raw_citations, chunks)

        content = output.model_dump(mode="json")
        content.pop("citations", None)

        return ToolResultEnvelope(
            tool_name=spec.id,
            result_kind=spec.result_kind,
            title=spec.name,
            content=content,
            citations=validated,
        )
