"""
Tools that need direct data access rather than LLM synthesis (spec section
12's lower-level "Agent Tools"). These are registered in the same registry
as the catalog PromptTools so the agent can call any tool -- catalog or
primitive -- through one uniform interface, but most are marked
category="internal" so they don't appear as document-type capability
buttons in the UI (which only renders tools present in a document's
`capabilities` list).
"""
from __future__ import annotations

import ast
import operator

from app.core.enums import DocumentType, ResultKind, TaskType
from app.core.exceptions import ToolExecutionError
from app.extraction.table_extractor import extract_table_from_page
from app.ingestion.pdf_parser import get_page_text
from app.models.document import Document
from app.retrieval.evidence_validator import validate_citations
from app.schemas.common import Citation, ToolResultEnvelope
from app.tools.base import BaseTool, ToolContext
from app.tools.prompt_tool import PromptTool, PromptToolSpec
from app.tools.registry import register_tool


def _search(ctx: ToolContext, document_ids: list[str] | None) -> ToolResultEnvelope:
    query = ctx.params.get("query") or ""
    if not query:
        raise ToolExecutionError(user_message="A search query is required.")
    top_k = int(ctx.params.get("top_k", 10))
    result = ctx.retriever.retrieve(query, workspace_id=ctx.workspace_id, document_ids=document_ids, top_k=top_k)
    hits = [
        {
            "document_id": c.document_id,
            "document_name": c.document_name,
            "page": c.page,
            "section": c.section,
            "snippet": c.text[:400],
            "score": round(c.score, 4),
        }
        for c in result.chunks
    ]
    citations = [
        Citation(document_id=c.document_id, document_name=c.document_name, page=c.page, section=c.section, excerpt=c.text[:250])
        for c in result.chunks
    ]
    return ToolResultEnvelope(
        tool_name="search_documents",
        result_kind=ResultKind.LIST,
        title=f'Search results for "{query}"',
        content={"query": query, "results": hits},
        citations=citations,
    )


class SearchDocumentsTool(BaseTool):
    id = "search_documents"
    name = "Search Documents"
    description = "Semantic search across all selected documents in the workspace."
    category = "universal"
    result_kind = ResultKind.LIST
    icon = "search"

    def run(self, ctx: ToolContext) -> ToolResultEnvelope:
        return _search(ctx, ctx.document_ids or None)


class SearchSingleDocumentTool(BaseTool):
    """Internal primitive: search restricted to one document_id, named
    explicitly in params (used by the agent when reasoning about a single
    document rather than the whole workspace selection)."""

    id = "search_document"
    name = "Search Document"
    description = "Semantic search within a single document."
    category = "internal"
    result_kind = ResultKind.LIST
    icon = "search"

    def run(self, ctx: ToolContext) -> ToolResultEnvelope:
        document_id = ctx.params.get("document_id")
        if not document_id:
            raise ToolExecutionError(user_message="document_id is required for search_document.")
        return _search(ctx, [document_id])


class GetPageTool(BaseTool):
    id = "get_page"
    name = "Get Page"
    description = "Fetch the raw extracted text of a specific page of a specific document."
    category = "internal"
    result_kind = ResultKind.TEXT
    icon = "file-text"

    def run(self, ctx: ToolContext) -> ToolResultEnvelope:
        document_id = ctx.params.get("document_id")
        page = ctx.params.get("page")
        if not document_id or page is None:
            raise ToolExecutionError(user_message="document_id and page are required for get_page.")
        doc = next((d for d in ctx.documents if d.id == document_id), None)
        if doc is None:
            raise ToolExecutionError(user_message="Document not found in this workspace selection.")
        text = get_page_text(doc.file_path, int(page))
        return ToolResultEnvelope(
            tool_name=self.id,
            result_kind=self.result_kind,
            title=f"{doc.original_filename} — Page {page}",
            content={"answer": text},
            citations=[
                Citation(document_id=doc.id, document_name=doc.original_filename, page=int(page), excerpt=text[:250])
            ],
        )


class GetSectionTool(BaseTool):
    id = "get_section"
    name = "Get Section"
    description = "Fetch the text of a named section of a document, using its detected structure."
    category = "internal"
    result_kind = ResultKind.TEXT
    icon = "file-text"

    MAX_PAGES = 8

    def run(self, ctx: ToolContext) -> ToolResultEnvelope:
        document_id = ctx.params.get("document_id")
        section_title = (ctx.params.get("section") or "").strip().lower()
        if not document_id or not section_title:
            raise ToolExecutionError(user_message="document_id and section are required for get_section.")
        doc = next((d for d in ctx.documents if d.id == document_id), None)
        if doc is None or not doc.profile:
            raise ToolExecutionError(user_message="Document not found or has no detected structure yet.")
        sections = doc.profile.get("sections", [])
        match = next((s for s in sections if s["title"].strip().lower() == section_title), None)
        if match is None:
            match = next((s for s in sections if section_title in s["title"].strip().lower()), None)
        if match is None:
            raise ToolExecutionError(user_message=f"No section matching '{section_title}' was detected.")

        start = match["page_start"]
        end = min(match.get("page_end") or start, start + self.MAX_PAGES - 1)
        pages_text = []
        for p in range(start, end + 1):
            pages_text.append(get_page_text(doc.file_path, p))
        text = "\n\n".join(t for t in pages_text if t)
        return ToolResultEnvelope(
            tool_name=self.id,
            result_kind=self.result_kind,
            title=f"{doc.original_filename} — {match['title']}",
            content={"answer": text},
            citations=[
                Citation(document_id=doc.id, document_name=doc.original_filename, page=start, section=match["title"], excerpt=text[:250])
            ],
        )


class ExtractMetadataTool(BaseTool):
    id = "extract_metadata"
    name = "Extract Metadata"
    description = "Return the document's structured metadata (title, authors, topics, key facts) as "
    "already computed during ingestion -- no additional model call."
    category = "internal"
    result_kind = ResultKind.SECTIONS
    icon = "info"

    def run(self, ctx: ToolContext) -> ToolResultEnvelope:
        sections = []
        for doc in ctx.documents:
            profile = doc.profile or {}
            lines = [f"Type: {doc.document_type}", f"Confidence: {doc.classification_confidence}"]
            if profile.get("title"):
                lines.append(f"Title: {profile['title']}")
            if profile.get("authors"):
                lines.append(f"Authors: {', '.join(profile['authors'])}")
            if profile.get("topics"):
                lines.append(f"Topics: {', '.join(profile['topics'])}")
            for k, v in (profile.get("key_metadata") or {}).items():
                lines.append(f"{k}: {v}")
            sections.append({"heading": doc.original_filename, "content": "\n".join(lines)})
        return ToolResultEnvelope(
            tool_name=self.id,
            result_kind=self.result_kind,
            title="Document Metadata",
            content={"summary": f"Metadata for {len(ctx.documents)} document(s).", "sections": sections},
            citations=[],
        )


class ExtractEntitiesTool(BaseTool):
    id = "extract_entities"
    name = "Extract Entities"
    description = "Return the named entities already identified for the selected documents during "
    "classification."
    category = "internal"
    result_kind = ResultKind.ENTITIES
    icon = "tag"

    def run(self, ctx: ToolContext) -> ToolResultEnvelope:
        entities = []
        for doc in ctx.documents:
            for e in (doc.profile or {}).get("entities", []):
                entities.append({**e, "description": f"From {doc.original_filename}"})
        return ToolResultEnvelope(
            tool_name=self.id,
            result_kind=self.result_kind,
            title="Extracted Entities",
            content={"entities": entities},
            citations=[],
        )


class ExtractTableTool(BaseTool):
    id = "extract_table"
    name = "Extract Table"
    description = "Extract the raw cell grid of a specific table on a specific page."
    category = "internal"
    result_kind = ResultKind.TABLE
    icon = "table"

    def run(self, ctx: ToolContext) -> ToolResultEnvelope:
        document_id = ctx.params.get("document_id")
        page = ctx.params.get("page")
        table_index = int(ctx.params.get("table_index", 0))
        if not document_id or page is None:
            raise ToolExecutionError(user_message="document_id and page are required for extract_table.")
        doc = next((d for d in ctx.documents if d.id == document_id), None)
        if doc is None:
            raise ToolExecutionError(user_message="Document not found in this workspace selection.")
        try:
            rows = extract_table_from_page(doc.file_path, int(page), table_index)
        except Exception as exc:
            raise ToolExecutionError(f"Table extraction failed: {exc}") from exc
        return ToolResultEnvelope(
            tool_name=self.id,
            result_kind=self.result_kind,
            title=f"{doc.original_filename} — Table (page {page})",
            content={"rows": rows},
            citations=[Citation(document_id=doc.id, document_name=doc.original_filename, page=int(page), excerpt="[table]")],
        )


# --- calculate --------------------------------------------------------------

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Expression contains disallowed syntax")


class CalculateTool(BaseTool):
    id = "calculate"
    name = "Calculate"
    description = "Evaluate a numeric arithmetic expression (e.g. derived from figures found in a "
    "document)."
    category = "internal"
    result_kind = ResultKind.TEXT
    icon = "calculator"

    def run(self, ctx: ToolContext) -> ToolResultEnvelope:
        expr = ctx.params.get("expression")
        if not expr:
            raise ToolExecutionError(user_message="An 'expression' parameter is required.")
        try:
            tree = ast.parse(expr, mode="eval")
            value = _safe_eval(tree.body)
        except Exception as exc:
            raise ToolExecutionError(user_message=f"Could not evaluate expression: {exc}") from exc
        return ToolResultEnvelope(
            tool_name=self.id,
            result_kind=self.result_kind,
            title="Calculation",
            content={"answer": f"{expr} = {value}"},
            citations=[],
        )


class ValidateEvidenceTool(BaseTool):
    """Wraps the evidence validator as a callable tool for agent
    introspection/testing. The main agentic-RAG graph calls
    `validate_citations` directly for efficiency; this exists so the
    capability is also reachable as a standalone tool per spec section 12."""

    id = "validate_evidence"
    name = "Validate Evidence"
    description = "Check whether a set of proposed citations are actually supported by retrieved "
    "evidence."
    category = "internal"
    result_kind = ResultKind.LIST
    icon = "check-circle"

    def run(self, ctx: ToolContext) -> ToolResultEnvelope:
        raw_citations = ctx.params.get("citations") or []
        citations = [Citation.model_validate(c) for c in raw_citations]
        query = ctx.params.get("query") or " ".join(c.excerpt for c in citations)[:200]
        result = ctx.retriever.retrieve(
            query or "evidence", workspace_id=ctx.workspace_id, document_ids=ctx.document_ids or None, top_k=20
        )
        validated = validate_citations(citations, result.chunks)
        items = [f"{'✓' if c.supported else '✗'} {c.document_name} p.{c.page}: {c.excerpt[:80]}" for c in validated]
        return ToolResultEnvelope(
            tool_name=self.id,
            result_kind=self.result_kind,
            title="Evidence Validation",
            content={"items": items},
            citations=validated,
        )


for tool in [
    SearchDocumentsTool(),
    SearchSingleDocumentTool(),
    GetPageTool(),
    GetSectionTool(),
    ExtractMetadataTool(),
    ExtractEntitiesTool(),
    ExtractTableTool(),
    CalculateTool(),
    ValidateEvidenceTool(),
]:
    register_tool(tool)

# --- generic (type-agnostic) generation primitives, still LLM-backed but
# not tied to any single document type -- reachable by the agent for
# generic/mixed workspaces and marked internal (not a per-type UI button).

register_tool(
    PromptTool(
        PromptToolSpec(
            id="generate_questions",
            name="Generate Questions",
            description="Generate a set of questions (with answers) about the selected document(s).",
            category="internal",
            applicable_types=None,
            system_prompt="Generate a well-rounded set of questions with correct answers and brief "
            "explanations, covering a good spread of the material.",
            result_kind=ResultKind.QUIZ,
            task_type=TaskType.GENERATION_SIMPLE,
            top_k=14,
            icon="help-circle",
        )
    )
)

register_tool(
    PromptTool(
        PromptToolSpec(
            id="generate_report",
            name="Generate Report",
            description="Generate a general-purpose grounded report on a topic/question spanning the "
            "selected document(s).",
            category="internal",
            applicable_types=None,
            system_prompt="Write a structured report addressing the user's configured topic/question, "
            "organized into clearly labeled sections, grounded strictly in the evidence.",
            result_kind=ResultKind.REPORT,
            task_type=TaskType.REPORT_GENERATION,
            top_k=16,
            max_tokens=3500,
            icon="file-text",
        )
    )
)

register_tool(
    PromptTool(
        PromptToolSpec(
            id="extract_structured_data",
            name="Extract Structured Data",
            description="Extract a user-specified set of fields from the document(s) as structured "
            "sections.",
            category="internal",
            applicable_types=None,
            system_prompt="Extract exactly the fields/data points requested in the user configuration "
            "(see the `fields` entry) from the evidence, one section per field. If a field is not "
            "present in the evidence, say 'Not found' for it rather than guessing.",
            result_kind=ResultKind.SECTIONS,
            task_type=TaskType.METADATA_EXTRACTION,
            top_k=12,
            icon="scan-search",
        )
    )
)
