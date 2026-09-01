"""
Web research tool.

Deliberately kept separate from local document retrieval (a distinct
client, a distinct evidence pool, and a distinct `web_sources` field in the
output rather than mixing external sources into `citations`) so source
provenance -- "this came from the uploaded PDF" vs "this came from the
web" -- always stays unambiguous to the user.
"""
from __future__ import annotations

from app.core.config import get_settings
from app.core.enums import DocumentType, ResultKind, TaskType
from app.core.exceptions import ToolExecutionError, WebSearchError
from app.core.logging import get_logger
from app.retrieval.evidence_validator import validate_citations
from app.schemas.common import ToolResultEnvelope
from app.tools.base import BaseTool, ToolContext
from app.tools.registry import register_tool
from app.tools.schemas import ResearchQueries, WebResearchOutput, WebSource

logger = get_logger(__name__)

MAX_SOURCES = 6


def tavily_search(query: str, api_key: str, max_results: int = 4) -> list[WebSource]:
    import httpx

    try:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
            },
            timeout=20.0,
        )
        resp.raise_for_status()
    except Exception as exc:
        raise WebSearchError(f"Web search request failed: {exc}") from exc

    data = resp.json()
    return [
        WebSource(title=r.get("title", r.get("url", "")), url=r.get("url", ""), snippet=(r.get("content") or "")[:500])
        for r in data.get("results", [])
    ]


class SearchWebTool(BaseTool):
    id = "search_web"
    name = "Search Web"
    description = "Search the public web for external context (requires TAVILY_API_KEY)."
    category = "internal"
    result_kind = ResultKind.LIST
    icon = "globe"

    def run(self, ctx: ToolContext) -> ToolResultEnvelope:
        settings = get_settings()
        if not settings.TAVILY_API_KEY:
            raise WebSearchError(
                user_message="Web search is not configured. Set TAVILY_API_KEY in the backend .env to enable it."
            )
        query = ctx.params.get("query")
        if not query:
            raise ToolExecutionError(user_message="A search query is required.")
        results = tavily_search(query, settings.TAVILY_API_KEY, max_results=int(ctx.params.get("max_results", 5)))
        return ToolResultEnvelope(
            tool_name=self.id,
            result_kind=self.result_kind,
            title=f'Web results for "{query}"',
            content={"items": [f"{r.title} — {r.url}" for r in results], "sources": [r.model_dump() for r in results]},
            citations=[],
        )


_QUERY_GEN_SYSTEM = """Given a research paper's topic (from the uploaded evidence) and the user's \
question about how the field has moved since, propose 2-4 focused web search queries that would \
surface recent, relevant developments. Return ONLY the queries."""

_SYNTHESIS_SYSTEM = """You are a research analyst producing a grounded report that compares an \
uploaded paper against current external context gathered from the web.

You will be given two separate evidence pools, clearly tagged:
- PAPER EVIDENCE: excerpts from the uploaded document(s), each tagged [document_id=<id> | <name> | page=<n>]
- WEB EVIDENCE: excerpts from web search results, each tagged [WEB | <title> | <url>]

Rules:
- Citations (the `citations` field) must ONLY reference PAPER EVIDENCE, using the exact document_id/page \
shown in its tags.
- Web sources you actually used must be listed in `web_sources` (title + url), NOT in `citations`.
- Be explicit about what has changed / what is new relative to the paper, and be honest if the web \
evidence doesn't clearly show change.
- Do not present web content as if it were from the paper, or vice versa."""


class WebResearchTool(BaseTool):
    id = "web_research"
    name = "Web Research"
    description = (
        "Identify what's changed in this field since the paper was published by researching the web "
        "and comparing findings against the paper, with clearly separated citations."
    )
    category = "research_paper"
    applicable_types = [DocumentType.RESEARCH_PAPER]
    result_kind = ResultKind.REPORT
    icon = "globe"

    def run(self, ctx: ToolContext) -> ToolResultEnvelope:
        settings = get_settings()
        if not settings.TAVILY_API_KEY:
            raise WebSearchError(
                user_message="Web research requires a Tavily API key. Set TAVILY_API_KEY in the backend .env."
            )
        question = ctx.params.get("query") or "What has changed in this field since this paper was published?"

        # 1. Local evidence: what is this paper actually about?
        local = ctx.retriever.retrieve(
            question, workspace_id=ctx.workspace_id, document_ids=ctx.document_ids or None, top_k=10
        )
        if not local.chunks:
            return ToolResultEnvelope(
                tool_name=self.id,
                result_kind=self.result_kind,
                title="Web Research",
                content={"summary": "No indexed content found for the selected document(s).", "sections": [], "web_sources": []},
                warnings=["Nothing to research against -- select a processed document first."],
            )
        paper_block = "\n\n---\n\n".join(
            f"[document_id={c.document_id} | {c.document_name} | page={c.page}]\n{c.text}" for c in local.chunks
        )

        # 2. Generate targeted web search queries.
        from app.llm.structured import generate_structured

        try:
            query_plan = generate_structured(
                ctx.gateway,
                ResearchQueries,
                task_type=TaskType.QUERY_REWRITE,
                system=_QUERY_GEN_SYSTEM,
                user_prompt=f"User question: {question}\n\nPaper evidence:\n{paper_block[:4000]}",
                max_tokens=300,
            )
        except Exception as exc:
            logger.warning("Web research query generation failed, using the raw question: %s", exc)
            query_plan = ResearchQueries(queries=[question])

        # 3. Search the web, collecting up to MAX_SOURCES sources.
        sources: list[WebSource] = []
        for q in query_plan.queries:
            if len(sources) >= MAX_SOURCES:
                break
            try:
                sources.extend(tavily_search(q, settings.TAVILY_API_KEY, max_results=3))
            except WebSearchError as exc:
                logger.warning("Web search failed for query '%s': %s", q, exc)
        # de-dupe by URL, cap total
        seen_urls: set[str] = set()
        deduped: list[WebSource] = []
        for s in sources:
            if s.url and s.url not in seen_urls:
                seen_urls.add(s.url)
                deduped.append(s)
        deduped = deduped[:MAX_SOURCES]

        if not deduped:
            web_block = "(no web results found)"
        else:
            web_block = "\n\n---\n\n".join(f"[WEB | {s.title} | {s.url}]\n{s.snippet}" for s in deduped)

        # 4. Synthesize the cited comparison report.
        user_prompt = (
            f"User question: {question}\n\n"
            f"PAPER EVIDENCE:\n{paper_block}\n\n"
            f"WEB EVIDENCE:\n{web_block}\n\n"
            "Produce your grounded comparison report now."
        )
        try:
            output = generate_structured(
                ctx.gateway,
                WebResearchOutput,
                task_type=TaskType.REPORT_GENERATION,
                system=_SYNTHESIS_SYSTEM,
                user_prompt=user_prompt,
                max_tokens=3500,
            )
        except Exception as exc:
            raise ToolExecutionError(f"Web research synthesis failed: {exc}") from exc

        validated = validate_citations(output.citations, local.chunks)
        content = output.model_dump(mode="json")
        content.pop("citations", None)

        return ToolResultEnvelope(
            tool_name=self.id,
            result_kind=self.result_kind,
            title="Web Research",
            content=content,
            citations=validated,
        )


register_tool(SearchWebTool())
register_tool(WebResearchTool())
