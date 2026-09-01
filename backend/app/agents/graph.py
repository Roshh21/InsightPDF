"""
The core agent graph (spec sections 11 & 13):

    understand_intent
          |
     (tool)  (qa)
       |       |
   run_tool  rewrite_query
       |       |
      END    retrieve <----+
               |            |
          [insufficient]    |
               |            |
          refine_query -----+
               |
          [sufficient]
               |
             analyze
               |
            validate
               |
              END

Deterministic where it can be (retrieval quality is judged by an actual
similarity-score heuristic, not an extra LLM call) and agentic where it has
to be (intent routing, query rewriting, grounded analysis all reason over
real evidence rather than being a fixed pipeline).
"""
from __future__ import annotations

import time

from langgraph.graph import END, StateGraph

from app.agents.schemas import IntentDecision, RagAnswer
from app.agents.state import AgentState
from app.core.config import get_settings
from app.core.enums import TaskType
from app.core.exceptions import InsightPDFError
from app.core.logging import get_logger
from app.llm.base import LLMMessage
from app.llm.gateway import ModelGateway
from app.llm.structured import generate_structured
from app.retrieval.evidence_validator import validate_citations
from app.retrieval.query_rewriter import refine_query, rewrite_query
from app.retrieval.retriever import Retriever
from app.tools.base import ToolContext
from app.tools.registry import ensure_loaded, get_tool

logger = get_logger(__name__)

_INTENT_SYSTEM_TEMPLATE = """You route user messages in a document-chat interface to either a direct \
Q&A answer or a specific known tool.

Available tools for the document(s) currently in scope (id: description):
{tool_list}

Return intent="tool" with the matching tool_id ONLY if the user's message clearly and specifically \
asks for one of these exact capabilities (e.g. "summarize this" -> summarize; "compare these papers" \
-> compare_papers/compare_documents; "quiz me" -> generate_quiz). Otherwise, including for any question \
that needs a direct answer, ordinary conversation, or a capability not in the list, return intent="qa" \
with tool_id=null."""


def build_agent_graph(gateway: ModelGateway, retriever: Retriever):
    ensure_loaded()
    settings = get_settings()

    def node_understand_intent(state: AgentState) -> dict:
        if state.get("forced_tool_id"):
            return {"intent": "tool", "tool_id": state["forced_tool_id"]}

        available = set()
        for doc in state["documents"]:
            available.update(doc.capabilities or [])
        if not available:
            return {"intent": "qa", "tool_id": None}

        tool_lines = []
        for tool_id in sorted(available):
            tool = get_tool(tool_id)
            if tool:
                tool_lines.append(f"- {tool.id}: {tool.description}")
        system = _INTENT_SYSTEM_TEMPLATE.format(tool_list="\n".join(tool_lines))

        try:
            decision = generate_structured(
                gateway,
                IntentDecision,
                task_type=TaskType.CLASSIFICATION,
                system=system,
                user_prompt=f"Recent conversation:\n{state.get('conversation_context','')}\n\nUser message: {state['question']}",
                max_tokens=300,
            )
        except Exception as exc:
            logger.warning("Intent routing failed, defaulting to qa: %s", exc)
            return {"intent": "qa", "tool_id": None}

        if decision.intent == "tool" and decision.tool_id in available:
            return {"intent": "tool", "tool_id": decision.tool_id}
        return {"intent": "qa", "tool_id": None}

    def route_after_intent(state: AgentState) -> str:
        return "run_tool" if state.get("intent") == "tool" and state.get("tool_id") else "rewrite_query"

    def node_run_tool(state: AgentState) -> dict:
        from app.services.tool_service import execute_tool

        tool = get_tool(state["tool_id"])
        if tool is None:
            return {"intent": "qa", "tool_error": "tool not found"}

        db = state.get("db")
        try:
            result, _execution = execute_tool(
                db,
                workspace_id=state["workspace_id"],
                document_ids=state["document_ids"],
                tool_id=state["tool_id"],
                params=state.get("tool_params") or {},
            )
            return {
                "answer_text": result.title or tool.name,
                "tool_result": result.model_dump(mode="json"),
                "result_kind": result.result_kind.value,
                "citations": result.citations,
                "warnings": result.warnings,
            }
        except Exception as exc:
            logger.warning("Tool '%s' failed during chat routing, falling back to Q&A: %s", state["tool_id"], exc)
            return {"intent": "qa", "tool_id": None, "tool_error": str(exc)}

    def route_after_tool(state: AgentState) -> str:
        return "rewrite_query" if state.get("intent") == "qa" else "end"

    def node_rewrite_query(state: AgentState) -> dict:
        rewritten = rewrite_query(gateway, state["question"], state.get("conversation_context", ""))
        return {"rewritten_query": rewritten, "attempts": 0}

    def node_retrieve(state: AgentState) -> dict:
        start = time.monotonic()
        result = retriever.retrieve(
            state["rewritten_query"],
            workspace_id=state["workspace_id"],
            document_ids=state["document_ids"] or None,
            top_k=settings.RETRIEVAL_TOP_K,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        return {
            "evidence": result.chunks,
            "retrieval_avg_score": result.avg_score,
            "retrieval_latency_ms": state.get("retrieval_latency_ms", 0) + latency_ms,
        }

    def route_after_retrieve(state: AgentState) -> str:
        chunks = state.get("evidence") or []
        avg_score = state.get("retrieval_avg_score") or 0.0
        attempts = state.get("attempts", 0)
        sufficient = len(chunks) >= 1 and avg_score >= settings.RETRIEVAL_MIN_RELEVANCE
        if sufficient or attempts >= settings.RETRIEVAL_MAX_REFINE_ATTEMPTS:
            return "analyze"
        return "refine"

    def node_refine_query(state: AgentState) -> dict:
        refined = refine_query(gateway, state["question"], state["rewritten_query"])
        return {"rewritten_query": refined, "attempts": state.get("attempts", 0) + 1}

    def node_analyze(state: AgentState) -> dict:
        chunks = state.get("evidence") or []
        if not chunks:
            return {
                "answer_text": (
                    "I couldn't find relevant evidence for that in the selected document(s) after "
                    "multiple search attempts. Try rephrasing, or check that the right document(s) are "
                    "selected and finished processing."
                ),
                "citations": [],
                "warnings": ["insufficient_evidence"],
            }

        evidence_block = "\n\n---\n\n".join(
            f"[document_id={c.document_id} | {c.document_name} | page={c.page}"
            f"{' | section=' + c.section if c.section else ''}]\n{c.text}"
            for c in chunks
        )
        system = (
            "You are the conversational assistant for a document intelligence platform. Answer the "
            "user's question grounded ONLY in the evidence below. Use the conversation history for "
            "context (e.g. resolving 'it', 'the second one'), but ground factual claims in the evidence. "
            "Every citation must use the exact document_id and page shown in the evidence tags. If the "
            "evidence only partially answers the question, say what is and isn't covered."
        )
        user_prompt = (
            f"Conversation history:\n{state.get('conversation_context','(none)')}\n\n"
            f"Question: {state['question']}\n\nEVIDENCE:\n{evidence_block}"
        )
        try:
            result = generate_structured(
                gateway, RagAnswer, task_type=TaskType.GENERATION_COMPLEX, system=system, user_prompt=user_prompt, max_tokens=1800
            )
            return {"answer_text": result.answer, "citations": result.citations}
        except InsightPDFError:
            # A known, typed failure (e.g. every configured free provider is
            # currently rate-limited/unavailable, or the model could not
            # produce valid structured output after recovery). Let it
            # propagate out of the graph rather than masking it as a normal
            # chat reply -- the API layer turns this into a clean, correctly
            # -coded structured error instead of a fake HTTP 200 "I ran into
            # an error" message.
            raise
        except Exception as exc:  # pragma: no cover - genuinely unexpected bug
            logger.exception("Unexpected error during agentic RAG analysis")
            raise InsightPDFError(
                f"Unexpected error during analysis: {exc}",
                user_message="Something went wrong generating an answer. Please try again.",
            ) from exc

    def node_validate(state: AgentState) -> dict:
        citations = state.get("citations") or []
        chunks = state.get("evidence") or []
        validated = validate_citations(citations, chunks)
        return {"citations": validated}

    graph = StateGraph(AgentState)
    graph.add_node("understand_intent", node_understand_intent)
    graph.add_node("run_tool", node_run_tool)
    graph.add_node("rewrite_query", node_rewrite_query)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("refine_query", node_refine_query)
    graph.add_node("analyze", node_analyze)
    graph.add_node("validate", node_validate)

    graph.set_entry_point("understand_intent")
    graph.add_conditional_edges(
        "understand_intent", route_after_intent, {"run_tool": "run_tool", "rewrite_query": "rewrite_query"}
    )
    graph.add_conditional_edges("run_tool", route_after_tool, {"rewrite_query": "rewrite_query", "end": END})
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_conditional_edges("retrieve", route_after_retrieve, {"analyze": "analyze", "refine": "refine_query"})
    graph.add_edge("refine_query", "retrieve")
    graph.add_edge("analyze", "validate")
    graph.add_edge("validate", END)

    return graph.compile()
