from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Optional

from app.agents.graph import build_agent_graph
from app.agents.state import AgentState
from app.embeddings.factory import get_embedding_provider
from app.llm.gateway import ModelGateway, get_model_gateway
from app.retrieval.retriever import Retriever
from app.vectorstores.factory import get_vector_store


class DocumentAgent:
    """Thin, stateless-per-call wrapper around the compiled agent graph.
    Holds the (heavier, reusable) gateway/retriever, builds the graph once."""

    def __init__(self, gateway: ModelGateway, retriever: Retriever):
        self.gateway = gateway
        self.retriever = retriever
        self._graph = build_agent_graph(gateway, retriever)

    def run(
        self,
        *,
        workspace_id: str,
        document_ids: list[str],
        documents: list[Any],
        question: str,
        conversation_context: str = "",
        forced_tool_id: Optional[str] = None,
        tool_params: Optional[dict] = None,
        db: Optional[Any] = None,
    ) -> AgentState:
        start = time.monotonic()
        initial: AgentState = {
            "workspace_id": workspace_id,
            "document_ids": document_ids,
            "documents": documents,
            "question": question,
            "conversation_context": conversation_context,
            "forced_tool_id": forced_tool_id,
            "tool_params": tool_params or {},
            "db": db,
            "attempts": 0,
            "citations": [],
            "warnings": [],
            "retrieval_latency_ms": 0,
        }
        result: AgentState = self._graph.invoke(initial)  # type: ignore[assignment]
        result["total_latency_ms"] = int((time.monotonic() - start) * 1000)
        return result

    def stream(
        self,
        *,
        workspace_id: str,
        document_ids: list[str],
        documents: list[Any],
        question: str,
        conversation_context: str = "",
        forced_tool_id: Optional[str] = None,
        tool_params: Optional[dict] = None,
        db: Optional[Any] = None,
    ):
        """Yields (node_name, partial_state_update) tuples as the graph
        executes -- used to stream coarse-grained agent progress over SSE.
        The caller accumulates partial updates to reconstruct final state."""
        initial: AgentState = {
            "workspace_id": workspace_id,
            "document_ids": document_ids,
            "documents": documents,
            "question": question,
            "conversation_context": conversation_context,
            "forced_tool_id": forced_tool_id,
            "tool_params": tool_params or {},
            "db": db,
            "attempts": 0,
            "citations": [],
            "warnings": [],
            "retrieval_latency_ms": 0,
        }
        for chunk in self._graph.stream(initial):
            for node_name, update in chunk.items():
                yield node_name, update


@lru_cache
def get_document_agent() -> DocumentAgent:
    gateway = get_model_gateway()
    retriever = Retriever(get_vector_store(), get_embedding_provider())
    return DocumentAgent(gateway, retriever)
