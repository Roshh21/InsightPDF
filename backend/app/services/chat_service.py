from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.document_agent import get_document_agent
from app.core.exceptions import InsightPDFError
from app.core.logging import get_logger
from app.evaluation.tracker import record_run_evaluation
from app.models.chat import ChatMessage, ChatSession
from app.services.document_service import resolve_documents
from app.services.model_usage_lookup import latest_successful_usage_since
from datetime import datetime, timezone

logger = get_logger(__name__)


def get_or_create_session(
    db: Session,
    *,
    workspace_id: str,
    session_id: Optional[str] = None,
    document_ids: Optional[list[str]] = None,
    spoiler_level: Optional[str] = None,
    title: Optional[str] = None,
) -> ChatSession:
    if session_id:
        session = db.get(ChatSession, session_id)
        if session is not None:
            if document_ids is not None:
                session.document_ids = document_ids
            if spoiler_level is not None:
                session.spoiler_level = spoiler_level
            return session
    session = ChatSession(
        workspace_id=workspace_id, document_ids=document_ids or [], spoiler_level=spoiler_level, title=title
    )
    db.add(session)
    db.flush()
    return session


def list_messages(db: Session, session_id: str) -> list[ChatMessage]:
    return list(
        db.execute(
            select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at)
        ).scalars().all()
    )


def _build_conversation_context(messages: list[ChatMessage], max_turns: int = 6) -> str:
    recent = messages[-max_turns * 2 :]
    lines = []
    for m in recent:
        prefix = "User" if m.role == "user" else "Assistant"
        content = m.content if len(m.content) < 500 else m.content[:500] + "..."
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def send_message(
    db: Session,
    *,
    session: ChatSession,
    user_content: str,
) -> tuple[ChatMessage, ChatMessage]:
    """Persists the user message, runs the agent, persists + returns the
    assistant reply. Raises InsightPDFError subclasses on unrecoverable
    failure (already-logged by the gateway/tools)."""

    prior_messages = list_messages(db, session.id)
    conversation_context = _build_conversation_context(prior_messages)
    if session.spoiler_level:
        conversation_context = f"[User-selected spoiler level: {session.spoiler_level}]\n{conversation_context}"

    user_message = ChatMessage(session_id=session.id, role="user", content=user_content)
    db.add(user_message)
    # Commit now (rather than just flush) so this session doesn't hold an
    # open write transaction across the agent run that follows -- the agent
    # makes several LLM calls, each of which independently logs usage via
    # its own short-lived session (see ModelGateway._log_usage). Under
    # SQLite specifically, a long-open write transaction here would starve
    # those out ("database is locked"); Postgres wouldn't have this problem,
    # but there's no downside to checkpointing here regardless of backend.
    db.commit()
    db.refresh(user_message)

    documents = resolve_documents(db, session.workspace_id, session.document_ids)
    document_ids = [d.id for d in documents]

    agent = get_document_agent()
    start_dt = datetime.now(timezone.utc)
    tool_params = {"spoiler_level": session.spoiler_level} if session.spoiler_level else {}

    try:
        result = agent.run(
            workspace_id=session.workspace_id,
            document_ids=document_ids,
            documents=documents,
            question=user_content,
            conversation_context=conversation_context,
            tool_params=tool_params,
            db=db,
        )
    except InsightPDFError:
        # Preserve the specific error (e.g. AllProvidersUnavailableError ->
        # 503 with an actionable message) rather than flattening it into a
        # generic failure.
        raise
    except Exception as exc:
        raise InsightPDFError(
            f"Chat turn failed: {exc}", user_message="Something went wrong answering that. Please try again."
        ) from exc

    usage = latest_successful_usage_since(db, start_dt)
    citations = result.get("citations") or []
    citations_payload = [c.model_dump(mode="json") if hasattr(c, "model_dump") else c for c in citations]

    assistant_message = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=result.get("answer_text", ""),
        citations=citations_payload,
        tool_used=result.get("tool_id"),
        result_payload=result.get("tool_result"),
        model_used=usage.model if usage else None,
        provider_used=usage.provider_used if usage else None,
        latency_ms=result.get("total_latency_ms"),
    )
    db.add(assistant_message)
    db.flush()

    record_run_evaluation(
        db,
        run_type="chat",
        answer_text=result.get("answer_text", ""),
        evidence_chunks=result.get("evidence") or [],
        citations=citations,
        total_latency_ms=result.get("total_latency_ms", 0),
        retrieval_latency_ms=result.get("retrieval_latency_ms"),
        retries=result.get("attempts", 0),
        chat_message_id=assistant_message.id,
    )

    return user_message, assistant_message


_NODE_LABELS = {
    "understand_intent": "Understanding your question",
    "run_tool": "Running tool",
    "rewrite_query": "Rewriting search query",
    "retrieve": "Searching documents",
    "refine_query": "Refining search",
    "analyze": "Generating answer",
    "validate": "Checking citations",
}


def stream_message(db: Session, *, session: ChatSession, user_content: str):
    """Generator yielding coarse-grained agent progress events, then a final
    'done' event with the persisted assistant message -- consumed by the
    /chat/stream SSE route. Real progress from LangGraph's own step stream,
    not a simulated/fake progress bar."""
    import time as _time

    prior_messages = list_messages(db, session.id)
    conversation_context = _build_conversation_context(prior_messages)
    if session.spoiler_level:
        conversation_context = f"[User-selected spoiler level: {session.spoiler_level}]\n{conversation_context}"

    user_message = ChatMessage(session_id=session.id, role="user", content=user_content)
    db.add(user_message)
    db.commit()
    db.refresh(user_message)

    documents = resolve_documents(db, session.workspace_id, session.document_ids)
    document_ids = [d.id for d in documents]

    agent = get_document_agent()
    tool_params = {"spoiler_level": session.spoiler_level} if session.spoiler_level else {}

    accumulated: dict = {}
    start_dt = datetime.now(timezone.utc)
    start_monotonic = _time.monotonic()

    yield {"event": "progress", "node": "start", "label": "Starting"}
    try:
        for node_name, update in agent.stream(
            workspace_id=session.workspace_id,
            document_ids=document_ids,
            documents=documents,
            question=user_content,
            conversation_context=conversation_context,
            tool_params=tool_params,
            db=db,
        ):
            accumulated.update(update)
            yield {"event": "progress", "node": node_name, "label": _NODE_LABELS.get(node_name, node_name)}
    except InsightPDFError as exc:
        yield {"event": "error", "message": exc.user_message}
        return
    except Exception as exc:
        logger.exception("Unexpected error during streaming chat turn")
        yield {"event": "error", "message": "Something went wrong answering that. Please try again."}
        return

    total_latency_ms = int((_time.monotonic() - start_monotonic) * 1000)
    usage = latest_successful_usage_since(db, start_dt)
    citations = accumulated.get("citations") or []
    citations_payload = [c.model_dump(mode="json") if hasattr(c, "model_dump") else c for c in citations]

    assistant_message = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=accumulated.get("answer_text", ""),
        citations=citations_payload,
        tool_used=accumulated.get("tool_id"),
        result_payload=accumulated.get("tool_result"),
        model_used=usage.model if usage else None,
        provider_used=usage.provider_used if usage else None,
        latency_ms=total_latency_ms,
    )
    db.add(assistant_message)
    db.flush()

    record_run_evaluation(
        db,
        run_type="chat",
        answer_text=accumulated.get("answer_text", ""),
        evidence_chunks=accumulated.get("evidence") or [],
        citations=citations,
        total_latency_ms=total_latency_ms,
        retrieval_latency_ms=accumulated.get("retrieval_latency_ms"),
        retries=accumulated.get("attempts", 0),
        chat_message_id=assistant_message.id,
    )
    db.commit()
    db.refresh(assistant_message)

    yield {
        "event": "done",
        "session_id": session.id,
        "assistant_message": {
            "id": assistant_message.id,
            "role": assistant_message.role,
            "content": assistant_message.content,
            "citations": citations_payload,
            "tool_used": assistant_message.tool_used,
            "result_payload": assistant_message.result_payload,
            "model_used": assistant_message.model_used,
            "provider_used": assistant_message.provider_used,
            "latency_ms": assistant_message.latency_ms,
            "created_at": assistant_message.created_at.isoformat(),
        },
    }
