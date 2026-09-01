from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.exceptions import InsightPDFError
from app.database import get_db
from app.models.chat import ChatSession
from app.schemas.chat import (
    ChatMessageRequest,
    ChatMessageResponse,
    ChatSessionCreateRequest,
    ChatSessionResponse,
    ChatTurnResponse,
)
from app.services import chat_service, workspace_service

router = APIRouter(tags=["chat"])


@router.post("/workspaces/{workspace_id}/chat/sessions", response_model=ChatSessionResponse)
def create_chat_session(workspace_id: str, payload: ChatSessionCreateRequest, db: Session = Depends(get_db)):
    workspace_service.get_workspace(db, workspace_id)
    session = chat_service.get_or_create_session(
        db,
        workspace_id=workspace_id,
        document_ids=payload.document_ids,
        spoiler_level=payload.spoiler_level,
        title=payload.title,
    )
    db.commit()
    db.refresh(session)
    return session


@router.get("/workspaces/{workspace_id}/chat/sessions", response_model=list[ChatSessionResponse])
def list_chat_sessions(workspace_id: str, db: Session = Depends(get_db)):
    rows = db.execute(
        select(ChatSession).where(ChatSession.workspace_id == workspace_id).order_by(ChatSession.created_at.desc())
    ).scalars().all()
    return list(rows)


@router.get("/chat/sessions/{session_id}/messages", response_model=list[ChatMessageResponse])
def get_messages(session_id: str, db: Session = Depends(get_db)):
    return chat_service.list_messages(db, session_id)


@router.post("/chat/messages", response_model=ChatTurnResponse)
def post_message(payload: ChatMessageRequest, db: Session = Depends(get_db)):
    session = None
    if payload.session_id:
        session = db.get(ChatSession, payload.session_id)
    if session is None:
        # A session_id-less request must still be scoped to a workspace via
        # document_ids' owning workspace -- for simplicity the frontend
        # always creates a session first, so this path is a convenience
        # fallback that requires document_ids to belong to a single workspace.
        raise InsightPDFError(
            "session_id is required", user_message="No chat session was specified."
        )
    user_message, assistant_message = chat_service.send_message(db, session=session, user_content=payload.message)
    db.commit()
    db.refresh(user_message)
    db.refresh(assistant_message)
    return ChatTurnResponse(
        session_id=session.id,
        user_message=ChatMessageResponse.model_validate(user_message),
        assistant_message=ChatMessageResponse.model_validate(assistant_message),
    )


@router.post("/chat/stream")
def stream_message(payload: ChatMessageRequest, db: Session = Depends(get_db)):
    session = db.get(ChatSession, payload.session_id) if payload.session_id else None
    if session is None:
        raise InsightPDFError("session_id is required", user_message="No chat session was specified.")

    def event_source():
        try:
            for event in chat_service.stream_message(db, session=session, user_content=payload.message):
                yield f"data: {json.dumps(event)}\n\n"
        except InsightPDFError as exc:
            yield f"data: {json.dumps({'event': 'error', 'message': exc.user_message})}\n\n"
        except Exception:  # pragma: no cover - defensive
            yield f"data: {json.dumps({'event': 'error', 'message': 'Something went wrong. Please try again.'})}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")
