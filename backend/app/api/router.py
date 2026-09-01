from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import chat, config, documents, evaluation, models_status, tools, workspaces

api_router = APIRouter()
api_router.include_router(workspaces.router)
api_router.include_router(documents.router)
api_router.include_router(tools.router)
api_router.include_router(chat.router)
api_router.include_router(evaluation.router)
api_router.include_router(models_status.router)
api_router.include_router(config.router)
