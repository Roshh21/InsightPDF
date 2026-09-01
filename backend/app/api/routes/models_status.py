from __future__ import annotations

from fastapi import APIRouter, Query

from app.llm.status import get_model_status
from app.schemas.evaluation import ModelStatusResponse, ProviderStatusResponse

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/status", response_model=ModelStatusResponse)
def model_status(refresh: bool = Query(default=False)):
    snapshot = get_model_status(force_refresh=refresh)
    return ModelStatusResponse(
        providers=[ProviderStatusResponse(**p.__dict__) for p in snapshot.providers],
        active_provider=snapshot.active_provider,
        vector_store=snapshot.vector_store,
        embedding_model=snapshot.embedding_model,
    )
