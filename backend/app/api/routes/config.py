from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(prefix="/config", tags=["config"])

_SLOT_KEY_REQUIRED = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


@router.get("/public")
def public_config():
    """Only ever expose non-secret, derived flags here -- never raw env
    values, API keys, or connection strings."""
    settings = get_settings()

    def configured(provider_id: str) -> bool:
        key_attr = _SLOT_KEY_REQUIRED.get(provider_id)
        return bool(key_attr and getattr(settings, key_attr, None))

    llm_slots = [
        {"slot": "primary", "provider": settings.PRIMARY_LLM_PROVIDER, "configured": configured(settings.PRIMARY_LLM_PROVIDER)},
        {"slot": "secondary", "provider": settings.SECONDARY_LLM_PROVIDER, "configured": configured(settings.SECONDARY_LLM_PROVIDER)},
        {"slot": "tertiary", "provider": settings.TERTIARY_LLM_PROVIDER, "configured": configured(settings.TERTIARY_LLM_PROVIDER)},
    ]

    return {
        "app_name": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
        "vector_store": settings.VECTOR_STORE,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_slots": llm_slots,
        "any_llm_configured": any(s["configured"] for s in llm_slots),
        "web_research_configured": bool(settings.TAVILY_API_KEY),
        "max_upload_mb": settings.MAX_UPLOAD_MB,
    }
