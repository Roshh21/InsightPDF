from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from app.core.config import get_settings
from app.llm.gateway import get_model_gateway

_CACHE_TTL_S = 20.0
_cache: dict[str, tuple[float, bool]] = {}

_SLOT_KEY_REQUIRED = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _cached_check(key: str, check_fn) -> bool:
    now = time.monotonic()
    cached = _cache.get(key)
    if cached and (now - cached[0]) < _CACHE_TTL_S:
        return cached[1]
    result = bool(check_fn())
    _cache[key] = (now, result)
    return result


@dataclass
class ProviderStatus:
    slot: str
    provider: str
    configured: bool
    available: bool
    cooling_down: bool
    cooldown_remaining_s: float
    model_fast: str
    model_strong: str


@dataclass
class ModelStatusSnapshot:
    providers: list[ProviderStatus]
    active_provider: Optional[str]
    vector_store: str
    embedding_model: str


def get_model_status(force_refresh: bool = False) -> ModelStatusSnapshot:
    settings = get_settings()
    gateway = get_model_gateway()

    if force_refresh:
        _cache.clear()

    by_name = {s.provider.name: s for s in gateway.slots}
    providers: list[ProviderStatus] = []
    active_provider: Optional[str] = None

    for slot_name, provider_id, model_fast, model_strong in (
        ("primary", settings.PRIMARY_LLM_PROVIDER, settings.PRIMARY_LLM_MODEL_FAST, settings.PRIMARY_LLM_MODEL_STRONG),
        ("secondary", settings.SECONDARY_LLM_PROVIDER, settings.SECONDARY_LLM_MODEL_FAST, settings.SECONDARY_LLM_MODEL_STRONG),
        ("tertiary", settings.TERTIARY_LLM_PROVIDER, settings.TERTIARY_LLM_MODEL_FAST, settings.TERTIARY_LLM_MODEL_STRONG),
    ):
        key_attr = _SLOT_KEY_REQUIRED.get(provider_id)
        configured = bool(key_attr and getattr(settings, key_attr, None))
        slot = by_name.get(provider_id) if configured else None

        available = False
        cooling_down = False
        cooldown_remaining = 0.0
        if slot is not None:
            cooling_down = gateway._in_cooldown(slot.provider.name)
            cooldown_remaining = gateway._cooldown_remaining_s(slot.provider.name)
            if cooling_down:
                available = False
            else:
                available = _cached_check(f"provider:{slot.provider.name}", slot.provider.is_available)

        if available and active_provider is None:
            active_provider = provider_id

        providers.append(
            ProviderStatus(
                slot=slot_name,
                provider=provider_id,
                configured=configured,
                available=available,
                cooling_down=cooling_down,
                cooldown_remaining_s=round(cooldown_remaining, 1),
                model_fast=model_fast,
                model_strong=model_strong,
            )
        )

    return ModelStatusSnapshot(
        providers=providers,
        active_provider=active_provider,
        vector_store=settings.VECTOR_STORE,
        embedding_model=settings.EMBEDDING_MODEL,
    )
