from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from app.core.config import Settings, get_settings
from app.core.enums import TaskType
from app.core.exceptions import (
    AllProvidersUnavailableError,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderError,
    ProviderQuotaError,
    ProviderUnavailableError,
)
from app.core.logging import get_logger
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import LLMMessage, LLMProvider, LLMResponse
from app.llm.openai_compatible_provider import OpenAICompatibleProvider
from app.llm.router import ModelRouter

logger = get_logger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"


def _build_provider(provider_id: str, settings: Settings) -> Optional[LLMProvider]:
    """Instantiate a provider by id. Returns None if its required API key is
    not configured -- callers skip that slot entirely rather than failing."""
    if provider_id == "groq":
        if not settings.GROQ_API_KEY:
            return None
        return OpenAICompatibleProvider(
            name="groq", api_key=settings.GROQ_API_KEY, base_url=GROQ_BASE_URL, timeout_s=settings.LLM_REQUEST_TIMEOUT_S
        )
    if provider_id == "gemini":
        if not settings.GEMINI_API_KEY:
            return None
        return OpenAICompatibleProvider(
            name="gemini",
            api_key=settings.GEMINI_API_KEY,
            base_url=GEMINI_BASE_URL,
            timeout_s=settings.LLM_REQUEST_TIMEOUT_S,
        )
    if provider_id == "openrouter":
        if not settings.OPENROUTER_API_KEY:
            return None
        headers = {"X-Title": settings.OPENROUTER_APP_NAME}
        if settings.OPENROUTER_SITE_URL:
            headers["HTTP-Referer"] = settings.OPENROUTER_SITE_URL
        return OpenAICompatibleProvider(
            name="openrouter",
            api_key=settings.OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            timeout_s=settings.LLM_REQUEST_TIMEOUT_S,
            extra_headers=headers,
        )
    if provider_id == "anthropic":
        if not settings.ANTHROPIC_API_KEY:
            return None
        return AnthropicProvider(api_key=settings.ANTHROPIC_API_KEY, timeout_s=settings.LLM_REQUEST_TIMEOUT_S)
    if provider_id == "openai":
        if not settings.OPENAI_API_KEY:
            return None
        return OpenAICompatibleProvider(
            name="openai",
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL or OPENAI_BASE_URL,
            timeout_s=settings.LLM_REQUEST_TIMEOUT_S,
        )
    if provider_id == "none" or not provider_id:
        return None
    logger.warning("Unknown LLM provider id in configuration: %s", provider_id)
    return None


@dataclass
class ProviderSlot:
    slot: str  # "primary" | "secondary" | "tertiary"
    provider_id: str
    provider: LLMProvider


class ModelGateway:
    """
    Single entry point every agent/tool uses to talk to an LLM.

    Behavior:
      1. Resolve the task type to a model via ModelRouter, per slot.
      2. Try each configured provider slot in order (Primary -> Secondary ->
         Tertiary). A slot is skipped entirely if its API key was missing at
         startup, or if it's currently in a rate-limit cooldown window.
      3. Retriable failures (quota, rate limit, auth misconfiguration,
         connection/timeout/5xx, model-not-found) log a fallback event and
         move to the next slot. A quota/rate-limit failure additionally puts
         that provider in a cooldown so subsequent requests don't keep
         hammering an already-exhausted free tier.
      4. Non-retriable failures (malformed request / app bug) are logged and
         raised immediately -- trying another provider would not fix a bad
         request and would only hide the bug.
      5. If every configured provider fails or none are configured, raises
         AllProvidersUnavailableError with a clear, actionable message --
         never an unexplained 500.
      6. Every attempt (success or failure) is recorded to `model_usage` so
         the evaluation dashboard reflects real usage and fallback counts --
         nothing here is fabricated.
    """

    def __init__(self, settings: Settings, slots: list[ProviderSlot], router: ModelRouter):
        self.settings = settings
        self.slots = slots
        self.router = router
        self._cooldowns: dict[str, float] = {}

    @property
    def configured_provider_names(self) -> list[str]:
        return [s.provider.name for s in self.slots]

    def _in_cooldown(self, provider_name: str) -> bool:
        until = self._cooldowns.get(provider_name)
        return until is not None and time.monotonic() < until

    def _cooldown_remaining_s(self, provider_name: str) -> float:
        until = self._cooldowns.get(provider_name, 0.0)
        return max(0.0, until - time.monotonic())

    def _set_cooldown(self, provider_name: str) -> None:
        self._cooldowns[provider_name] = time.monotonic() + self.settings.LLM_PROVIDER_COOLDOWN_S

    def generate(
        self,
        task_type: TaskType,
        messages: list[LLMMessage],
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        json_mode: bool = False,
        exclude_providers: Optional[set[str]] = None,
    ) -> LLMResponse:
        exclude_providers = exclude_providers or set()
        candidates = [s for s in self.slots if s.provider.name not in exclude_providers]

        if not candidates:
            if self.slots:
                raise AllProvidersUnavailableError(
                    "All configured LLM providers were excluded after repeated failures for this request.",
                    user_message="All configured free providers failed for this request. Please try again shortly.",
                )
            raise AllProvidersUnavailableError(
                "No LLM provider is configured. Set at least one of GROQ_API_KEY, GEMINI_API_KEY, "
                "or OPENROUTER_API_KEY (all free) in the backend .env file.",
                user_message=(
                    "No LLM provider is configured yet. Add a free API key (Groq, Gemini, or OpenRouter) "
                    "to the backend .env file -- see the README for how to get one."
                ),
            )

        requested_provider = candidates[0].provider.name
        last_error: Optional[Exception] = None
        skipped_on_cooldown: list[str] = []

        for i, slot in enumerate(candidates):
            provider = slot.provider
            is_fallback_attempt = i > 0

            if self._in_cooldown(provider.name):
                skipped_on_cooldown.append(provider.name)
                logger.info(
                    "Skipping %s (%s slot): still in rate-limit cooldown for %.0fs",
                    provider.name,
                    slot.slot,
                    self._cooldown_remaining_s(provider.name),
                )
                continue

            model = self.router.model_for_slot(slot.slot, task_type)
            try:
                response = provider.generate(
                    messages, model=model, system=system, max_tokens=max_tokens, temperature=temperature, json_mode=json_mode
                )
                self._log_usage(
                    task_type=task_type,
                    provider_requested=requested_provider,
                    provider_used=provider.name,
                    model=model,
                    success=True,
                    fallback_used=is_fallback_attempt,
                    latency_ms=response.latency_ms,
                    tokens_in=response.tokens_in,
                    tokens_out=response.tokens_out,
                )
                if is_fallback_attempt:
                    logger.warning(
                        "Fell back to %s (%s slot) for task=%s after %s failed",
                        provider.name,
                        slot.slot,
                        task_type.value,
                        requested_provider,
                    )
                return response

            except ProviderBadRequestError as exc:
                self._log_usage(
                    task_type=task_type,
                    provider_requested=requested_provider,
                    provider_used=provider.name,
                    model=model,
                    success=False,
                    fallback_used=is_fallback_attempt,
                    error_type="bad_request",
                    error_message=str(exc),
                )
                logger.error("Bad request to %s (not failing over -- likely an application bug): %s", provider.name, exc)
                raise

            except ProviderQuotaError as exc:
                self._set_cooldown(provider.name)
                self._log_usage(
                    task_type=task_type,
                    provider_requested=requested_provider,
                    provider_used=provider.name,
                    model=model,
                    success=False,
                    fallback_used=is_fallback_attempt,
                    error_type="quota",
                    error_message=str(exc),
                )
                logger.warning(
                    "%s hit a rate limit/quota (cooling down %ss), trying next provider if any: %s",
                    provider.name,
                    self.settings.LLM_PROVIDER_COOLDOWN_S,
                    exc,
                )
                last_error = exc
                continue

            except (ProviderAuthError, ProviderUnavailableError) as exc:
                error_type = "auth" if isinstance(exc, ProviderAuthError) else "unavailable"
                self._log_usage(
                    task_type=task_type,
                    provider_requested=requested_provider,
                    provider_used=provider.name,
                    model=model,
                    success=False,
                    fallback_used=is_fallback_attempt,
                    error_type=error_type,
                    error_message=str(exc),
                )
                logger.warning("%s failed (%s), trying next provider if any: %s", provider.name, error_type, exc)
                last_error = exc
                continue

            except ProviderError as exc:  # pragma: no cover - defensive catch-all for the base class
                self._log_usage(
                    task_type=task_type,
                    provider_requested=requested_provider,
                    provider_used=provider.name,
                    model=model,
                    success=False,
                    fallback_used=is_fallback_attempt,
                    error_type="unknown",
                    error_message=str(exc),
                )
                last_error = exc
                continue

        if skipped_on_cooldown and not last_error:
            raise AllProvidersUnavailableError(
                f"All configured providers are currently rate-limited (cooling down): {', '.join(skipped_on_cooldown)}.",
                user_message=(
                    f"All configured free providers ({', '.join(skipped_on_cooldown)}) are currently rate-limited. "
                    "They'll be retried automatically shortly -- please try again in a minute."
                ),
            )
        raise AllProvidersUnavailableError(
            f"All configured free providers are temporarily unavailable. Last error: {last_error}",
            user_message="All configured free providers are temporarily unavailable. Please try again shortly.",
        )

    def _log_usage(self, **kwargs) -> None:
        try:
            from app.database import session_scope
            from app.models.model_usage import ModelUsage

            with session_scope() as db:
                db.add(ModelUsage(**kwargs))
        except Exception as exc:  # never let observability break the request
            logger.debug("Failed to record model usage: %s", exc)


@lru_cache
def get_model_gateway() -> ModelGateway:
    settings = get_settings()
    router = ModelRouter(settings)

    slots: list[ProviderSlot] = []
    for slot_name, provider_id in (
        ("primary", settings.PRIMARY_LLM_PROVIDER),
        ("secondary", settings.SECONDARY_LLM_PROVIDER),
        ("tertiary", settings.TERTIARY_LLM_PROVIDER),
    ):
        provider = _build_provider(provider_id, settings)
        if provider is None:
            if provider_id and provider_id != "none":
                logger.info(
                    "%s LLM slot (%s) has no API key configured -- skipping. "
                    "The app will still start; that slot just won't be used.",
                    slot_name.capitalize(),
                    provider_id,
                )
            continue
        slots.append(ProviderSlot(slot=slot_name, provider_id=provider_id, provider=provider))

    if not slots:
        logger.warning(
            "No LLM provider is configured (checked PRIMARY/SECONDARY/TERTIARY_LLM_PROVIDER). "
            "The app will start, but chat/tools will return a clear error until you add a free API key "
            "(Groq, Gemini, or OpenRouter -- see README)."
        )

    return ModelGateway(settings=settings, slots=slots, router=router)
