from __future__ import annotations

import time
from typing import Optional

from app.core.exceptions import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderQuotaError,
    ProviderUnavailableError,
)
from app.core.logging import get_logger
from app.llm.base import LLMMessage, LLMProvider, LLMResponse

logger = get_logger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """
    Works with any backend that speaks the OpenAI `/chat/completions` wire
    format -- which today includes Groq, Google Gemini (via its official
    OpenAI-compatibility endpoint, see
    https://ai.google.dev/gemini-api/docs/openai), OpenRouter, and OpenAI
    itself. One implementation, one dependency (`openai`), instantiated
    once per configured provider slot with a different base_url/api_key.

    This is a deliberate design choice over provider-specific SDKs: it
    keeps error handling, JSON-mode support, and availability checks
    consistent across every free provider this app talks to, and avoids
    depending on provider SDKs (e.g. `google-genai`) that are unnecessary
    when the provider already exposes a stable OpenAI-compatible surface.
    """

    def __init__(
        self,
        name: str,
        api_key: Optional[str],
        base_url: str,
        timeout_s: int = 60,
        supports_json_mode: bool = True,
        extra_headers: Optional[dict[str, str]] = None,
    ):
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.supports_json_mode = supports_json_mode
        self.extra_headers = extra_headers or {}
        self._client = None
        self._timeout_s = timeout_s

    def _get_client(self):
        if not self.api_key:
            raise ProviderAuthError(f"{self.name}: no API key is configured")
        if self._client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover
                raise ProviderUnavailableError("openai package not installed") from exc
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self._timeout_s,
                default_headers=self.extra_headers or None,
            )
        return self._client

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> LLMResponse:
        import openai

        client = self._get_client()
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend({"role": m.role, "content": m.content} for m in messages)

        kwargs: dict = {}
        if json_mode and self.supports_json_mode:
            # Broadly-supported "JSON mode" (not the stricter json_schema
            # mode, which not every free provider/model implements
            # consistently) -- nudges the model to emit valid JSON. Our own
            # parsing/validation in app/llm/structured.py remains the
            # authoritative check regardless of whether this helped.
            kwargs["response_format"] = {"type": "json_object"}

        start = time.monotonic()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=chat_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        except openai.AuthenticationError as exc:
            raise ProviderAuthError(f"{self.name}: {exc}") from exc
        except openai.RateLimitError as exc:
            raise ProviderQuotaError(f"{self.name}: {exc}") from exc
        except openai.APIConnectionError as exc:
            raise ProviderUnavailableError(f"{self.name}: {exc}") from exc
        except openai.BadRequestError as exc:
            # Some providers (observed with certain OpenRouter free models)
            # reject response_format even when advertised as supported.
            # Retry once without it before giving up on this provider.
            if json_mode and "response_format" in kwargs:
                logger.info("%s rejected response_format, retrying without it", self.name)
                try:
                    resp = client.chat.completions.create(
                        model=model, messages=chat_messages, max_tokens=max_tokens, temperature=temperature
                    )
                except Exception as exc2:
                    raise ProviderBadRequestError(f"{self.name}: {exc2}") from exc2
            else:
                raise ProviderBadRequestError(f"{self.name}: {exc}") from exc
        except openai.APIStatusError as exc:
            if exc.status_code == 429:
                raise ProviderQuotaError(f"{self.name}: {exc}") from exc
            if exc.status_code in (401, 403):
                raise ProviderAuthError(f"{self.name}: {exc}") from exc
            if exc.status_code >= 500 or exc.status_code == 503:
                raise ProviderUnavailableError(f"{self.name}: {exc}") from exc
            if exc.status_code == 404:
                # Most commonly: the configured model id no longer exists
                # (e.g. an OpenRouter free model was delisted). Treat as
                # "provider unavailable for this request" so the gateway
                # fails over rather than surfacing a raw 404.
                raise ProviderUnavailableError(
                    f"{self.name}: model '{model}' not found (it may have been renamed or delisted): {exc}"
                ) from exc
            raise ProviderBadRequestError(f"{self.name}: {exc}") from exc
        except Exception as exc:  # network errors, timeouts, etc.
            raise ProviderUnavailableError(f"{self.name}: {exc}") from exc

        latency_ms = int((time.monotonic() - start) * 1000)
        choice = resp.choices[0] if resp.choices else None
        text = (choice.message.content or "") if choice else ""
        usage = resp.usage
        return LLMResponse(
            text=text,
            provider=self.name,
            model=model,
            latency_ms=latency_ms,
            tokens_in=getattr(usage, "prompt_tokens", None) if usage else None,
            tokens_out=getattr(usage, "completion_tokens", None) if usage else None,
        )

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            client = self._get_client()
            client.models.list()
            return True
        except Exception as exc:
            logger.info("%s availability check failed: %s", self.name, exc)
            return False
