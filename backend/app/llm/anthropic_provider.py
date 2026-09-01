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


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: Optional[str], timeout_s: int = 60):
        self.api_key = api_key
        self._client = None
        self._timeout_s = timeout_s

    def _get_client(self):
        if not self.api_key:
            raise ProviderAuthError("ANTHROPIC_API_KEY is not configured")
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise ProviderUnavailableError("anthropic package not installed") from exc
            self._client = anthropic.Anthropic(api_key=self.api_key, timeout=self._timeout_s)
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
        import anthropic

        client = self._get_client()
        start = time.monotonic()
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or anthropic.NOT_GIVEN,
                messages=[{"role": m.role, "content": m.content} for m in messages],
            )
        except anthropic.AuthenticationError as exc:
            raise ProviderAuthError(str(exc)) from exc
        except anthropic.RateLimitError as exc:
            raise ProviderQuotaError(str(exc)) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderUnavailableError(str(exc)) from exc
        except anthropic.BadRequestError as exc:
            raise ProviderBadRequestError(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code == 429:
                raise ProviderQuotaError(str(exc)) from exc
            if exc.status_code in (401, 403):
                raise ProviderAuthError(str(exc)) from exc
            if exc.status_code >= 500:
                raise ProviderUnavailableError(str(exc)) from exc
            raise ProviderBadRequestError(str(exc)) from exc
        except Exception as exc:  # network errors, timeouts, etc.
            raise ProviderUnavailableError(str(exc)) from exc

        latency_ms = int((time.monotonic() - start) * 1000)
        text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")
        return LLMResponse(
            text=text,
            provider=self.name,
            model=model,
            latency_ms=latency_ms,
            tokens_in=getattr(resp.usage, "input_tokens", None),
            tokens_out=getattr(resp.usage, "output_tokens", None),
        )

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            client = self._get_client()
            # Lightweight, verifies auth + connectivity without much cost.
            client.models.list(limit=1)
            return True
        except Exception as exc:
            logger.info("Anthropic availability check failed: %s", exc)
            return False
