from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMMessage:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_ms: int
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None


class LLMProvider(ABC):
    """Abstraction over chat-completion backends.

    The agent/tools only ever talk to `ModelGateway` (app/llm/gateway.py),
    never to a concrete provider directly -- this is what makes cloud/local
    failover and per-task model routing possible without touching business
    logic.
    """

    name: str = "base"

    @abstractmethod
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
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Best-effort, side-effect-free-ish availability check.

        Must actually verify (ping / lightweight call) rather than assuming
        availability from configuration alone.
        """
