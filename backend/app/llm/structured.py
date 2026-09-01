from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.core.enums import TaskType
from app.core.exceptions import AllProvidersUnavailableError, StructuredOutputError
from app.core.logging import get_logger
from app.llm.base import LLMMessage
from app.llm.gateway import ModelGateway

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> str:
    """Robustly pull a JSON object/array out of an LLM response.

    Handles markdown code fences and any leading/trailing prose, which is
    the crude-JSON-parsing failure mode the previous implementation had.
    """
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    # Fall back to the first {...} or [...] span.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
    return text.strip()


def generate_structured(
    gateway: ModelGateway,
    schema: type[T],
    *,
    task_type: TaskType,
    system: str,
    user_prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.1,
    max_repair_attempts_per_provider: int = 1,
    max_total_attempts: int = 4,
) -> T:
    """Ask the model for JSON matching `schema` and parse/validate it.

    Two layers of recovery, both bounded:
      1. If a provider's response fails to parse/validate, ask that SAME
         provider to correct it (up to `max_repair_attempts_per_provider`
         times) -- most malformed-JSON cases are a stray comma or an extra
         code fence, and the fastest fix is showing the model its own
         mistake.
      2. If a provider keeps failing after its repair attempts, it is
         excluded and the request moves to the next configured provider
         (this is what makes "structured-output failure" one of the
         triggers for provider failover, not just transport-level errors).

    Raises StructuredOutputError if every provider's output is unparsable
    even after recovery, or AllProvidersUnavailableError if the gateway
    itself has nothing left to try.
    """

    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    full_system = (
        f"{system}\n\n"
        "You must respond with ONLY a single valid JSON object/array and nothing else -- "
        "no markdown fences, no commentary, no preamble.\n"
        f"It must validate against this JSON Schema:\n{schema_json}"
    )

    base_messages = [LLMMessage(role="user", content=user_prompt)]
    messages = list(base_messages)
    excluded_providers: set[str] = set()
    current_provider: str | None = None
    repairs_for_current_provider = 0
    last_error: Exception | None = None

    for attempt in range(max_total_attempts):
        try:
            response = gateway.generate(
                task_type,
                messages,
                system=full_system,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=True,
                exclude_providers=excluded_providers,
            )
        except AllProvidersUnavailableError:
            raise

        if response.provider != current_provider:
            current_provider = response.provider
            repairs_for_current_provider = 0

        raw = _extract_json(response.text)
        try:
            data = json.loads(raw)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            repairs_for_current_provider += 1
            logger.warning(
                "Structured output parse failed (provider=%s, attempt %s/%s): %s",
                current_provider,
                attempt + 1,
                max_total_attempts,
                exc,
            )
            if repairs_for_current_provider > max_repair_attempts_per_provider:
                # This provider isn't converging on valid output -- stop
                # asking it and move to the next configured provider instead
                # of burning the whole attempt budget on one bad source.
                excluded_providers.add(current_provider)
                messages = list(base_messages)
                current_provider = None
                repairs_for_current_provider = 0
            else:
                messages = messages + [
                    LLMMessage(role="assistant", content=response.text),
                    LLMMessage(
                        role="user",
                        content=(
                            f"That was not valid JSON matching the schema. Error: {exc}\n"
                            "Return ONLY the corrected JSON object, nothing else."
                        ),
                    ),
                ]

    raise StructuredOutputError(
        f"No provider returned valid structured output after {max_total_attempts} attempts. "
        f"Last error: {last_error}"
    )
