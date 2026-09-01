from __future__ import annotations

from app.core.enums import TaskType
from app.core.logging import get_logger
from app.llm.base import LLMMessage
from app.llm.gateway import ModelGateway

logger = get_logger(__name__)

_REWRITE_SYSTEM = """You rewrite user questions into effective search queries for retrieving \
relevant passages from a document via semantic (embedding) search.

Rules:
- Resolve pronouns and vague references using the recent conversation history if given \
(e.g. "compare it with the second one" -> name the actual documents/entities).
- Expand abbreviations and add likely synonyms if it helps recall.
- Keep it a single, focused query -- not a list, not a restated essay.
- If the question is already specific and self-contained, return it close to unchanged.
Return ONLY the rewritten query text."""

_REFINE_SYSTEM = """The first retrieval attempt for this question did not return enough \
relevant evidence. Propose a DIFFERENT search query that approaches the same information \
need from another angle (different terminology, a narrower or broader phrasing, or \
targeting a specific likely section such as "results" or "methodology").
Return ONLY the new query text, nothing else."""


def rewrite_query(gateway: ModelGateway, question: str, conversation_context: str = "") -> str:
    prompt = question if not conversation_context else f"Recent conversation:\n{conversation_context}\n\nCurrent question: {question}"
    try:
        response = gateway.generate(
            TaskType.QUERY_REWRITE,
            [LLMMessage(role="user", content=prompt)],
            system=_REWRITE_SYSTEM,
            max_tokens=200,
            temperature=0.0,
        )
        rewritten = response.text.strip().strip('"')
        return rewritten or question
    except Exception as exc:
        logger.warning("Query rewrite failed, using original question: %s", exc)
        return question


def refine_query(gateway: ModelGateway, original_question: str, previous_query: str) -> str:
    prompt = f"Original question: {original_question}\nPrevious search query (insufficient results): {previous_query}"
    try:
        response = gateway.generate(
            TaskType.QUERY_REWRITE,
            [LLMMessage(role="user", content=prompt)],
            system=_REFINE_SYSTEM,
            max_tokens=200,
            temperature=0.3,
        )
        refined = response.text.strip().strip('"')
        return refined or original_question
    except Exception as exc:
        logger.warning("Query refinement failed, reusing previous query: %s", exc)
        return previous_query
