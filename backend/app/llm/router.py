from __future__ import annotations

from app.core.config import Settings
from app.core.enums import ModelTier, TaskType

# Which tier ("fast" vs "strong") each task type should use. This is the
# "model routing" layer -- classification/extraction don't need the same
# horsepower as multi-document comparison or report generation.
TASK_TIER_MAP: dict[TaskType, ModelTier] = {
    TaskType.CLASSIFICATION: ModelTier.FAST,
    TaskType.METADATA_EXTRACTION: ModelTier.FAST,
    TaskType.QUERY_REWRITE: ModelTier.FAST,
    TaskType.RETRIEVAL_GRADING: ModelTier.FAST,
    TaskType.GENERATION_SIMPLE: ModelTier.FAST,
    TaskType.EVALUATION: ModelTier.FAST,
    TaskType.GENERATION_COMPLEX: ModelTier.STRONG,
    TaskType.COMPARISON: ModelTier.STRONG,
    TaskType.REPORT_GENERATION: ModelTier.STRONG,
}


class ModelRouter:
    """Resolves a TaskType into a concrete model name for a given provider
    slot (primary/secondary/tertiary). Each slot has its own fast/strong
    model pair, independently configurable via env vars, so the "how much
    horsepower does this task need" decision is orthogonal to "which
    provider is currently serving the request"."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def tier_for(self, task_type: TaskType) -> ModelTier:
        return TASK_TIER_MAP.get(task_type, ModelTier.FAST)

    def model_for_slot(self, slot: str, task_type: TaskType) -> str:
        tier = self.tier_for(task_type)
        fast_attr = f"{slot.upper()}_LLM_MODEL_FAST"
        strong_attr = f"{slot.upper()}_LLM_MODEL_STRONG"
        model = getattr(self.settings, strong_attr if tier == ModelTier.STRONG else fast_attr, None)
        return model or ""
