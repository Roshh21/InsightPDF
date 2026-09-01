from __future__ import annotations

from enum import Enum


class DocumentType(str, Enum):
    RESEARCH_PAPER = "research_paper"
    LITERATURE = "literature"
    STUDY_MATERIAL = "study_material"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    BUSINESS_REPORT = "business_report"
    GENERIC = "generic"


class ProcessingStatus(str, Enum):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ToolExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ModelTier(str, Enum):
    FAST = "fast"
    STRONG = "strong"


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    METADATA_EXTRACTION = "metadata_extraction"
    QUERY_REWRITE = "query_rewrite"
    RETRIEVAL_GRADING = "retrieval_grading"
    GENERATION_SIMPLE = "generation_simple"
    GENERATION_COMPLEX = "generation_complex"
    COMPARISON = "comparison"
    REPORT_GENERATION = "report_generation"
    EVALUATION = "evaluation"


class ProviderName(str, Enum):
    GROQ = "groq"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    NONE = "none"


class ResultKind(str, Enum):
    """Discriminator for how the frontend should render a tool result."""

    TEXT = "text"
    SECTIONS = "sections"
    TABLE = "table"
    COMPARISON = "comparison"
    LIST = "list"
    ENTITIES = "entities"
    QUIZ = "quiz"
    QUESTION_PAPER = "question_paper"
    FLASHCARDS = "flashcards"
    REPORT = "report"
