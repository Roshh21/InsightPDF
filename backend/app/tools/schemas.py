from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.common import Citation


class TextOutput(BaseModel):
    """Plain grounded answer -- used for direct Q&A style tools (ask_question, explain)."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)


class ListOutput(BaseModel):
    items: list[str]
    citations: list[Citation] = Field(default_factory=list)


class ExtractedEntity(BaseModel):
    name: str
    type: str
    description: str = ""


class EntitiesOutput(BaseModel):
    entities: list[ExtractedEntity]
    citations: list[Citation] = Field(default_factory=list)


class ContentSection(BaseModel):
    heading: str
    content: str = Field(description="Markdown-ish plain text. Use short paragraphs / '- ' bullet lines.")


class SectionsOutput(BaseModel):
    """The default, most-used tool output shape: a handful of labeled
    sections plus grounding citations. Covers summaries, analyses,
    extractions, checklists, briefs, reviews, explanations, etc."""

    summary: str = Field(description="1-3 sentence top-line takeaway")
    sections: list[ContentSection]
    citations: list[Citation] = Field(default_factory=list)


class ComparisonRow(BaseModel):
    document_id: str
    document_name: str
    values: dict[str, str] = Field(description="attribute label -> value for this document")


class ComparisonOutput(BaseModel):
    attributes: list[str]
    rows: list[ComparisonRow]
    narrative: str = Field(description="A short paragraph calling out the key differences/similarities")
    citations: list[Citation] = Field(default_factory=list)


class QuizQuestion(BaseModel):
    question: str
    type: str = Field(description="mcq | true_false | short_answer")
    options: list[str] = Field(default_factory=list)
    correct_answer: str
    explanation: str
    difficulty: str = Field(default="medium", description="easy | medium | hard")
    marks: Optional[int] = None


class QuizOutput(BaseModel):
    title: str
    questions: list[QuizQuestion]
    citations: list[Citation] = Field(default_factory=list)


class QuestionPaperSection(BaseModel):
    section_title: str
    instructions: Optional[str] = None
    marks_per_question: int
    questions: list[str]


class QuestionPaperOutput(BaseModel):
    title: str
    total_marks: int
    duration_minutes: Optional[int] = None
    sections: list[QuestionPaperSection]
    answer_key: Optional[list[str]] = None
    citations: list[Citation] = Field(default_factory=list)


class WebSource(BaseModel):
    title: str
    url: str
    snippet: str = ""


class WebResearchOutput(BaseModel):
    summary: str
    sections: list[ContentSection]
    citations: list[Citation] = Field(default_factory=list, description="Citations into the UPLOADED documents only")
    web_sources: list[WebSource] = Field(default_factory=list, description="External sources used, kept separate from document citations")


class ResearchQueries(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=5)


class Flashcard(BaseModel):
    front: str
    back: str


class FlashcardsOutput(BaseModel):
    cards: list[Flashcard]
    citations: list[Citation] = Field(default_factory=list)
