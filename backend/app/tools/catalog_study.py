"""Study material / education mode tools (spec section 8).

`chapter_summary` and `generate_quiz` are shared with the literature catalog
/ universal catalog respectively rather than duplicated here.
"""
from __future__ import annotations

from app.core.enums import DocumentType, ResultKind, TaskType
from app.tools.prompt_tool import PromptTool, PromptToolSpec
from app.tools.registry import register_tool

_SM = [DocumentType.STUDY_MATERIAL]

_GROUNDING_NOTE = (
    "\n\nIMPORTANT: only use content that is actually present in the evidence. Never invent facts, "
    "formulas, or questions about topics not covered in the uploaded material."
)

_SPECS = [
    PromptToolSpec(
        id="simplified_explanation",
        name="Simplified Explanation",
        description="Re-explain a concept or section in simpler, more accessible language.",
        category="study_material",
        applicable_types=_SM,
        system_prompt="Re-explain the requested concept/section in simple language suitable for a "
        "student encountering it for the first time. Use short sentences, concrete examples, and "
        "define any jargon you must use." + _GROUNDING_NOTE,
        icon="lightbulb",
    ),
    PromptToolSpec(
        id="notes_generation",
        name="Notes",
        description="Generate concise, well-organized study notes from the material.",
        category="study_material",
        applicable_types=_SM,
        system_prompt="Generate concise, well-organized study notes: short headings with bullet points "
        "underneath, capturing the key facts, definitions, and relationships a student needs to "
        "remember." + _GROUNDING_NOTE,
        top_k=16,
        icon="notebook-pen",
    ),
    PromptToolSpec(
        id="important_concepts",
        name="Important Concepts",
        description="List and briefly explain the most important concepts covered.",
        category="study_material",
        applicable_types=_SM,
        system_prompt="List the most important concepts in the material with a 1-2 sentence explanation "
        "of each." + _GROUNDING_NOTE,
        result_kind=ResultKind.SECTIONS,
        icon="brain",
    ),
    PromptToolSpec(
        id="definitions_extraction",
        name="Definitions",
        description="Extract term/definition pairs from the material.",
        category="study_material",
        applicable_types=_SM,
        system_prompt="Extract every clearly defined term and its definition as it appears in the "
        "material. Format each item as 'Term: Definition'." + _GROUNDING_NOTE,
        result_kind=ResultKind.LIST,
        icon="book-a",
    ),
    PromptToolSpec(
        id="formula_extraction",
        name="Formula Extraction",
        description="Extract formulas/equations mentioned, with variable definitions.",
        category="study_material",
        applicable_types=_SM,
        system_prompt="Extract every formula/equation present in the material. For each, state the "
        "formula (using plain-text notation, e.g. 'v = u + a*t') and define each variable. If truly no "
        "formulas are present, say so rather than inventing any." + _GROUNDING_NOTE,
        result_kind=ResultKind.LIST,
        icon="sigma",
    ),
    PromptToolSpec(
        id="flashcards",
        name="Flashcards",
        description="Generate front/back flashcards covering the key facts and concepts.",
        category="study_material",
        applicable_types=_SM,
        system_prompt="Generate flashcards: a short prompt on the front (a term or question) and a "
        "concise, correct answer on the back. Default to 12 cards unless the user configuration "
        "specifies a different count." + _GROUNDING_NOTE,
        result_kind=ResultKind.FLASHCARDS,
        top_k=16,
        icon="layers",
    ),
    PromptToolSpec(
        id="practice_questions",
        name="Practice Questions",
        description="Generate practice questions of mixed types with answers and explanations.",
        category="study_material",
        applicable_types=_SM,
        system_prompt="Generate practice questions (mix of multiple-choice and short-answer, marked via "
        "the `type` field) with correct answers and explanations, covering a spread of the material's "
        "sub-topics rather than clustering on one." + _GROUNDING_NOTE,
        result_kind=ResultKind.QUIZ,
        top_k=16,
        icon="pencil",
    ),
    PromptToolSpec(
        id="important_questions",
        name="Important Questions",
        description="Identify likely exam-important questions based on emphasis in the material.",
        category="study_material",
        applicable_types=_SM,
        system_prompt="Identify the questions most likely to be exam-important based on which topics the "
        "material emphasizes (repeated concepts, worked examples, explicit 'important' callouts if any). "
        "List each as a question only (no answers needed here)." + _GROUNDING_NOTE,
        result_kind=ResultKind.LIST,
        icon="star",
    ),
    PromptToolSpec(
        id="question_paper_generator",
        name="Question Paper Generator",
        description="Generate a full question paper from the uploaded material, configurable by total "
        "marks, difficulty, question count/types, and chapters, with an optional answer key.",
        category="study_material",
        applicable_types=_SM,
        system_prompt="Generate a complete exam question paper strictly grounded in the uploaded "
        "material. Respect the user configuration for total_marks, difficulty, num_questions, "
        "question_types, and chapters if given -- organize into sections by question type/marks "
        "(e.g. a 'Short Answer (2 marks each)' section, a 'Long Answer (5 marks each)' section). Make "
        "sure the sum of (questions x marks_per_question) across all sections equals total_marks as "
        "closely as possible. Only include an `answer_key` if the user configuration sets "
        "include_answer_key to true; otherwise omit it (return null)." + _GROUNDING_NOTE,
        result_kind=ResultKind.QUESTION_PAPER,
        task_type=TaskType.REPORT_GENERATION,
        top_k=20,
        max_tokens=4000,
        icon="file-question",
    ),
]

for _spec in _SPECS:
    register_tool(PromptTool(_spec))
