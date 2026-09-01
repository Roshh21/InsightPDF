"""Literature / novel mode tools (spec section 7), with spoiler control.

Spoiler level is passed as `params.spoiler_level` (one of "none",
"chapter", "full") by the frontend's Spoiler Level selector and rendered
into the prompt generically by PromptTool -- every tool's system prompt
below explicitly instructs the model how to respect it.
"""
from __future__ import annotations

from app.core.enums import DocumentType, ResultKind, TaskType
from app.tools.prompt_tool import PromptTool, PromptToolSpec
from app.tools.registry import register_tool

_LIT = [DocumentType.LITERATURE]
_LIT_AND_STUDY = [DocumentType.LITERATURE, DocumentType.STUDY_MATERIAL]

_SPOILER_RULES = (
    "\n\nSPOILER CONTROL: honor `spoiler_level` from the user configuration if present:\n"
    "- 'none': discuss only setup, premise, tone, and characters/events up to roughly the first act. "
    "Never reveal major plot twists, the ending, or character fates.\n"
    "- 'chapter': you may discuss events up through the chapter/point the user specifies in their "
    "configuration or question, but not beyond it.\n"
    "- 'full': you may discuss the entire work, including the ending, freely.\n"
    "If no spoiler_level is given, default to 'chapter' behavior using whatever portion of the evidence "
    "is provided (do not reach beyond the retrieved evidence to guess later plot points)."
)

_SPECS = [
    PromptToolSpec(
        id="spoiler_free_summary",
        name="Spoiler-Free Summary",
        description="A summary of the premise, setting, and tone with no plot spoilers.",
        category="literature",
        applicable_types=_LIT,
        system_prompt="Write a spoiler-free summary: premise, setting, tone/genre, and main characters "
        "as introduced. Do not reveal plot developments beyond the opening." + _SPOILER_RULES,
        icon="book-open",
    ),
    PromptToolSpec(
        id="full_summary",
        name="Full Summary",
        description="A complete summary of the work, respecting the selected spoiler level.",
        category="literature",
        applicable_types=_LIT,
        system_prompt="Write a comprehensive summary of the work's plot and arc." + _SPOILER_RULES,
        top_k=18,
        icon="book-marked",
    ),
    PromptToolSpec(
        id="chapter_summary",
        name="Chapter Summary",
        description="Summarize a specific chapter or section (specify via the query/configuration).",
        category="literature",
        applicable_types=_LIT_AND_STUDY,
        system_prompt="Summarize the specific chapter/section requested by the user (see the "
        "`chapter` configuration field if present). If none is specified, summarize the chapter(s) "
        "best represented in the evidence. For textbook/study material (no narrative spoilers apply), "
        "just summarize clearly and skip the spoiler rules below." + _SPOILER_RULES,
        icon="bookmark",
    ),
    PromptToolSpec(
        id="character_analysis",
        name="Character Analysis",
        description="Analyze the main character(s): traits, motivations, arc, and role in the story.",
        category="literature",
        applicable_types=_LIT,
        system_prompt="Analyze the character(s) requested (or the most central characters if none "
        "specified): personality traits, motivations, development/arc, and role in the narrative."
        + _SPOILER_RULES,
        icon="user",
    ),
    PromptToolSpec(
        id="character_relationships",
        name="Character Relationships",
        description="Map out how the main characters relate to and affect one another.",
        category="literature",
        applicable_types=_LIT,
        system_prompt="Describe the relationships between the main characters -- alliances, conflicts, "
        "family/romantic ties, and how these relationships evolve." + _SPOILER_RULES,
        icon="users",
    ),
    PromptToolSpec(
        id="theme_analysis",
        name="Theme Analysis",
        description="Identify and discuss the major themes of the work.",
        category="literature",
        applicable_types=_LIT,
        system_prompt="Identify the major themes and discuss how the text develops each one, with "
        "supporting evidence." + _SPOILER_RULES,
        icon="palette",
    ),
    PromptToolSpec(
        id="plot_analysis",
        name="Plot Analysis",
        description="Break down the plot structure: exposition, rising action, climax, resolution.",
        category="literature",
        applicable_types=_LIT,
        system_prompt="Analyze the plot structure available in the evidence (exposition, rising action, "
        "climax, falling action, resolution as applicable) and how tension/stakes build."
        + _SPOILER_RULES,
        icon="waypoints",
    ),
    PromptToolSpec(
        id="important_events",
        name="Important Events",
        description="List the pivotal events of the story in order.",
        category="literature",
        applicable_types=_LIT,
        system_prompt="List the pivotal events evident in the evidence, in chronological order, with a "
        "one-line note on why each matters." + _SPOILER_RULES,
        icon="calendar-clock",
    ),
    PromptToolSpec(
        id="motifs_symbols",
        name="Motifs / Symbols",
        description="Identify recurring motifs and symbols and what they represent.",
        category="literature",
        applicable_types=_LIT,
        system_prompt="Identify recurring motifs/symbols/imagery and discuss their likely significance, "
        "grounded in specific instances from the evidence." + _SPOILER_RULES,
        icon="sparkle",
    ),
    PromptToolSpec(
        id="review",
        name="Review",
        description="Write a balanced critical review: strengths, weaknesses, and overall impression.",
        category="literature",
        applicable_types=_LIT,
        system_prompt="Write a balanced critical review covering strengths (prose, pacing, character "
        "work, originality) and weaknesses, based on the evidence available. This is critical "
        "commentary, not a plot summary." + _SPOILER_RULES,
        result_kind=ResultKind.REPORT,
        task_type=TaskType.GENERATION_COMPLEX,
        icon="star",
    ),
]

for _spec in _SPECS:
    register_tool(PromptTool(_spec))
