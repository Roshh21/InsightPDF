"""Universal tools -- available regardless of document type (spec section 5).

`chat` itself is not registered here: it's the always-on conversational
interface (see app/agents), not a discrete one-shot tool execution.
"""
from __future__ import annotations

from app.core.enums import ResultKind, TaskType
from app.tools.prompt_tool import PromptTool, PromptToolSpec
from app.tools.registry import register_tool

register_tool(
    PromptTool(
        PromptToolSpec(
            id="summarize",
            name="Summarize",
            description="Produce a clear, well-structured summary of the document covering its main "
            "purpose, key points, and conclusions.",
            category="universal",
            system_prompt="You write clear, accurate, well-organized document summaries for a "
            "document intelligence platform. Prefer plain language. Organize into logical sections "
            "(e.g. Overview, Key Points, Conclusion) rather than one big paragraph.",
            result_kind=ResultKind.SECTIONS,
            task_type=TaskType.GENERATION_SIMPLE,
            top_k=14,
            icon="file-text",
        )
    )
)

register_tool(
    PromptTool(
        PromptToolSpec(
            id="ask_question",
            name="Ask a Question",
            description="Answer a specific user question about the document, grounded in evidence.",
            category="universal",
            system_prompt="Answer the user's question directly and precisely using only the "
            "provided evidence. If evidence is thin, say what is and isn't supported.",
            result_kind=ResultKind.TEXT,
            task_type=TaskType.GENERATION_SIMPLE,
            top_k=10,
            icon="message-circle-question",
        )
    )
)

register_tool(
    PromptTool(
        PromptToolSpec(
            id="explain",
            name="Explain",
            description="Explain a specific passage, section, or concept from the document in simpler terms.",
            category="universal",
            system_prompt="Explain the requested passage/concept in simple, accessible language, "
            "as if to someone encountering it for the first time. Use short sentences and, where "
            "helpful, a brief analogy. Do not oversimplify to the point of being inaccurate.",
            result_kind=ResultKind.TEXT,
            task_type=TaskType.GENERATION_SIMPLE,
            top_k=8,
            icon="lightbulb",
        )
    )
)

register_tool(
    PromptTool(
        PromptToolSpec(
            id="extract_key_info",
            name="Extract Key Information",
            description="Pull out the most important facts, figures, and statements from the document "
            "as a clean bulleted list.",
            category="universal",
            system_prompt="Extract the most important, concrete pieces of information (facts, figures, "
            "definitions, claims, dates) as a list of short, self-contained bullet points. Avoid vague "
            "generalities -- prefer specific, checkable statements.",
            result_kind=ResultKind.LIST,
            task_type=TaskType.GENERATION_SIMPLE,
            top_k=14,
            icon="list-checks",
        )
    )
)

register_tool(
    PromptTool(
        PromptToolSpec(
            id="find_important_sections",
            name="Find Important Sections",
            description="Identify and rank the most important sections or passages of the document, "
            "with a brief note on why each matters.",
            category="universal",
            system_prompt="Identify the passages/sections that carry the most important content or "
            "signal for a reader trying to quickly understand this document. For each, give a one-line "
            "reason it's important.",
            result_kind=ResultKind.SECTIONS,
            task_type=TaskType.GENERATION_SIMPLE,
            top_k=14,
            icon="target",
        )
    )
)

register_tool(
    PromptTool(
        PromptToolSpec(
            id="compare_documents",
            name="Compare Documents",
            description="Compare the selected documents side by side across shared attributes relevant "
            "to their content.",
            category="universal",
            system_prompt="Compare the selected documents. Choose a small set of attributes that are "
            "genuinely comparable across ALL of them (e.g. topic, scope, key claims, conclusions -- pick "
            "whatever is most meaningful for these specific documents), then fill in a value per document "
            "per attribute, grounded in evidence. Close with a short narrative on the most notable "
            "similarities/differences.",
            result_kind=ResultKind.COMPARISON,
            task_type=TaskType.COMPARISON,
            requires_multi_document=True,
            top_k=16,
            icon="columns-3",
        )
    )
)

register_tool(
    PromptTool(
        PromptToolSpec(
            id="generate_quiz",
            name="Generate Quiz",
            description="Generate a short multiple-choice quiz testing understanding of the document's "
            "content.",
            category="universal",
            system_prompt="Write a quiz of clear, unambiguous multiple-choice questions (4 options each, "
            "one correct) that test genuine understanding of the material -- not trivial word-matching. "
            "Include a short explanation for the correct answer for each question. Default to 5 questions "
            "unless the user configuration specifies a different count.",
            result_kind=ResultKind.QUIZ,
            task_type=TaskType.GENERATION_SIMPLE,
            top_k=14,
            icon="help-circle",
        )
    )
)
