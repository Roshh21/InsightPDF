# backend/summarizer_agent.py
from langchain_core.prompts import ChatPromptTemplate
from .llm_provider import get_llm

BASE_SUMMARY_SYSTEM_PROMPT = """
You are an expert document summarizer.
Given a long document and its type, produce a concise, structured bullet-point summary.

Guidelines:
- Use headings and bullet points.
- Focus only on information present in the document.
- Do NOT hallucinate.
- Capture key entities, relationships, and important details.

Document-type specific structure:
- Research Paper: abstract, methods, dataset, results, limitations.
- Novel / Literature: plot, characters, conflicts, themes.
- Study Material / Notes: key concepts, formulas, definitions.
- Technical Documentation: overview, components, APIs, workflows.
- Business Report: insights, action items, risks, goals.
"""

# 1) First-pass summary
FIRST_PASS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", BASE_SUMMARY_SYSTEM_PROMPT),
        (
            "human",
            "Document type: {doc_type}\n\nDocument content:\n{content}\n\n"
            "Produce a first draft of the structured summary.",
        ),
    ]
)

# 2) Critic agent to assess quality
CRITIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a critical reviewer of summaries.
Given the document type, the original document text, and a summary, decide if the summary is:

- GOOD: structurally sound, captures the main sections, covers key details.
- BAD: missing important sections, too shallow, or skipping key characters/concepts.

Also list briefly what is missing if BAD.

Respond in JSON with keys:
- "grade": "GOOD" or "BAD"
- "missing": short text describing missing pieces (can be empty for GOOD)
""",
        ),
        (
            "human",
            "Document type:\n{doc_type}\n\nDocument content:\n{content}\n\n"
            "Summary:\n{summary}\n\n"
            "Your JSON response:",
        ),
    ]
)

# 3) Refiner agent to improve the summary
REFINER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a refinement agent.
Using the original document, its type, the first draft summary, and the critic's notes,
produce an IMPROVED structured summary that fills in missing pieces.

Requirements:
- Keep the structure (headings + bullet points).
- Add missing sections or details mentioned by the critic.
- Do not remove correct content from the first draft; extend and refine it.
""",
        ),
        (
            "human",
            "Document type:\n{doc_type}\n\nDocument content:\n{content}\n\n"
            "First draft summary:\n{summary}\n\n"
            "Critic notes (missing pieces):\n{missing}\n\n"
            "Produce the improved summary now:",
        ),
    ]
)

def _first_pass_summary(doc_text: str, doc_type: str) -> str:
    llm = get_llm()
    chain = FIRST_PASS_PROMPT | llm
    return chain.invoke({"doc_type": doc_type, "content": doc_text[:12000]})

def _critique_summary(doc_text: str, doc_type: str, summary: str) -> tuple[str, str]:
    llm = get_llm()
    chain = CRITIC_PROMPT | llm
    raw = chain.invoke(
        {"doc_type": doc_type, "content": doc_text[:8000], "summary": summary}
    )
    # very lightweight JSON-ish parsing – LLM output is small
    grade = "BAD"
    missing = ""
    if isinstance(raw, str):
        lower = raw.lower()
        if '"grade"' in lower:
            # crude but robust: look for GOOD/BAD
            if "good" in lower:
                grade = "GOOD"
            if "bad" in lower:
                grade = "BAD"
        else:
            # fallback: treat "good" anywhere as GOOD
            if "good" in lower and "bad" not in lower:
                grade = "GOOD"
        # attempt to extract "missing" roughly
        if '"missing"' in lower:
            missing = raw
    return grade, missing

def _refine_summary(doc_text: str, doc_type: str, summary: str, missing: str) -> str:
    llm = get_llm()
    chain = REFINER_PROMPT | llm
    return chain.invoke(
        {
            "doc_type": doc_type,
            "content": doc_text[:12000],
            "summary": summary,
            "missing": missing,
        }
    )

def summarize_document(doc_text: str, doc_type: str) -> str:
    """
    Agentic summarization:
    - First-pass summary
    - Critic grades it
    - If BAD, run a refinement pass using critic feedback
    """
    draft = _first_pass_summary(doc_text, doc_type)
    grade, missing = _critique_summary(doc_text, doc_type, draft)

    if grade == "GOOD":
        return draft

    improved = _refine_summary(doc_text, doc_type, draft, missing)
    return improved