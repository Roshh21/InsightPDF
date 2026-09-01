from __future__ import annotations

import re

from app.classification.profiles import DocumentSection
from app.ingestion.pdf_parser import ParsedDocument

# Heuristic heading patterns -- deliberately cheap (no LLM call) since this
# runs on every page of every uploaded document.
_NUMBERED_HEADING_RE = re.compile(r"^\s*(\d{1,2}(\.\d{1,2})*)[.)]?\s+([A-Z][A-Za-z0-9 ,:'&/-]{2,80})\s*$")
_CHAPTER_RE = re.compile(r"^\s*(chapter|section|part)\s+([0-9ivxlc]+)\b[:.\-]?\s*(.*)$", re.IGNORECASE)
_COMMON_HEADINGS = {
    "abstract", "introduction", "related work", "background", "methodology", "methods",
    "materials and methods", "experiments", "experimental setup", "results", "discussion",
    "conclusion", "conclusions", "limitations", "future work", "references", "acknowledgments",
    "executive summary", "overview", "architecture", "api reference", "requirements",
    "risk factors", "financial highlights", "appendix",
}
_ALL_CAPS_RE = re.compile(r"^[A-Z][A-Z0-9 &/\-]{3,60}$")


def _looks_like_heading(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return None

    m = _NUMBERED_HEADING_RE.match(stripped)
    if m:
        return m.group(3).strip()

    m = _CHAPTER_RE.match(stripped)
    if m:
        rest = m.group(3).strip(" :-")
        label = f"{m.group(1).title()} {m.group(2)}"
        return f"{label} - {rest}" if rest else label

    lowered = stripped.lower().strip(" :.")
    if lowered in _COMMON_HEADINGS:
        return stripped.strip(" :.")

    if _ALL_CAPS_RE.match(stripped) and len(stripped.split()) <= 8:
        return stripped.title()

    return None


def detect_sections(parsed: ParsedDocument, max_sections: int = 60) -> list[DocumentSection]:
    """Cheap, deterministic structural pass. This intentionally does not use
    the LLM -- it's meant to be a fast, always-available skeleton that the
    classifier's LLM-derived sections (if any) can be reconciled with."""
    sections: list[DocumentSection] = []
    for page in parsed.pages:
        for line in page.text.splitlines()[:40]:  # headings are almost always near the top of a text block
            title = _looks_like_heading(line)
            if title:
                if sections and sections[-1].title.lower() == title.lower():
                    continue
                if sections:
                    sections[-1].page_end = page.page_number - 1 if page.page_number > sections[-1].page_start else sections[-1].page_start
                sections.append(DocumentSection(title=title, page_start=page.page_number, level=1))
                if len(sections) >= max_sections:
                    return sections
    if sections:
        sections[-1].page_end = parsed.page_count
    return sections
