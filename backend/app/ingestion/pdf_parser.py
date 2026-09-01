from __future__ import annotations

from dataclasses import dataclass

from app.core.exceptions import DocumentValidationError, EmptyDocumentError, ScannedDocumentError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Below this many extracted characters per page (on average), we treat the
# document as likely scanned/image-only rather than failing silently on an
# empty summary later.
MIN_AVG_CHARS_PER_PAGE = 20


@dataclass
class ParsedPage:
    page_number: int  # 1-indexed
    text: str


@dataclass
class ParsedDocument:
    pages: list[ParsedPage]
    page_count: int

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


def parse_pdf(file_path: str) -> ParsedDocument:
    """Extract per-page text. Raises DocumentValidationError subclasses for
    invalid/empty/scanned PDFs instead of failing deep inside the pipeline."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise DocumentValidationError("pypdf is not installed") from exc

    try:
        reader = PdfReader(file_path)
    except Exception as exc:
        raise DocumentValidationError(f"Could not open file as a PDF: {exc}") from exc

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise DocumentValidationError("PDF is password-protected and could not be opened") from exc

    n_pages = len(reader.pages)
    if n_pages == 0:
        raise DocumentValidationError("PDF has no pages")

    pages: list[ParsedPage] = []
    total_chars = 0
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Failed to extract text from page %s: %s", i, exc)
            text = ""
        text = text.strip()
        total_chars += len(text)
        pages.append(ParsedPage(page_number=i, text=text))

    avg_chars = total_chars / max(n_pages, 1)
    if total_chars == 0:
        raise EmptyDocumentError()
    if avg_chars < MIN_AVG_CHARS_PER_PAGE:
        raise ScannedDocumentError(
            f"Average {avg_chars:.1f} extractable chars/page across {n_pages} pages"
        )

    return ParsedDocument(pages=pages, page_count=n_pages)


def get_page_text(file_path: str, page_number: int) -> str:
    """On-demand single-page fetch, used by the get_page agent tool.
    Cheaper than re-running full validation for a single lookup."""
    from pypdf import PdfReader

    try:
        reader = PdfReader(file_path)
        if reader.is_encrypted:
            reader.decrypt("")
        if page_number < 1 or page_number > len(reader.pages):
            raise ValueError(f"Page {page_number} out of range (document has {len(reader.pages)} pages)")
        return (reader.pages[page_number - 1].extract_text() or "").strip()
    except ValueError:
        raise
    except Exception as exc:
        raise DocumentValidationError(f"Could not read page {page_number}: {exc}") from exc
