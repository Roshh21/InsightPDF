from __future__ import annotations

from app.classification.profiles import TableInfo
from app.core.logging import get_logger

logger = get_logger(__name__)

MAX_PAGES_TO_SCAN_FOR_TABLES = 60


def detect_tables(file_path: str, max_pages: int = MAX_PAGES_TO_SCAN_FOR_TABLES) -> list[TableInfo]:
    """Lightweight table presence detection for the document profile
    (used to decide whether to surface table-aware tooling / mention tables
    in extraction results). Full table content extraction happens on-demand
    via `extract_table_from_page` so ingestion stays fast."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed, skipping table detection")
        return []

    tables: list[TableInfo] = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages], start=1):
                try:
                    found = page.find_tables()
                except Exception:
                    found = []
                for t in found:
                    rows = t.extract()
                    if not rows or len(rows) < 2:
                        continue
                    n_cols = max((len(r) for r in rows), default=0)
                    tables.append(TableInfo(page=i, n_rows=len(rows), n_cols=n_cols))
    except Exception as exc:
        logger.warning("Table detection failed for %s: %s", file_path, exc)
    return tables


def extract_table_from_page(file_path: str, page_number: int, table_index: int = 0) -> list[list[str]]:
    """Extract the raw cell grid for a specific table on a specific
    (1-indexed) page -- used by the `extract_table` agent tool."""
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("pdfplumber is not installed") from exc

    with pdfplumber.open(file_path) as pdf:
        if page_number < 1 or page_number > len(pdf.pages):
            raise ValueError(f"Page {page_number} out of range (document has {len(pdf.pages)} pages)")
        page = pdf.pages[page_number - 1]
        found = page.find_tables()
        if not found:
            return []
        if table_index >= len(found):
            table_index = 0
        rows = found[table_index].extract()
        return [[cell if cell is not None else "" for cell in row] for row in rows]
