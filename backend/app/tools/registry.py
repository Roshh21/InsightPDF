from __future__ import annotations

from typing import Any

from app.core.enums import DocumentType
from app.tools.base import BaseTool

_REGISTRY: dict[str, BaseTool] = {}


def register_tool(tool: BaseTool) -> BaseTool:
    if tool.id in _REGISTRY:
        raise ValueError(f"Tool id already registered: {tool.id}")
    _REGISTRY[tool.id] = tool
    return tool


def get_tool(tool_id: str) -> BaseTool | None:
    return _REGISTRY.get(tool_id)


def all_tools() -> list[BaseTool]:
    return list(_REGISTRY.values())


def catalog(doc_type: DocumentType | None = None) -> list[dict[str, Any]]:
    """Full tool catalog, optionally filtered to tools applicable to a
    document type. This backs GET /tools/catalog -- the frontend filters
    further by a document's own `capabilities` list, but this endpoint is
    what lets it know each tool's display name/description/icon/shape
    without hard-coding any of that client-side."""
    tools = all_tools()
    if doc_type is not None:
        tools = [t for t in tools if t.applicable_types is None or doc_type in t.applicable_types]
    return [t.catalog_entry() for t in tools]


def ensure_loaded() -> None:
    """Import all catalog modules exactly once, registering their tools.
    Safe to call repeatedly (idempotent) since each module guards itself."""
    if _REGISTRY:
        return
    from app.tools import (  # noqa: F401
        catalog_business,
        catalog_literature,
        catalog_research,
        catalog_study,
        catalog_techdoc,
        catalog_universal,
        special_tools,
        web_research,
    )
