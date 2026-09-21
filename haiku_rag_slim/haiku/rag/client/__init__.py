from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from haiku.rag.client.client import HaikuRAG, RebuildMode, all_found
    from haiku.rag.client.documents import DocumentImport
    from haiku.rag.client.scope import DatabaseScope

_EXPORTS = {
    "DatabaseScope": "haiku.rag.client.scope",
    "DocumentImport": "haiku.rag.client.documents",
    "HaikuRAG": "haiku.rag.client.client",
    "RebuildMode": "haiku.rag.client.client",
    "all_found": "haiku.rag.client.client",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Each export on first use only: reaching a sibling module (`scope`,
    `exceptions`) runs this package first, and must not cost lancedb and
    pydantic_ai."""
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
