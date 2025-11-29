from datetime import datetime
from typing import TYPE_CHECKING

from cachetools import LRUCache
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


_docling_document_cache: LRUCache[str, "DoclingDocument"] = LRUCache(maxsize=100)


def _get_cached_docling_document(document_id: str, json_str: str) -> "DoclingDocument":
    """Get or parse DoclingDocument with LRU caching by document ID."""
    if document_id in _docling_document_cache:
        return _docling_document_cache[document_id]

    from docling_core.types.doc.document import DoclingDocument

    doc = DoclingDocument.model_validate_json(json_str)
    _docling_document_cache[document_id] = doc
    return doc


def invalidate_docling_document_cache(document_id: str) -> None:
    """Remove a document from the DoclingDocument cache."""
    _docling_document_cache.pop(document_id, None)


class Document(BaseModel):
    """
    Represents a document with an ID, content, and metadata.
    """

    id: str | None = None
    content: str
    uri: str | None = None
    title: str | None = None
    metadata: dict = {}
    docling_document_json: str | None = None
    docling_version: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def get_docling_document(self) -> "DoclingDocument | None":
        """Parse and return the stored DoclingDocument.

        Uses LRU cache (keyed by document ID) to avoid repeated parsing.

        Returns:
            The parsed DoclingDocument, or None if not stored or no ID.
        """
        if self.docling_document_json is None:
            return None

        # No caching for documents without ID
        if self.id is None:
            from docling_core.types.doc.document import DoclingDocument

            return DoclingDocument.model_validate_json(self.docling_document_json)

        return _get_cached_docling_document(self.id, self.docling_document_json)
