import json
from datetime import datetime
from typing import TYPE_CHECKING

from cachetools import LRUCache
from pydantic import BaseModel, Field

from haiku.rag.store.compression import decompress_json

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


_docling_document_cache: LRUCache[str, "DoclingDocument"] = LRUCache(maxsize=100)


def _validate_without_pages(compressed_data: bytes) -> "DoclingDocument":
    """Decompress and validate DoclingDocument, stripping page images."""
    from docling_core.types.doc.document import DoclingDocument

    json_str = decompress_json(compressed_data)
    data = json.loads(json_str)
    data.pop("pages", None)
    return DoclingDocument.model_validate(data)


def _get_cached_docling_document(
    document_id: str, compressed_data: bytes
) -> "DoclingDocument":
    """Get or parse DoclingDocument with LRU caching by document ID.

    Strips page images before validation for performance — cached documents
    do not contain page data. Use _parse_full_docling_document for
    operations that need page images (e.g. visualize_chunk).
    """
    if document_id in _docling_document_cache:
        return _docling_document_cache[document_id]

    doc = _validate_without_pages(compressed_data)
    _docling_document_cache[document_id] = doc
    return doc


def _parse_full_docling_document(compressed_data: bytes) -> "DoclingDocument":
    """Parse DoclingDocument with full page data (no caching, no stripping)."""
    from docling_core.types.doc.document import DoclingDocument

    json_str = decompress_json(compressed_data)
    return DoclingDocument.model_validate_json(json_str)


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
    docling_document: bytes | None = Field(default=None, exclude=True)
    docling_version: str | None = Field(default=None, exclude=True)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def get_docling_document(
        self, *, include_pages: bool = False
    ) -> "DoclingDocument | None":
        """Parse and return the stored DoclingDocument.

        By default, strips page images before parsing for performance.
        Uses LRU cache (keyed by document ID) to avoid repeated parsing.

        Args:
            include_pages: If True, parse with full page data (slower,
                bypasses cache). Only needed for operations that access
                page images (e.g. visualize_chunk).

        Returns:
            The parsed DoclingDocument, or None if not stored or no ID.
        """
        if self.docling_document is None:
            return None

        if include_pages:
            return _parse_full_docling_document(self.docling_document)

        # No caching for documents without ID
        if self.id is None:
            return _validate_without_pages(self.docling_document)

        return _get_cached_docling_document(self.id, self.docling_document)
