import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING

from cachetools import LRUCache
from pydantic import BaseModel, Field

from haiku.rag.store.compression import decompress_json

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

logger = logging.getLogger(__name__)
perf_logger = logging.getLogger("haiku.rag.perf")

_docling_document_cache: LRUCache[str, "DoclingDocument"] = LRUCache(maxsize=100)


def configure_docling_cache(maxsize: int) -> None:
    """Resize the DoclingDocument LRU cache.

    Existing entries are preserved up to the new maxsize.
    """
    global _docling_document_cache
    if _docling_document_cache.maxsize == maxsize:
        return
    old = _docling_document_cache
    _docling_document_cache = LRUCache(maxsize=maxsize)
    # Copy existing entries (LRU order preserved by iteration)
    for key in old:
        _docling_document_cache[key] = old[key]
    perf_logger.debug("docling.cache_resized maxsize=%d", maxsize)


def _get_cached_docling_document(
    document_id: str, compressed_data: bytes
) -> "DoclingDocument":
    """Get or parse DoclingDocument with LRU caching by document ID."""
    if document_id in _docling_document_cache:
        perf_logger.debug("docling.cache_hit doc=%s", document_id[:8])
        return _docling_document_cache[document_id]

    from docling_core.types.doc.document import DoclingDocument

    perf_logger.debug(
        "docling.cache_miss doc=%s cache_size=%d/%d",
        document_id[:8],
        len(_docling_document_cache),
        _docling_document_cache.maxsize,
    )

    t0 = time.perf_counter()
    json_str = decompress_json(compressed_data)
    decompress_time = time.perf_counter() - t0
    perf_logger.debug(
        "docling.decompress doc=%s bytes=%d json_chars=%d took %.3fs",
        document_id[:8],
        len(compressed_data),
        len(json_str),
        decompress_time,
    )

    t0 = time.perf_counter()
    doc = DoclingDocument.model_validate_json(json_str)
    validate_time = time.perf_counter() - t0
    perf_logger.debug(
        "docling.model_validate doc=%s took %.3fs",
        document_id[:8],
        validate_time,
    )

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
    docling_document: bytes | None = Field(default=None, exclude=True)
    docling_version: str | None = Field(default=None, exclude=True)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def get_docling_document(self) -> "DoclingDocument | None":
        """Parse and return the stored DoclingDocument.

        Uses LRU cache (keyed by document ID) to avoid repeated parsing.

        Returns:
            The parsed DoclingDocument, or None if not stored or no ID.
        """
        if self.docling_document is None:
            return None

        # No caching for documents without ID
        if self.id is None:
            from docling_core.types.doc.document import DoclingDocument

            json_str = decompress_json(self.docling_document)
            return DoclingDocument.model_validate_json(json_str)

        return _get_cached_docling_document(self.id, self.docling_document)
