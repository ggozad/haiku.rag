import json
from datetime import datetime
from typing import TYPE_CHECKING

from cachetools import LRUCache
from pydantic import BaseModel, Field

from haiku.rag.store.compression import compress_docling_split, decompress_json

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument, PageItem


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
    do not contain page data.
    """
    if document_id in _docling_document_cache:
        return _docling_document_cache[document_id]

    doc = _validate_without_pages(compressed_data)
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
    docling_pages: bytes | None = Field(default=None, exclude=True)
    docling_version: str | None = Field(default=None, exclude=True)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def set_docling(self, docling_doc: "DoclingDocument") -> None:
        """Serialize and store a DoclingDocument, splitting structure and pages.

        Sets docling_document (zstd-compressed structure without pages),
        docling_pages (zstd-compressed page images), and docling_version.
        """
        structure, pages = compress_docling_split(docling_doc.model_dump_json())
        self.docling_document = structure
        self.docling_pages = pages
        self.docling_version = docling_doc.version

    def get_docling_document(self) -> "DoclingDocument | None":
        """Parse and return the stored DoclingDocument (without page images).

        Uses LRU cache (keyed by document ID) to avoid repeated parsing.

        Returns:
            The parsed DoclingDocument, or None if not stored.
        """
        if self.docling_document is None:
            return None

        # No caching for documents without ID
        if self.id is None:
            return _validate_without_pages(self.docling_document)

        return _get_cached_docling_document(self.id, self.docling_document)

    def get_page_images(self, page_numbers: list[int]) -> "dict[int, PageItem]":
        """Decompress and return page images for the requested page numbers.

        Loads only the docling_pages blob — does not need the structure.
        Validates only the requested pages through Pydantic (for pil_image property).

        Args:
            page_numbers: Page numbers to retrieve.

        Returns:
            Dict mapping page number to validated PageItem.
        """
        if self.docling_pages is None:
            return {}

        from docling_core.types.doc.document import PageItem

        pages_json = decompress_json(self.docling_pages)
        all_pages = json.loads(pages_json)

        result: dict[int, PageItem] = {}
        for page_no in page_numbers:
            page_data = all_pages.get(str(page_no))
            if page_data is not None:
                result[page_no] = PageItem.model_validate(page_data)
        return result
