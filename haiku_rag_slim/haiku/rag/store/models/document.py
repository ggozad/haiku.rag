import json
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from haiku.rag.store.compression import compress_docling_split, decompress_json

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument, PageItem


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

        Returns:
            The parsed DoclingDocument, or None if not stored.
        """
        if self.docling_document is None:
            return None

        from docling_core.types.doc.document import DoclingDocument

        json_str = decompress_json(self.docling_document)
        return DoclingDocument.model_validate_json(json_str)

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
