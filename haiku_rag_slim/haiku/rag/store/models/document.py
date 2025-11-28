from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


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

        Returns:
            The parsed DoclingDocument, or None if not stored.
        """
        if self.docling_document_json is None:
            return None

        from docling_core.types.doc.document import DoclingDocument

        return DoclingDocument.model_validate_json(self.docling_document_json)
