import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

from haiku.rag.store.compression import compress_docling, decompress_json

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


class Document(BaseModel):
    """
    Represents a document with an ID, content, and metadata.

    ``source`` names the database a document came from: the name from
    ``lancedb.databases`` or a path's stem, never a path or URI. Every document
    a database returns carries it; it is never persisted, and None only on a
    document built by hand.
    """

    id: str | None = None
    content: str
    uri: str | None = None
    title: str | None = None
    source: str | None = None
    metadata: dict = {}
    docling_document: bytes | None = Field(default=None, exclude=True)
    docling_version: str | None = Field(default=None, exclude=True)
    original: bytes | None = Field(default=None, exclude=True)
    original_sha256: str | None = Field(default=None, exclude=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("created_at", "updated_at")
    @classmethod
    def _to_utc(cls, value: datetime) -> datetime:
        return value if value.tzinfo else value.astimezone(UTC)

    def set_docling(
        self, docling_doc: "DoclingDocument", original: bytes | None = None
    ) -> None:
        """Serialize and store a DoclingDocument without its images.

        Sets docling_document (zstd-compressed) and docling_version.
        ``original`` is the file the structure was converted from: it replaces
        any stored one, and None clears it.
        """
        self.docling_document = compress_docling(docling_doc.model_dump(mode="json"))
        self.docling_version = docling_doc.version
        self.original = original
        self.original_sha256 = (
            hashlib.sha256(original).hexdigest() if original is not None else None
        )

    def get_docling_document(self) -> "DoclingDocument | None":
        """Parse and return the stored DoclingDocument, without its images.

        Returns:
            The parsed DoclingDocument, or None if not stored.
        """
        if self.docling_document is None:
            return None

        from docling_core.types.doc.document import DoclingDocument

        json_str = decompress_json(self.docling_document)
        return DoclingDocument.model_validate_json(json_str)
