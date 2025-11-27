from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from haiku.rag.store.models.chunk import ChunkMetadata

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


class ChunkWithMetadata:
    """A chunk with its text content and structured metadata."""

    def __init__(self, text: str, metadata: ChunkMetadata):
        self.text = text
        self.metadata = metadata


class DocumentChunker(ABC):
    """Abstract base class for document chunkers.

    Document chunkers split DoclingDocuments into smaller text chunks suitable
    for embedding and retrieval, respecting document structure and semantic boundaries.
    """

    @abstractmethod
    async def chunk(self, document: "DoclingDocument") -> list[ChunkWithMetadata]:
        """Split a document into chunks with metadata.

        Args:
            document: The DoclingDocument to chunk.

        Returns:
            List of ChunkWithMetadata containing text and structured metadata
            (doc_item_refs, headings, labels, page_numbers).

        Raises:
            ValueError: If chunking fails.
        """
        pass
