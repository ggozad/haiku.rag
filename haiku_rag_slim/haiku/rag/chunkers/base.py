from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.store.models.chunk import Chunk


class DocumentChunker(ABC):
    """Abstract base class for document chunkers.

    Document chunkers split DoclingDocuments into smaller text chunks suitable
    for embedding and retrieval, respecting document structure and semantic boundaries.
    """

    @abstractmethod
    async def chunk(self, document: "DoclingDocument") -> list["Chunk"]:
        """Split a document into chunks with metadata.

        Args:
            document: The DoclingDocument to chunk.

        Returns:
            List of Chunk with content and structured metadata in the metadata dict
            (doc_item_refs, headings, labels, page_numbers).

        Raises:
            ValueError: If chunking fails.
        """
        pass
