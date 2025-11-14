from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


class DocumentChunker(ABC):
    """Abstract base class for document chunkers.

    Document chunkers split DoclingDocuments into smaller text chunks suitable
    for embedding and retrieval, respecting document structure and semantic boundaries.
    """

    @abstractmethod
    async def chunk(self, document: "DoclingDocument") -> list[str]:
        """Split a document into chunks.

        Args:
            document: The DoclingDocument to chunk.

        Returns:
            List of text chunks with semantic boundaries preserved.

        Raises:
            ValueError: If chunking fails.
        """
        pass
