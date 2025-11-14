from typing import TYPE_CHECKING

from haiku.rag.chunkers.base import DocumentChunker
from haiku.rag.config import AppConfig

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


class DoclingServeChunker(DocumentChunker):
    """Remote document chunker using docling-serve API.

    Placeholder - will be implemented in a future commit.
    """

    def __init__(self, config: AppConfig):
        raise NotImplementedError("DoclingServeChunker not yet implemented")

    async def chunk(self, document: "DoclingDocument") -> list[str]:
        """Split the document into chunks via docling-serve.

        Args:
            document: The DoclingDocument to be split into chunks.

        Returns:
            A list of text chunks with semantic boundaries.

        Raises:
            NotImplementedError: This chunker is not yet implemented.
        """
        raise NotImplementedError("DoclingServeChunker not yet implemented")
