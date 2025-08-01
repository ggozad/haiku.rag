from io import BytesIO
from typing import ClassVar

import tiktoken
from docling.chunking import HybridChunker  # type: ignore
from docling.document_converter import DocumentConverter
from docling_core.types.io import DocumentStream

from haiku.rag.config import Config


class Chunker:
    """A class that chunks text into smaller pieces for embedding and retrieval.

    Uses docling's structure-aware chunking to create semantically meaningful chunks
    that respect document boundaries.

    Args:
        chunk_size: The maximum size of a chunk in tokens.
    """

    encoder: ClassVar[tiktoken.Encoding] = tiktoken.encoding_for_model("gpt-4o")

    def __init__(
        self,
        chunk_size: int = Config.CHUNK_SIZE,
    ):
        self.chunk_size = chunk_size
        self.docling_chunker = HybridChunker(max_tokens=chunk_size)  # type: ignore

    async def chunk(self, text: str) -> list[str]:
        """Split the text into chunks using docling's structure-aware chunking.

        Args:
            text: The text to be split into chunks.

        Returns:
            A list of text chunks with semantic boundaries.
        """
        if not text:
            return []

        # Convert to docling document
        bytes_io = BytesIO(text.encode("utf-8"))
        doc_stream = DocumentStream(name="text.md", stream=bytes_io)
        converter = DocumentConverter()
        result = converter.convert(doc_stream)
        doc = result.document

        # Chunk using docling's hybrid chunker
        docling_chunks = list(self.docling_chunker.chunk(doc))
        return [chunk.text for chunk in docling_chunks]


chunker = Chunker()
