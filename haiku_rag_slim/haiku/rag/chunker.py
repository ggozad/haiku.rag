from typing import TYPE_CHECKING, ClassVar

import tiktoken

from haiku.rag.config import Config

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


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
        chunk_size: int = Config.processing.chunk_size,
    ):
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
        from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

        self.chunk_size = chunk_size
        tokenizer = OpenAITokenizer(
            tokenizer=tiktoken.encoding_for_model("gpt-4o"), max_tokens=chunk_size
        )

        self.chunker = HybridChunker(tokenizer=tokenizer)

    async def chunk(self, document: "DoclingDocument") -> list[str]:
        """Split the document into chunks using docling's structure-aware chunking.

        Args:
            document: The DoclingDocument to be split into chunks.

        Returns:
            A list of text chunks with semantic boundaries.
        """
        if document is None:
            return []

        # Chunk using docling's hybrid chunker
        chunks = list(self.chunker.chunk(document))
        return [self.chunker.contextualize(chunk) for chunk in chunks]


chunker = Chunker()