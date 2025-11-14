from typing import TYPE_CHECKING

from haiku.rag.config import Config

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


class Chunker:
    """A class that chunks text into smaller pieces for embedding and retrieval.

    Uses docling's structure-aware chunking to create semantically meaningful chunks
    that respect document boundaries.

    Args:
        chunk_size: The maximum size of a chunk in tokens.
        tokenizer_name: HuggingFace model name for tokenization.
    """

    def __init__(
        self,
        chunk_size: int = Config.processing.chunk_size,
        tokenizer_name: str = Config.processing.chunking_tokenizer,
    ):
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
        from docling_core.transforms.chunker.tokenizer.huggingface import (
            HuggingFaceTokenizer,
        )
        from transformers import AutoTokenizer

        self.chunk_size = chunk_size
        self.tokenizer_name = tokenizer_name

        hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer, max_tokens=chunk_size)

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
