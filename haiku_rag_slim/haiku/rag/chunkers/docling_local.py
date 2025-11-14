from typing import TYPE_CHECKING

from haiku.rag.chunkers.base import DocumentChunker
from haiku.rag.config import AppConfig, Config

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


class DoclingLocalChunker(DocumentChunker):
    """Local document chunker using docling's HybridChunker.

    Uses docling's structure-aware chunking to create semantically meaningful chunks
    that respect document boundaries. Chunking is performed locally using the
    HuggingFace tokenizer specified in configuration.

    Args:
        config: Application configuration.
    """

    def __init__(self, config: AppConfig = Config):
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
        from docling_core.transforms.chunker.tokenizer.huggingface import (
            HuggingFaceTokenizer,
        )
        from transformers import AutoTokenizer

        self.config = config
        self.chunk_size = config.processing.chunk_size
        self.tokenizer_name = config.processing.chunking_tokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        tokenizer = HuggingFaceTokenizer(
            tokenizer=hf_tokenizer, max_tokens=self.chunk_size
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
