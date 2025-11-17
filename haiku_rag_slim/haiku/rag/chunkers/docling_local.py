from typing import TYPE_CHECKING

from haiku.rag.chunkers.base import DocumentChunker
from haiku.rag.config import AppConfig, Config

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


def _create_markdown_serializer_provider(use_markdown_tables: bool = True):
    """Create a markdown serializer provider with configurable table rendering.

    This function creates a custom serializer provider that extends ChunkingSerializerProvider
    from docling-core. It's implemented as a factory function to avoid importing
    docling-core at module level.

    Args:
        use_markdown_tables: If True, use MarkdownTableSerializer for rendering tables as
            markdown. If False, use default TripletTableSerializer for narrative format.
    """
    from docling_core.transforms.chunker.hierarchical_chunker import (
        ChunkingDocSerializer,
        ChunkingSerializerProvider,
    )
    from docling_core.transforms.serializer.markdown import MarkdownTableSerializer

    class MDTableSerializerProvider(ChunkingSerializerProvider):
        """Serializer provider for markdown table output."""

        def __init__(self, use_markdown_tables: bool = True):
            self.use_markdown_tables = use_markdown_tables

        def get_serializer(self, doc):
            if self.use_markdown_tables:
                return ChunkingDocSerializer(
                    doc=doc,
                    table_serializer=MarkdownTableSerializer(),
                )
            else:
                # Use default ChunkingDocSerializer (TripletTableSerializer)
                return ChunkingDocSerializer(doc=doc)

    return MDTableSerializerProvider(use_markdown_tables=use_markdown_tables)


class DoclingLocalChunker(DocumentChunker):
    """Local document chunker using docling's chunkers.

    Supports both hybrid (structure-aware) and hierarchical chunking strategies.
    Chunking is performed locally using the HuggingFace tokenizer specified in
    configuration.

    Args:
        config: Application configuration.
    """

    def __init__(self, config: AppConfig = Config):
        from docling_core.transforms.chunker.hierarchical_chunker import (
            HierarchicalChunker,
        )
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
        from docling_core.transforms.chunker.tokenizer.huggingface import (
            HuggingFaceTokenizer,
        )
        from transformers import AutoTokenizer

        self.config = config
        self.chunk_size = config.processing.chunk_size
        self.chunker_type = config.processing.chunker_type
        self.tokenizer_name = config.processing.chunking_tokenizer

        if self.chunker_type == "hybrid":
            hf_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            tokenizer = HuggingFaceTokenizer(
                tokenizer=hf_tokenizer, max_tokens=self.chunk_size
            )
            serializer_provider = _create_markdown_serializer_provider(
                use_markdown_tables=config.processing.chunking_use_markdown_tables
            )
            self.chunker = HybridChunker(
                tokenizer=tokenizer,
                merge_peers=config.processing.chunking_merge_peers,
                serializer_provider=serializer_provider,
            )
        elif self.chunker_type == "hierarchical":
            serializer_provider = _create_markdown_serializer_provider(
                use_markdown_tables=config.processing.chunking_use_markdown_tables
            )
            self.chunker = HierarchicalChunker(serializer_provider=serializer_provider)
        else:
            raise ValueError(
                f"Unsupported chunker_type: {self.chunker_type}. "
                "Must be 'hybrid' or 'hierarchical'."
            )

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
