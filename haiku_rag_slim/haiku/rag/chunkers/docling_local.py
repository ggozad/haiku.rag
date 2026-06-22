import asyncio
from functools import cache
from typing import TYPE_CHECKING, cast

from haiku.rag.chunkers.base import DocumentChunker
from haiku.rag.config import AppConfig, Config
from haiku.rag.store.models.chunk import Chunk, ChunkMetadata

if TYPE_CHECKING:
    from docling_core.transforms.chunker.doc_chunk import DocMeta
    from docling_core.types.doc.document import DoclingDocument


@cache
def _get_tokenizer(name: str):
    # `AutoTokenizer.from_pretrained` triggers an HF Hub `model_info` request
    # per call. Batch ingest builds one chunker per document, so without this
    # cache HF rate-limits at 1000 requests / 5 minutes.
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


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

        self.config = config
        self.chunk_size = config.processing.chunk_size
        self.chunker_type = config.processing.chunker_type
        self.tokenizer_name = config.processing.chunking_tokenizer

        if self.chunker_type == "hybrid":
            hf_tokenizer = _get_tokenizer(self.tokenizer_name)
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

    def _chunk_sync(self, document: "DoclingDocument") -> list[Chunk]:
        """Synchronous chunking helper (CPU-bound, no I/O).

        Runs the underlying HybridChunker/HierarchicalChunker and extracts
        structured metadata from each DocChunk.
        """
        raw_chunks = list(self.chunker.chunk(document))
        result: list[Chunk] = []

        for chunk in raw_chunks:
            text = chunk.text

            # Extract metadata from DocChunk.meta (cast to DocMeta for type safety)
            doc_item_refs: list[str] = []
            labels: list[str] = []
            page_numbers: list[int] = []
            headings: list[str] | None = None

            meta = cast("DocMeta | None", chunk.meta)
            if meta and meta.doc_items:
                for doc_item in meta.doc_items:
                    # Get JSON pointer reference
                    if doc_item.self_ref:
                        doc_item_refs.append(doc_item.self_ref)
                    # Get label
                    if doc_item.label:
                        labels.append(doc_item.label)
                    # Get page numbers from provenance
                    if doc_item.prov:
                        for prov in doc_item.prov:
                            if (
                                prov.page_no is not None
                                and prov.page_no not in page_numbers
                            ):
                                page_numbers.append(prov.page_no)

            # Get headings from chunk metadata
            if meta and meta.headings:
                headings = list(meta.headings)

            chunk_metadata = ChunkMetadata(
                doc_item_refs=doc_item_refs,
                headings=headings,
                labels=labels,
                page_numbers=sorted(page_numbers),
            )
            result.append(
                Chunk(
                    content=text,
                    metadata=chunk_metadata.model_dump(),
                    order=len(result),
                )
            )

        return result

    async def chunk(self, document: "DoclingDocument") -> list[Chunk]:
        """Split the document into chunks with metadata.

        Extracts structured metadata from each DocChunk including:
        - doc_item_refs: JSON pointer references to DocItems (e.g., "#/texts/5")
        - headings: Section heading hierarchy
        - labels: Semantic labels for each doc_item (e.g., "paragraph", "table")
        - page_numbers: Page numbers where content appears

        Args:
            document: The DoclingDocument to be split into chunks.

        Returns:
            List of Chunk containing content and structured metadata.
        """
        if document is None:
            return []

        return await asyncio.to_thread(self._chunk_sync, document)
