"""Document chunker abstraction for haiku.rag."""

from haiku.rag.chunkers.base import DocumentChunker
from haiku.rag.config import AppConfig, get_config

__all__ = ["DocumentChunker", "get_chunker"]


def get_chunker(config: AppConfig | None = None) -> DocumentChunker:
    """Get a document chunker instance based on configuration.

    Args:
        config: Configuration to use. Defaults to the current global config.

    Returns:
        DocumentChunker instance configured according to the config.

    Raises:
        ValueError: If the chunker provider is not recognized.
    """
    config = config if config is not None else get_config()
    if config.processing.chunker == "docling-local":
        from haiku.rag.chunkers.docling_local import DoclingLocalChunker

        return DoclingLocalChunker(config)

    if config.processing.chunker == "docling-serve":
        from haiku.rag.chunkers.docling_serve import DoclingServeChunker

        return DoclingServeChunker(config)

    raise ValueError(f"Unsupported chunker: {config.processing.chunker}")
