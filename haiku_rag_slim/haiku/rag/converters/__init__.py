"""Document converter abstraction for haiku.rag."""

from haiku.rag.config import AppConfig, Config
from haiku.rag.converters.base import DocumentConverter

__all__ = ["DocumentConverter", "get_converter"]


def get_converter(config: AppConfig = Config) -> DocumentConverter:
    """Get a document converter instance based on configuration.

    Args:
        config: Configuration to use. Defaults to global Config.

    Returns:
        DocumentConverter instance configured according to the config.

    Raises:
        ValueError: If the converter provider is not recognized.
    """
    if config.processing.converter == "docling-local":
        from haiku.rag.converters.docling_local import DoclingLocalConverter

        return DoclingLocalConverter(config)

    if config.processing.converter == "docling-serve":
        from haiku.rag.converters.docling_serve import DoclingServeConverter

        return DoclingServeConverter(config)

    raise ValueError(f"Unsupported converter provider: {config.processing.converter}")
