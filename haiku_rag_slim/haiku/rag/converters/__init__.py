"""Document converter abstraction for haiku.rag."""

from haiku.rag.config import AppConfig, get_config
from haiku.rag.converters.base import DocumentConverter

__all__ = ["DocumentConverter", "get_converter"]


def get_converter(config: AppConfig | None = None) -> DocumentConverter:
    """Get a document converter instance based on configuration.

    Args:
        config: Configuration to use. Defaults to the current global config.

    Returns:
        DocumentConverter instance configured according to the config.

    Raises:
        ValueError: If the converter provider is not recognized.
    """
    config = config if config is not None else get_config()
    if config.processing.converter == "docling-local":
        from haiku.rag.converters.docling_local import DoclingLocalConverter

        return DoclingLocalConverter(config)

    if config.processing.converter == "docling-serve":
        from haiku.rag.converters.docling_serve import DoclingServeConverter

        return DoclingServeConverter(config)

    raise ValueError(f"Unsupported converter provider: {config.processing.converter}")
