try:
    from haiku.rag.inspector.app import run_inspector
except ImportError as e:
    raise ImportError(
        "textual is not installed. Please install it with `pip install 'haiku.rag-slim[inspector]'` or use the full haiku.rag package."
    ) from e

__all__ = ["run_inspector"]
