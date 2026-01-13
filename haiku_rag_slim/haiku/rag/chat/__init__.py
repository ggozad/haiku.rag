from datetime import datetime
from pathlib import Path


def run_chat(
    db_path: Path | None = None,
    read_only: bool = False,
    before: datetime | None = None,
) -> None:
    """Run the chat TUI.

    Args:
        db_path: Path to the LanceDB database. If None, uses default from config.
        read_only: Whether to open the database in read-only mode.
        before: Query database as it existed before this datetime.
    """
    try:
        from haiku.rag.chat.app import ChatApp
    except ImportError as e:
        raise ImportError(
            "textual is not installed. Please install it with `pip install 'haiku.rag-slim[tui]'` or use the full haiku.rag package."
        ) from e

    from haiku.rag.config import get_config

    config = get_config()
    if db_path is None:
        db_path = config.storage.data_dir / "haiku.rag.lancedb"

    app = ChatApp(db_path, read_only=read_only, before=before)
    app.run()
