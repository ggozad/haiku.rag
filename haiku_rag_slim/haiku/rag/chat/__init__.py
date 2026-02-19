from datetime import datetime
from pathlib import Path


def run_chat(
    db_path: Path | None = None,
    read_only: bool = False,
    before: datetime | None = None,
    model: str | None = None,
) -> None:
    """Run the chat TUI.

    Args:
        db_path: Path to the LanceDB database. If None, uses default from config.
        read_only: Whether to open the database in read-only mode.
        before: Query database as it existed before this datetime.
        model: Model to use for the chat agent.
    """
    try:
        from haiku.rag.chat.app import ChatApp
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "textual is not installed. Please install it with `pip install 'haiku.rag-slim[tui]'` or use the full haiku.rag package."
        ) from e

    from haiku.rag.config import get_config
    from haiku.rag.skills.rag import create_skill

    config = get_config()
    if db_path is None:
        db_path = config.storage.data_dir / "haiku.rag.lancedb"

    skill = create_skill(db_path=db_path, config=config)

    app = ChatApp(
        db_path,
        skill=skill,
        read_only=read_only,
        before=before,
        model=model,
    )
    app.run()
