from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from haiku.rag.client.scope import DatabaseScope


def run_chat(
    db_path: Path | None = None,
    read_only: bool = False,
    model: str | None = None,
    scope: "DatabaseScope | None" = None,
) -> None:
    """Run the chat TUI.

    Args:
        db_path: Path to the LanceDB database, when no scope is given.
        scope: The databases to cover, resolved by the caller.
        read_only: Whether to open the database in read-only mode.
        model: Model to use for the chat.
    """
    try:
        from haiku.rag.chat.app import ChatApp
    except ImportError as e:
        raise ImportError(
            "textual is not installed. Please install it with `pip install 'haiku.rag-slim[tui]'` or use the full haiku.rag package."
        ) from e

    from haiku.rag.capabilities.rag import create_capability
    from haiku.rag.config import get_config
    from haiku.rag.utils import get_model, parse_model_option

    config = get_config()
    if scope is None:
        from haiku.rag.client.scope import DatabaseScope

        scope = DatabaseScope.resolve(config, database_path=db_path)

    if model:
        config.qa.model = parse_model_option(model)

    # The app opens the scope and lends that client to the capability, which
    # reads what `--db PATH` or `--db-name NAME` selected.
    capability = create_capability(
        config=config,
        defer_loading=False,
        vision=config.qa.model.vision,
    )

    app = ChatApp(
        capability=capability,
        read_only=read_only,
        model=model or get_model(config.qa.model, config),
        scope=scope,
    )
    app.run()
