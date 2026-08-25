from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from haiku.rag.client.scope import DatabaseScope


def run_chat(
    db_path: Path | None = None,
    read_only: bool = False,
    model: str | None = None,
    capabilities: list[str] | None = None,
    scope: "DatabaseScope | None" = None,
) -> None:
    """Run the chat TUI.

    Args:
        db_path: Path to the LanceDB database, when no scope is given.
        scope: The databases to cover, resolved by the caller.
        read_only: Whether to open the database in read-only mode.
        model: Model to use for the chat.
        capabilities: Capabilities to enable ("rag", "analysis"). Defaults to ["rag"].
    """
    try:
        from haiku.rag.chat.app import ChatApp
    except ImportError as e:
        raise ImportError(
            "textual is not installed. Please install it with `pip install 'haiku.rag-slim[tui]'` or use the full haiku.rag package."
        ) from e

    from haiku.rag.config import get_config
    from haiku.rag.utils import get_model, parse_model_option

    config = get_config()
    if scope is None:
        from haiku.rag.client.scope import DatabaseScope

        scope = DatabaseScope.resolve(config, database_path=db_path)

    if model:
        model_config = parse_model_option(model)
        config.qa.model = model_config
        config.analysis.model = model_config

    # The capabilities read the databases the scope covers, not whatever the
    # configuration happens to name: `--db PATH` would otherwise leave them on
    # the default database, and `--database NAME` on the whole set.
    if scope.covers_multiple:
        capability_config, capability_db_path = config, None
    else:
        capability_config, capability_db_path = scope.databases[0].connection(config)

    enabled = capabilities or ["rag"]
    capability_list = []
    defer_loading = len(enabled) > 1

    # One agent drives every attached capability, so a capability's
    # image-attachment gate must track that single model: analysis.model only
    # when analysis runs alone, otherwise qa.model. Passing it to every
    # capability keeps their vision flag aligned with the model actually running.
    if "rag" not in enabled and "analysis" in enabled:
        driving_model = config.analysis.model or config.qa.model
    else:
        driving_model = config.qa.model

    if "rag" in enabled:
        from haiku.rag.capabilities.rag import create_capability

        capability_list.append(
            create_capability(
                db_path=capability_db_path,
                config=capability_config,
                defer_loading=defer_loading,
                vision=driving_model.vision,
            )
        )

    if "analysis" in enabled:
        from haiku.rag.capabilities.analysis import create_capability

        capability_list.append(
            create_capability(
                db_path=capability_db_path,
                config=capability_config,
                defer_loading=defer_loading,
                vision=driving_model.vision,
            )
        )

    app = ChatApp(
        capabilities=capability_list,
        read_only=read_only,
        model=model or get_model(driving_model, config),
        scope=scope,
    )
    app.run()
