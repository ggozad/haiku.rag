import asyncio
import json
import sys
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from dotenv import find_dotenv, load_dotenv

# Load environment variables from .env file for API keys and service URLs.
# Env loading needs to be before config import; usecwd=True searches from cwd, not this .py file's location
load_dotenv(find_dotenv(usecwd=True))

from haiku.rag.config import (  # noqa: E402
    AppConfig,
    find_config_file,
    get_config,
    load_yaml_config,
    set_config,
)
from haiku.rag.logging import configure_cli_logging  # noqa: E402
from haiku.rag.store.exceptions import (  # noqa: E402
    AmbiguousDatabaseError,
    MigrationRequiredError,
    ReadOnlyError,
    SourceUnavailableError,
)
from haiku.rag.store.models.chunk import SearchType  # noqa: E402
from haiku.rag.utils import is_up_to_date  # noqa: E402

if TYPE_CHECKING:
    from haiku.rag.app import HaikuRAGApp
    from haiku.rag.config.models import AppConfig

_cli = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


def cli():
    # Imported here rather than at module scope: the settings module pulls in
    # lancedb, and the CLI's startup must not pay for it.
    from haiku.rag.store.repositories.settings import ConfigMismatchError

    try:
        _cli()
    except (
        AmbiguousDatabaseError,
        ConfigMismatchError,
        MigrationRequiredError,
        ReadOnlyError,
        SourceUnavailableError,
    ) as e:
        typer.echo(f"Error: {e}", err=True)
        sys.exit(1)


# Module-level flags set by callback
_read_only: bool = False
_database: str | None = None
_database_path: Path | None = None


def create_app(db: Path | None = None, *, federated: bool = False) -> "HaikuRAGApp":
    """Create HaikuRAGApp with loaded config and resolved database path.

    Args:
        db: Optional database path. If None, uses `--database`, then the path
            from config.
        federated: Whether this command works across `lancedb.databases`.

    Returns:
        HaikuRAGApp instance with proper config and db path.

    Raises:
        AmbiguousDatabaseError: Several databases are configured and this
            command works on one, without `--db` or `--database` naming which.
    """
    from haiku.rag.app import HaikuRAGApp

    db_path = resolve_db_path(db, federated=federated)
    return HaikuRAGApp(
        db_path=db_path,
        config=get_config(),
        read_only=_read_only,
        federated=federated and db is None and _database is None,
    )


def resolve_db_path(db: Path | None = None, *, federated: bool = False) -> Path:
    """The database a command works on, from `--db`, `--database`, or config."""
    if db is not None and _database is not None:
        raise AmbiguousDatabaseError(
            "pass --db or --database, not both: they name the same thing"
        )
    require_one_database(get_config(), db, federated=federated)
    if db is not None:
        return db
    if _database_path is not None:
        return _database_path
    return get_config().storage.data_dir / "haiku.rag.lancedb"


def require_one_database(
    config: "AppConfig", db: Path | None, *, federated: bool
) -> None:
    """Refuse a one-database command that cannot tell which one to use."""
    databases = config.lancedb.databases
    if federated or db is not None or not databases:
        return
    raise AmbiguousDatabaseError(
        f"lancedb.databases names {', '.join(sorted(databases))}; this command "
        "works on a single database: pass --database NAME, or --db PATH."
    )


def select_database(name: str) -> Path | None:
    """Point the configuration at one database from `lancedb.databases`.

    Returns its local path, or None where it lives behind a URI. Rewriting the
    configuration is what lets every command, the TUIs included, work on the
    selected database without knowing the set exists.
    """
    from haiku.rag.utils import locate_database

    config = get_config()
    databases = config.lancedb.databases
    if name not in databases:
        raise AmbiguousDatabaseError(
            f"unknown database {name!r}; lancedb.databases names "
            f"{', '.join(sorted(databases)) or 'nothing'}"
        )
    uri, db_path = locate_database(databases[name])
    selected = config.model_copy(deep=True)
    selected.lancedb.databases = {}
    selected.lancedb.uri = uri
    set_config(selected)
    return db_path


async def check_version():
    """Check if haiku.rag is up to date and show warning if not."""
    up_to_date, current_version, latest_version = await is_up_to_date()
    if not up_to_date:
        typer.echo(
            f"Warning: haiku.rag is outdated. Current: {current_version}, Latest: {latest_version}",
        )
        typer.echo("Please update.")


def version_callback(value: bool):
    if value:
        v = version("haiku.rag-slim")
        typer.echo(f"haiku.rag version {v}")
        raise typer.Exit()


@_cli.callback()
def main(
    _version: bool = typer.Option(
        False,
        "-v",
        "--version",
        callback=version_callback,
        help="Show version and exit",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Path to YAML configuration file",
    ),
    read_only: bool = typer.Option(
        False,
        "--read-only",
        help="Open database in read-only mode",
    ),
    database: str | None = typer.Option(
        None,
        "--database",
        help="Name of a database from lancedb.databases to work on",
    ),
):
    """haiku.rag CLI - Vector database RAG system"""
    global _read_only, _database, _database_path
    _read_only = read_only
    _database = database
    _database_path = None
    # Load config from --config, local folder, or default directory
    config_path = find_config_file(cli_path=config)
    if config_path:
        yaml_data = load_yaml_config(config_path)
        set_config(AppConfig.model_validate(yaml_data))
    else:
        set_config(AppConfig())

    if database is not None:
        _database_path = select_database(database)

    configure_cli_logging()

    from haiku.rag.telemetry import configure as configure_telemetry

    is_production = get_config().environment != "development"
    configure_telemetry(
        service_name="haiku-rag", console=False if is_production else None
    )

    if get_config().environment != "development":
        # Suppress warnings in production
        warnings.filterwarnings("ignore")

    # Run version check before any command
    try:
        asyncio.run(check_version())
    except Exception:
        # Do not block CLI on version check issues
        pass


@_cli.command("list", help="List all stored documents")
def list_documents(
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
    filter: str | None = typer.Option(
        None,
        "--filter",
        "-f",
        help="SQL WHERE clause to filter documents (e.g., \"uri LIKE '%arxiv%'\")",
    ),
):
    app = create_app(db)
    asyncio.run(app.list_documents(filter=filter))


def _parse_meta_options(meta: list[str] | None) -> dict[str, Any]:
    """Parse repeated --meta KEY=VALUE options into a dictionary.

    Raises a Typer error if any entry is malformed.
    """
    result: dict[str, Any] = {}
    if not meta:
        return result
    for item in meta:
        if "=" not in item:
            raise typer.BadParameter("--meta must be in KEY=VALUE format")
        key, value = item.split("=", 1)
        if not key:
            raise typer.BadParameter("--meta key cannot be empty")
        # Best-effort JSON coercion: numbers, booleans, null, arrays/objects
        try:
            parsed = json.loads(value)
            result[key] = parsed
        except Exception:
            # Leave as string if not valid JSON literal
            result[key] = value
    return result


@_cli.command("add", help="Add a document from text input")
def add_document_text(
    text: str = typer.Argument(
        help="The text content of the document to add",
    ),
    title: str | None = typer.Option(
        None,
        "--title",
        help="Optional title for the document",
    ),
    meta: list[str] | None = typer.Option(
        None,
        "--meta",
        help="Metadata entries as KEY=VALUE (repeatable)",
        metavar="KEY=VALUE",
    ),
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
):
    app = create_app(db)
    metadata = _parse_meta_options(meta)
    asyncio.run(
        app.add_document_from_text(text=text, title=title, metadata=metadata or None)
    )


@_cli.command("add-src", help="Add a document from a file path, directory, or URL")
def add_document_src(
    source: str = typer.Argument(
        help="The file path, directory, or URL of the document(s) to add",
    ),
    title: str | None = typer.Option(
        None,
        "--title",
        help="Optional human-readable title to store with the document",
    ),
    meta: list[str] | None = typer.Option(
        None,
        "--meta",
        help="Metadata entries as KEY=VALUE (repeatable)",
        metavar="KEY=VALUE",
    ),
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
):
    app = create_app(db)
    metadata = _parse_meta_options(meta)
    asyncio.run(
        app.add_document_from_source(
            source=source, title=title, metadata=metadata or None
        )
    )


@_cli.command("get", help="Get and display a document by its ID")
def get_document(
    doc_id: str = typer.Argument(
        help="The ID of the document to get",
    ),
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
):
    app = create_app(db)
    asyncio.run(app.get_document(doc_id=doc_id))


@_cli.command("delete", help="Delete a document by its ID")
def delete_document(
    doc_id: str = typer.Argument(
        help="The ID of the document to delete",
    ),
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
):
    app = create_app(db)
    asyncio.run(app.delete_document(doc_id=doc_id))


# Add alias `rm` for delete
_cli.command("rm", help="Alias for delete: remove a document by its ID")(
    delete_document
)


@_cli.command("search", help="Search for documents by a query")
def search(
    query: str | None = typer.Argument(
        None,
        help="The search query (omit when using --image)",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-l",
        help="Maximum number of results to return (default: config search.default_limit)",
    ),
    filter: str | None = typer.Option(
        None,
        "--filter",
        "-f",
        help="SQL WHERE clause to filter documents (e.g., \"uri LIKE '%arxiv%'\")",
    ),
    search_type: SearchType | None = typer.Option(
        None,
        "--search-type",
        "-s",
        help="Type of search to perform (text searches only)",
    ),
    image: Path | None = typer.Option(
        None,
        "--image",
        help="Path to an image file to use as the query (requires a multimodal embedder)",
    ),
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
):
    app = create_app(db, federated=True)
    asyncio.run(
        app.search(
            query=query,
            limit=limit,
            filter=filter,
            search_type=search_type,
            image=image,
        )
    )


@_cli.command("visualize", help="Show visual grounding for a chunk")
def visualize(
    chunk_id: str = typer.Argument(
        help="The ID of the chunk to visualize",
    ),
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
    no_expand: bool = typer.Option(
        False,
        "--no-expand",
        help="Highlight only the chunk itself, without its expanded context",
    ),
):
    app = create_app(db)
    asyncio.run(app.visualize_chunk(chunk_id=chunk_id, expand=not no_expand))


@_cli.command("ask", help="Ask a question using the QA agent")
def ask(
    question: str = typer.Argument(
        help="The question to ask",
    ),
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
    filter: str | None = typer.Option(
        None,
        "--filter",
        "-f",
        help="SQL WHERE clause to filter documents (e.g., \"uri LIKE '%arxiv%'\")",
    ),
    image: list[Path] | None = typer.Option(
        None,
        "--image",
        help="Path to an image to attach to the question (repeatable; requires a vision-capable model)",
    ),
):
    app = create_app(db, federated=True)
    asyncio.run(
        app.ask(
            question=question,
            filter=filter,
            images=image,
        )
    )


@_cli.command("analyze", help="Answer questions using the analysis capability")
def analyze(
    question: str = typer.Argument(
        help="The question to answer",
    ),
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
    filter: str | None = typer.Option(
        None,
        "--filter",
        "-f",
        help="SQL WHERE clause to filter documents (e.g., \"uri LIKE '%arxiv%'\")",
    ),
    image: list[Path] | None = typer.Option(
        None,
        "--image",
        help="Path to an image to attach to the question (repeatable; requires a vision-capable model)",
    ),
):
    app = create_app(db, federated=True)
    asyncio.run(
        app.analyze(
            question=question,
            filter=filter,
            images=image,
        )
    )


@_cli.command("settings", help="Display current configuration settings")
def settings():
    from haiku.rag.app import HaikuRAGApp

    config = get_config()
    app = HaikuRAGApp(db_path=Path(), config=config, read_only=True)
    app.show_settings()


@_cli.command("init-config", help="Generate a YAML configuration file")
def init_config(
    output: Path = typer.Argument(
        Path("haiku.rag.yaml"),
        help="Output path for the config file",
    ),
):
    """Generate a YAML configuration file with defaults."""
    import yaml

    from haiku.rag.config.loader import generate_default_config

    if output.exists():
        typer.echo(
            f"Error: {output} already exists. Remove it first or choose a different path."
        )
        raise typer.Exit(1)

    config_data = generate_default_config()

    # Write YAML with comments
    with open(output, "w") as f:
        f.write("# haiku.rag configuration file\n")
        f.write(
            "# See https://ggozad.github.io/haiku.rag/configuration/ for details\n\n"
        )
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    typer.echo(f"Configuration file created: {output}")
    typer.echo("Edit the file to customize your settings.")


@_cli.command(
    "rebuild",
    help="Rebuild the database by deleting all chunks and re-indexing all documents",
)
def rebuild(
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
    embed_only: bool = typer.Option(
        False,
        "--embed-only",
        help="Only regenerate embeddings, keep existing chunks",
    ),
    rechunk: bool = typer.Option(
        False,
        "--rechunk",
        help="Re-chunk from existing content without accessing source files",
    ),
    title_only: bool = typer.Option(
        False,
        "--title-only",
        help="Only generate titles for documents without one",
    ),
    descriptions: bool = typer.Option(
        False,
        "--descriptions",
        help=(
            "Run the VLM over already-stored picture bytes, patch descriptions "
            "into the docling blob, then re-chunk + re-embed. Skips the docling "
            "parse entirely. Requires processing.pictures='description'."
        ),
    ),
    set_embedder: bool = typer.Option(
        False,
        "--set-embedder",
        help=(
            "Adopt the current embedder identity without re-embedding, when the "
            "vector dimension is unchanged. Use after swapping the serving stack "
            "for the same model (e.g. Ollama to vLLM)."
        ),
    ),
):
    from haiku.rag.client import RebuildMode

    exclusive = sum([embed_only, rechunk, title_only, descriptions, set_embedder])
    if exclusive > 1:
        typer.echo(
            "Error: --embed-only, --rechunk, --title-only, --descriptions, and "
            "--set-embedder are mutually exclusive"
        )
        raise typer.Exit(1)

    if embed_only:
        mode = RebuildMode.EMBED_ONLY
    elif rechunk:
        mode = RebuildMode.RECHUNK
    elif title_only:
        mode = RebuildMode.TITLE_ONLY
    elif descriptions:
        mode = RebuildMode.DESCRIPTIONS
    elif set_embedder:
        mode = RebuildMode.SET_EMBEDDER
    else:
        mode = RebuildMode.FULL

    app = create_app(db)
    asyncio.run(app.rebuild(mode=mode))


@_cli.command("vacuum", help="Optimize and clean up all tables to reduce disk usage")
def vacuum(
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
):
    app = create_app(db)
    asyncio.run(app.vacuum())


@_cli.command("migrate", help="Run pending database migrations")
def migrate(
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
):
    app = create_app(db)
    try:
        applied = asyncio.run(app.migrate())
        if applied:
            typer.echo(f"Applied {len(applied)} migration(s):")
            for desc in applied:
                typer.echo(f"  - {desc}")
            typer.echo("Migration completed successfully.")
        else:
            typer.echo("No migrations pending. Database is up to date.")
    except Exception as e:
        typer.echo(f"Migration failed: {e}")
        raise typer.Exit(1)


@_cli.command(
    "create-index", help="Create vector index for efficient similarity search"
)
def create_index(
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
):
    app = create_app(db)
    asyncio.run(app.create_index())


@_cli.command("init", help="Initialize a new database")
def init_db(
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
):
    app = create_app(db)
    asyncio.run(app.init())


@_cli.command("info", help="Show database info")
def info(
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
):
    app = create_app(db)
    asyncio.run(app.info())


@_cli.command("doctor", help="Check database and provider health")
def doctor(
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
    duplicates_out: Path | None = typer.Option(
        None,
        "--duplicates-out",
        help="Write near-duplicate document groups to this YAML file",
    ),
):
    app = create_app(db)
    if asyncio.run(app.doctor(duplicates_out=duplicates_out)):
        raise typer.Exit(code=1)


@_cli.command("history", help="Show version history for database tables")
def history(
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
    table: str | None = typer.Option(
        None,
        "--table",
        "-t",
        help="Specific table to show history for (documents, document_meta, chunks, document_items, settings)",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-l",
        help="Maximum number of versions to show per table",
    ),
):
    app = create_app(db)
    asyncio.run(app.history(table=table, limit=limit))


tag_cli = typer.Typer(
    help="Manage database tags (named versions across all tables)",
    no_args_is_help=True,
)
_cli.add_typer(tag_cli, name="tag")


@tag_cli.command("create", help="Tag the current database state")
def tag_create(
    name: str = typer.Argument(help="Name of the tag to create"),
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
):
    app = create_app(db)
    try:
        asyncio.run(app.create_tag(name))
    except (ValueError, RuntimeError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@tag_cli.command("list", help="List database tags")
def tag_list(
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
):
    app = create_app(db)
    try:
        asyncio.run(app.list_tags())
    except (ValueError, RuntimeError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@tag_cli.command("delete", help="Delete a tag")
def tag_delete(
    name: str = typer.Argument(help="Name of the tag to delete"),
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
):
    app = create_app(db)
    try:
        asyncio.run(app.delete_tag(name))
    except (ValueError, RuntimeError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@tag_cli.command("restore", help="Restore the database to a tagged state")
def tag_restore(
    name: str = typer.Argument(help="Name of the tag to restore"),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Skip the confirmation prompt. Provides no locking or "
        "concurrent-writer protection.",
    ),
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
):
    app = create_app(db)
    if app._is_local and not app.db_path.exists():
        typer.echo(f"Error: Database path does not exist: {app.db_path}", err=True)
        raise typer.Exit(1)
    if not yes:
        typer.echo(f"Database: {app.db_path}")
        typer.echo(f"Tag: {name}")
        typer.echo("This changes the live database state across all tables.")
        typer.echo("Stop all ingestion and other writers before continuing.")
        typer.echo("The operation is coordinated but not transactionally atomic.")
        typer.echo("A safety tag will preserve the current state.")
        if not typer.confirm("Continue?", default=False):
            raise typer.Exit(1)
    try:
        asyncio.run(app.restore_tag(name))
    except (ValueError, RuntimeError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@_cli.command("download-models", help="Download Docling and Ollama models per config")
def download_models_cmd():
    from haiku.rag.app import HaikuRAGApp

    app = HaikuRAGApp(db_path=Path(), config=get_config(), read_only=True)
    try:
        asyncio.run(app.download_models())
    except Exception as e:
        typer.echo(f"Error downloading models: {e}")
        raise typer.Exit(1)


@_cli.command("inspect", help="Launch interactive TUI to inspect database contents")
def inspect(
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
):
    """Launch the inspector TUI for browsing documents and chunks."""
    try:
        from haiku.rag.inspector import run_inspector
    except ImportError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from e

    run_inspector(resolve_db_path(db), read_only=True)


@_cli.command("chat", help="Launch interactive chat TUI for conversational RAG")
def chat(
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Model to use for the chat (e.g. openai-chat:gpt-4o)",
    ),
    capability: list[str] | None = typer.Option(
        None,
        "--capability",
        "-c",
        help="Capabilities to enable: rag, analysis (can repeat, default: rag)",
    ),
):
    """Launch the chat TUI for conversational RAG."""
    from haiku.rag.chat import run_chat

    db_path = resolve_db_path(db)
    capabilities = capability if capability else ["rag"]

    try:
        run_chat(
            db_path,
            read_only=True,
            model=model,
            capabilities=capabilities,
        )
    except ImportError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from e


@_cli.command(
    "mcp",
    help="Run the MCP server. For continuous ingestion, use haiku-ingester serve.",
)
def mcp(
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
    stdio: bool = typer.Option(
        False,
        "--stdio",
        help="Run MCP server on stdio Transport",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host to bind MCP server to (use 0.0.0.0 in containers; ignored with --stdio)",
    ),
    port: int = typer.Option(
        8001,
        "--port",
        help="Port to bind MCP server to (ignored with --stdio)",
    ),
) -> None:
    """Run the MCP server."""
    app = create_app(db)

    transport = "stdio" if stdio else None

    asyncio.run(app.run_mcp(transport=transport, host=host, port=port))


if __name__ == "__main__":
    cli()  # pragma: no cover - module-as-script entry, never imported by tests
