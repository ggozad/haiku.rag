import asyncio
import json
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv

from haiku.rag.app import HaikuRAGApp
from haiku.rag.config import (
    AppConfig,
    find_config_file,
    get_config,
    load_yaml_config,
    set_config,
)
from haiku.rag.logging import configure_cli_logging
from haiku.rag.utils import is_up_to_date

# Load environment variables from .env file for API keys and service URLs
load_dotenv()

cli = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]}, no_args_is_help=True
)


def create_app(db: Path | None = None) -> HaikuRAGApp:
    """Create HaikuRAGApp with loaded config and resolved database path.

    Args:
        db: Optional database path. If None, uses path from config.

    Returns:
        HaikuRAGApp instance with proper config and db path.
    """
    config = get_config()
    db_path = db if db else config.storage.data_dir / "haiku.rag.lancedb"
    return HaikuRAGApp(db_path=db_path, config=config)


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


@cli.callback()
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
):
    """haiku.rag CLI - Vector database RAG system"""
    # Load config from --config, local folder, or default directory
    config_path = find_config_file(cli_path=config)
    if config_path:
        yaml_data = load_yaml_config(config_path)
        loaded_config = AppConfig.model_validate(yaml_data)
        set_config(loaded_config)

    # Configure logging minimally for CLI context
    if get_config().environment == "development":
        # Lazy import logfire only in development
        try:
            import logfire  # type: ignore

            logfire.configure(send_to_logfire="if-token-present")
            logfire.instrument_pydantic_ai()
        except Exception:
            pass
    else:
        configure_cli_logging()
        warnings.filterwarnings("ignore")

    # Run version check before any command
    try:
        asyncio.run(check_version())
    except Exception:
        # Do not block CLI on version check issues
        pass


@cli.command("list", help="List all stored documents")
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


@cli.command("add", help="Add a document from text input")
def add_document_text(
    text: str = typer.Argument(
        help="The text content of the document to add",
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
    asyncio.run(app.add_document_from_text(text=text, metadata=metadata or None))


@cli.command("add-src", help="Add a document from a file path, directory, or URL")
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


@cli.command("get", help="Get and display a document by its ID")
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


@cli.command("delete", help="Delete a document by its ID")
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
cli.command("rm", help="Alias for delete: remove a document by its ID")(delete_document)


@cli.command("search", help="Search for documents by a query")
def search(
    query: str = typer.Argument(
        help="The search query to use",
    ),
    limit: int = typer.Option(
        5,
        "--limit",
        "-l",
        help="Maximum number of results to return",
    ),
    filter: str | None = typer.Option(
        None,
        "--filter",
        "-f",
        help="SQL WHERE clause to filter documents (e.g., \"uri LIKE '%arxiv%'\")",
    ),
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
):
    app = create_app(db)
    asyncio.run(app.search(query=query, limit=limit, filter=filter))


@cli.command("ask", help="Ask a question using the QA agent")
def ask(
    question: str = typer.Argument(
        help="The question to ask",
    ),
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
    cite: bool = typer.Option(
        False,
        "--cite",
        help="Include citations in the response",
    ),
    deep: bool = typer.Option(
        False,
        "--deep",
        help="Use deep multi-agent QA for complex questions",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Show verbose progress output (only with --deep)",
    ),
):
    app = create_app(db)
    asyncio.run(app.ask(question=question, cite=cite, deep=deep, verbose=verbose))


@cli.command("research", help="Run multi-agent research and output a concise report")
def research(
    question: str = typer.Argument(
        help="The research question to investigate",
    ),
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Show planning, searching previews, evaluation summary, and stop reason",
    ),
):
    app = create_app(db)
    asyncio.run(app.research(question=question, verbose=verbose))


@cli.command("settings", help="Display current configuration settings")
def settings():
    config = get_config()
    app = HaikuRAGApp(db_path=Path(), config=config)
    app.show_settings()


@cli.command("init-config", help="Generate a YAML configuration file")
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


@cli.command(
    "rebuild",
    help="Rebuild the database by deleting all chunks and re-indexing all documents",
)
def rebuild(
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
):
    app = create_app(db)
    asyncio.run(app.rebuild())


@cli.command("vacuum", help="Optimize and clean up all tables to reduce disk usage")
def vacuum(
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
):
    app = create_app(db)
    asyncio.run(app.vacuum())


@cli.command("info", help="Show read-only database info (no upgrades or writes)")
def info(
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
):
    app = create_app(db)
    asyncio.run(app.info())


@cli.command("download-models", help="Download Docling and Ollama models per config")
def download_models_cmd():
    from haiku.rag.utils import prefetch_models

    try:
        prefetch_models()
        typer.echo("Models downloaded successfully.")
    except Exception as e:
        typer.echo(f"Error downloading models: {e}")
        raise typer.Exit(1)


@cli.command(
    "serve",
    help="Start haiku.rag server. Use --monitor, --mcp, and/or --agui to enable services.",
)
def serve(
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the LanceDB database file",
    ),
    monitor: bool = typer.Option(
        False,
        "--monitor",
        help="Enable file monitoring",
    ),
    mcp: bool = typer.Option(
        False,
        "--mcp",
        help="Enable MCP server",
    ),
    stdio: bool = typer.Option(
        False,
        "--stdio",
        help="Run MCP server on stdio Transport (requires --mcp)",
    ),
    mcp_port: int = typer.Option(
        8001,
        "--mcp-port",
        help="Port to bind MCP server to (ignored with --stdio)",
    ),
    agui: bool = typer.Option(
        False,
        "--agui",
        help="Enable AG-UI HTTP server for graph streaming",
    ),
) -> None:
    """Start the server with selected services."""
    # Require at least one service flag
    if not (monitor or mcp or agui):
        typer.echo(
            "Error: At least one service flag (--monitor, --mcp, or --agui) must be specified"
        )
        raise typer.Exit(1)

    if stdio and not mcp:
        typer.echo("Error: --stdio requires --mcp")
        raise typer.Exit(1)

    app = create_app(db)

    transport = "stdio" if stdio else None

    asyncio.run(
        app.serve(
            enable_monitor=monitor,
            enable_mcp=mcp,
            mcp_transport=transport,
            mcp_port=mcp_port,
            enable_agui=agui,
        )
    )


if __name__ == "__main__":
    cli()
