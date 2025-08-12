import asyncio
import warnings
from importlib.metadata import version
from pathlib import Path

import typer
from rich.console import Console

from haiku.rag.app import HaikuRAGApp
from haiku.rag.config import Config
from haiku.rag.utils import is_up_to_date

if not Config.ENV == "development":
    warnings.filterwarnings("ignore")

cli = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]}, no_args_is_help=True
)

console = Console()


async def check_version():
    """Check if haiku.rag is up to date and show warning if not."""
    up_to_date, current_version, latest_version = await is_up_to_date()
    if not up_to_date:
        console.print(
            f"[yellow]Warning: haiku.rag is outdated. Current: {current_version}, Latest: {latest_version}[/yellow]"
        )
        console.print("[yellow]Please update.[/yellow]")


def version_callback(value: bool):
    if value:
        v = version("haiku.rag")
        console.print(f"haiku.rag version {v}")
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
):
    """haiku.rag CLI - SQLite-based RAG system"""
    # Run version check before any command
    asyncio.run(check_version())


@cli.command("list", help="List all stored documents")
def list_documents(
    db: Path = typer.Option(
        Config.DEFAULT_DATA_DIR / "haiku.rag.sqlite",
        "--db",
        help="Path to the SQLite database file",
    ),
):
    app = HaikuRAGApp(db_path=db)
    asyncio.run(app.list_documents())


@cli.command("add", help="Add a document from text input")
def add_document_text(
    text: str = typer.Argument(
        help="The text content of the document to add",
    ),
    db: Path = typer.Option(
        Config.DEFAULT_DATA_DIR / "haiku.rag.sqlite",
        "--db",
        help="Path to the SQLite database file",
    ),
):
    app = HaikuRAGApp(db_path=db)
    asyncio.run(app.add_document_from_text(text=text))


@cli.command("add-src", help="Add a document from a file path or URL")
def add_document_src(
    source: str = typer.Argument(
        help="The file path or URL of the document to add",
    ),
    db: Path = typer.Option(
        Config.DEFAULT_DATA_DIR / "haiku.rag.sqlite",
        "--db",
        help="Path to the SQLite database file",
    ),
):
    app = HaikuRAGApp(db_path=db)
    asyncio.run(app.add_document_from_source(source=source))


@cli.command("get", help="Get and display a document by its ID")
def get_document(
    doc_id: int = typer.Argument(
        help="The ID of the document to get",
    ),
    db: Path = typer.Option(
        Config.DEFAULT_DATA_DIR / "haiku.rag.sqlite",
        "--db",
        help="Path to the SQLite database file",
    ),
):
    app = HaikuRAGApp(db_path=db)
    asyncio.run(app.get_document(doc_id=doc_id))


@cli.command("delete", help="Delete a document by its ID")
def delete_document(
    doc_id: int = typer.Argument(
        help="The ID of the document to delete",
    ),
    db: Path = typer.Option(
        Config.DEFAULT_DATA_DIR / "haiku.rag.sqlite",
        "--db",
        help="Path to the SQLite database file",
    ),
):
    app = HaikuRAGApp(db_path=db)
    asyncio.run(app.delete_document(doc_id=doc_id))


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
    k: int = typer.Option(
        60,
        "--k",
        help="Reciprocal Rank Fusion k parameter",
    ),
    db: Path = typer.Option(
        Config.DEFAULT_DATA_DIR / "haiku.rag.sqlite",
        "--db",
        help="Path to the SQLite database file",
    ),
):
    app = HaikuRAGApp(db_path=db)
    asyncio.run(app.search(query=query, limit=limit, k=k))


@cli.command("ask", help="Ask a question using the QA agent")
def ask(
    question: str = typer.Argument(
        help="The question to ask",
    ),
    db: Path = typer.Option(
        Config.DEFAULT_DATA_DIR / "haiku.rag.sqlite",
        "--db",
        help="Path to the SQLite database file",
    ),
    cite: bool = typer.Option(
        False,
        "--cite",
        help="Include citations in the response",
    ),
):
    app = HaikuRAGApp(db_path=db)
    asyncio.run(app.ask(question=question, cite=cite))


@cli.command("settings", help="Display current configuration settings")
def settings():
    app = HaikuRAGApp(db_path=Path())  # Don't need actual DB for settings
    app.show_settings()


@cli.command(
    "rebuild",
    help="Rebuild the database by deleting all chunks and re-indexing all documents",
)
def rebuild(
    db: Path = typer.Option(
        Config.DEFAULT_DATA_DIR / "haiku.rag.sqlite",
        "--db",
        help="Path to the SQLite database file",
    ),
):
    app = HaikuRAGApp(db_path=db)
    asyncio.run(app.rebuild())


@cli.command(
    "serve", help="Start the haiku.rag MCP server (by default in streamable HTTP mode)"
)
def serve(
    db: Path = typer.Option(
        Config.DEFAULT_DATA_DIR / "haiku.rag.sqlite",
        "--db",
        help="Path to the SQLite database file",
    ),
    stdio: bool = typer.Option(
        False,
        "--stdio",
        help="Run MCP server on stdio Transport",
    ),
    sse: bool = typer.Option(
        False,
        "--sse",
        help="Run MCP server on SSE transport",
    ),
) -> None:
    """Start the MCP server."""
    if stdio and sse:
        console.print("[red]Error: Cannot use both --stdio and --http options[/red]")
        raise typer.Exit(1)

    app = HaikuRAGApp(db_path=db)

    transport = None
    if stdio:
        transport = "stdio"
    elif sse:
        transport = "sse"

    asyncio.run(app.serve(transport=transport))


if __name__ == "__main__":
    cli()
