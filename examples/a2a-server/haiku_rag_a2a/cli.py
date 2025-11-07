import asyncio
import logging
from pathlib import Path

import typer
import uvicorn

from haiku.rag.config import AppConfig, Config, load_yaml_config
from haiku.rag.utils import get_default_data_dir
from haiku_rag_a2a.a2a import create_a2a_app
from haiku_rag_a2a.a2a.client import run_interactive_client

logger = logging.getLogger(__name__)

cli = typer.Typer(name="haiku-rag-a2a", no_args_is_help=True)


@cli.command("serve", help="Start haiku.rag A2A (Agent-to-Agent) server")
def serve(
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Path to the database directory",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config",
        help="Path to the configuration file",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host to bind A2A server to",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        help="Port to bind A2A server to",
    ),
    max_contexts: int = typer.Option(
        1000,
        "--max-contexts",
        help="Maximum number of conversation contexts to keep in memory",
    ),
) -> None:
    """Start the A2A server."""
    config = Config
    if config_file:
        yaml_data = load_yaml_config(config_file)
        config = AppConfig.model_validate(yaml_data)

    if db is None:
        db = get_default_data_dir()

    if not db.exists():
        typer.echo(f"Error: Database directory {db} does not exist")
        raise typer.Exit(1)

    logger.info(f"Starting A2A server on {host}:{port}")

    app = create_a2a_app(db_path=db, config=config, max_contexts=max_contexts)
    uvicorn_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(uvicorn_config)
    asyncio.run(server.serve())


@cli.command("client", help="Run interactive client to chat with A2A server")
def client(
    url: str = typer.Option(
        "http://localhost:8000",
        "--url",
        help="URL of the A2A server",
    ),
):
    """Run the interactive A2A client."""
    asyncio.run(run_interactive_client(url))


if __name__ == "__main__":
    cli()
