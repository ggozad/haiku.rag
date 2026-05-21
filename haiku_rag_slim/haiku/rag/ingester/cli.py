import asyncio
from pathlib import Path

import typer
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from haiku.rag.config import (  # noqa: E402
    AppConfig,
    find_config_file,
    get_config,
    load_yaml_config,
    set_config,
)
from haiku.rag.ingester.queue.migrations import open_queue  # noqa: E402

cli = typer.Typer(
    name="haiku-ingester",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="Production ingester for haiku.rag.",
)

queue_cli = typer.Typer(
    name="queue",
    no_args_is_help=True,
    help="Operate the ingester's SQLite job queue.",
)
cli.add_typer(queue_cli)


def _load_config_with_override(config_path: Path | None) -> AppConfig:
    """Mirror the haiku-rag CLI's config-loading pattern."""
    if config_path:
        config = AppConfig.model_validate(load_yaml_config(config_path))
        set_config(config)
        return config
    if (found := find_config_file(None)) is not None:
        config = AppConfig.model_validate(load_yaml_config(found))
        set_config(config)
        return config
    return get_config()


def _resolve_queue_path(config: AppConfig, override: Path | None) -> Path:
    return Path(override).expanduser() if override else config.ingester.queue.path


async def _ensure_schema(path: Path) -> None:
    conn = await open_queue(path)
    await conn.close()


@queue_cli.command("init")
def queue_init(
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Path to haiku.rag.yaml."
    ),
    queue: Path | None = typer.Option(
        None,
        "--queue",
        "-q",
        help="Override the queue DB path (defaults to ingester.queue.path).",
    ),
) -> None:
    """Create the queue DB and apply the current schema. Idempotent."""
    app_config = _load_config_with_override(config)
    path = _resolve_queue_path(app_config, queue)
    asyncio.run(_ensure_schema(path))
    typer.echo(f"Queue initialized at {path}")


@queue_cli.command("migrate")
def queue_migrate(
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Path to haiku.rag.yaml."
    ),
    queue: Path | None = typer.Option(
        None,
        "--queue",
        "-q",
        help="Override the queue DB path (defaults to ingester.queue.path).",
    ),
) -> None:
    """Apply any pending schema migrations to an existing queue DB. Idempotent."""
    app_config = _load_config_with_override(config)
    path = _resolve_queue_path(app_config, queue)
    asyncio.run(_ensure_schema(path))
    typer.echo(f"Queue at {path} is up to date")
