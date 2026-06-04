import asyncio
import sys
from pathlib import Path

import typer
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import make_url

load_dotenv(find_dotenv(usecwd=True))

from haiku.rag.config import (  # noqa: E402
    AppConfig,
    QueueConfig,
    find_config_file,
    get_config,
    load_yaml_config,
    set_config,
)
from haiku.rag.ingester.app import IngesterApp  # noqa: E402
from haiku.rag.ingester.queue.migrations import open_queue  # noqa: E402
from haiku.rag.logging import configure_cli_logging  # noqa: E402
from haiku.rag.store.exceptions import (  # noqa: E402
    MigrationRequiredError,
    ReadOnlyError,
)

_cli = typer.Typer(
    name="haiku-ingester",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="Production ingester for haiku.rag.",
)


@_cli.callback()
def main(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to haiku.rag.yaml. Falls back to a discovered project YAML, then the process default.",
    ),
) -> None:
    """Top-level callback so every subcommand inherits --config without
    each one redeclaring it. Mirrors haiku-rag's CLI shape."""
    _load_config_with_override(config)


def cli() -> None:
    """Entry point that translates store-state errors into a clean exit."""
    from haiku.rag.telemetry import configure as configure_telemetry

    configure_cli_logging()
    configure_telemetry(service_name="haiku-ingester")
    try:
        _cli()
    except (MigrationRequiredError, ReadOnlyError) as e:
        typer.echo(f"Error: {e}", err=True)
        sys.exit(1)


queue_cli = typer.Typer(
    name="queue",
    no_args_is_help=True,
    help="Operate the ingester's job queue.",
)
_cli.add_typer(queue_cli)


def _load_config_with_override(config_path: Path | None) -> AppConfig:
    """Load AppConfig from `config_path`, the discovered project YAML, or the
    process default — in that order."""
    if config_path:
        config = AppConfig.model_validate(load_yaml_config(config_path))
        set_config(config)
        return config
    if (found := find_config_file(None)) is not None:
        config = AppConfig.model_validate(load_yaml_config(found))
        set_config(config)
        return config
    return get_config()


def _resolve_queue_config(config: AppConfig, override: Path | None) -> QueueConfig:
    """The configured queue, with `--queue` applied as a path override. The
    override is ignored when a dburi is set — the queue lives in a server."""
    queue = config.ingester.queue
    if override is not None and queue.dburi is None:
        return queue.model_copy(update={"path": Path(override).expanduser()})
    return queue


def _queue_target(queue: QueueConfig) -> str:
    """A display string for the queue location, with any dburi password
    masked so it isn't echoed to the terminal or logs."""
    if queue.dburi:
        return make_url(queue.dburi).render_as_string(hide_password=True)
    return str(queue.path)


async def _ensure_schema(queue: QueueConfig) -> None:
    engine = await open_queue(queue)
    await engine.dispose()


@queue_cli.command("init")
def queue_init(
    queue: Path | None = typer.Option(
        None,
        "--queue",
        "-q",
        help="Override the queue DB path (defaults to ingester.queue.path).",
    ),
) -> None:
    """Create the queue DB and apply the current schema. Idempotent."""
    queue_config = _resolve_queue_config(get_config(), queue)
    asyncio.run(_ensure_schema(queue_config))
    typer.echo(f"Queue initialized at {_queue_target(queue_config)}")


@queue_cli.command("migrate")
def queue_migrate(
    queue: Path | None = typer.Option(
        None,
        "--queue",
        "-q",
        help="Override the queue DB path (defaults to ingester.queue.path).",
    ),
) -> None:
    """Apply any pending schema migrations to an existing queue DB. Idempotent."""
    queue_config = _resolve_queue_config(get_config(), queue)
    asyncio.run(_ensure_schema(queue_config))
    typer.echo(f"Queue at {_queue_target(queue_config)} is up to date")


def _resolve_db_path(config: AppConfig, override: Path | None) -> Path:
    return override or (config.storage.data_dir / "haiku.rag.lancedb")


@_cli.command("serve")
def serve(
    db: Path | None = typer.Option(
        None,
        "--db",
        help="LanceDB path (overrides config.storage.data_dir).",
    ),
    host: str | None = typer.Option(
        None,
        "--host",
        help="Bind the HTTP control plane to HOST (overrides ingester.api.host; use 0.0.0.0 in containers).",
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        help="Bind the HTTP control plane to PORT (overrides ingester.api.port).",
    ),
    no_api: bool = typer.Option(
        False,
        "--no-api",
        help="Run pollers + workers without the HTTP control plane.",
    ),
) -> None:
    """Run the production ingester: pollers + workers (and the HTTP API
    unless --no-api is set). Blocks until SIGINT/SIGTERM."""
    app_config = get_config()
    if host is not None:
        app_config.ingester.api.host = host
    if port is not None:
        app_config.ingester.api.port = port
    db_path = _resolve_db_path(app_config, db)
    app = IngesterApp(config=app_config, db_path=db_path)
    asyncio.run(app.serve(api=not no_api))


@_cli.command("run-batch")
def run_batch(
    db: Path | None = typer.Option(
        None,
        "--db",
        help="LanceDB path (overrides config.storage.data_dir).",
    ),
) -> None:
    """Run one discover sweep across every configured source, drain the queue,
    then exit. New and changed resources are ingested, resources that vanished
    from a source are deleted. Exits non-zero if any job dead-letters or a
    source's sweep does not complete."""
    asyncio.run(_run_batch(get_config(), db))


async def _run_batch(app_config: AppConfig, db_path: Path | None) -> None:
    db = _resolve_db_path(app_config, db_path)
    app = IngesterApp(config=app_config, db_path=db)
    report = await app.run_batch()
    typer.echo(f"Batch complete: {report.succeeded} succeeded, {report.dead} dead")
    if report.failed_sweeps:
        typer.echo(
            f"Sources that failed to sweep: {', '.join(report.failed_sweeps)}",
            err=True,
        )
    if report.dead or report.failed_sweeps:
        raise typer.Exit(1)
