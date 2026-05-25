import asyncio
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import typer
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from haiku.rag.client import HaikuRAG  # noqa: E402
from haiku.rag.config import (  # noqa: E402
    AppConfig,
    find_config_file,
    get_config,
    load_yaml_config,
    set_config,
)
from haiku.rag.ingester.app import IngesterApp  # noqa: E402
from haiku.rag.ingester.exceptions import PermanentError, TransientError  # noqa: E402
from haiku.rag.ingester.queue.migrations import open_queue  # noqa: E402
from haiku.rag.ingester.queue.models import Job, JobOp, JobStatus  # noqa: E402
from haiku.rag.ingester.workers.pipeline import run_job  # noqa: E402
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


def _configure_logfire() -> None:
    """Logfire emits spans only when LOGFIRE_TOKEN is set; otherwise it
    stays silent. Console output is disabled in either case so span lines
    don't interleave with the ingester's own RichHandler logs — telemetry
    lives in the Logfire UI."""
    try:
        import logfire

        logfire.configure(
            service_name="haiku-ingester",
            send_to_logfire="if-token-present",
            console=False,
        )
        logfire.instrument_pydantic_ai()
    except Exception:  # pragma: no cover
        pass


def cli() -> None:
    """Entry point that translates store-state errors into a clean exit."""
    configure_cli_logging()
    _configure_logfire()
    try:
        _cli()
    except (MigrationRequiredError, ReadOnlyError) as e:
        typer.echo(f"Error: {e}", err=True)
        sys.exit(1)


queue_cli = typer.Typer(
    name="queue",
    no_args_is_help=True,
    help="Operate the ingester's SQLite job queue.",
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


def _resolve_queue_path(config: AppConfig, override: Path | None) -> Path:
    return Path(override).expanduser() if override else config.ingester.queue.path


async def _ensure_schema(path: Path) -> None:
    conn = await open_queue(path)
    await conn.close()


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
    path = _resolve_queue_path(get_config(), queue)
    asyncio.run(_ensure_schema(path))
    typer.echo(f"Queue initialized at {path}")


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
    path = _resolve_queue_path(get_config(), queue)
    asyncio.run(_ensure_schema(path))
    typer.echo(f"Queue at {path} is up to date")


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


@_cli.command("run-once")
def run_once(
    uri: str = typer.Argument(..., help="URI to ingest (file://, http(s)://, s3://)."),
    db: Path | None = typer.Option(
        None,
        "--db",
        help="LanceDB path (overrides config.storage.data_dir).",
    ),
    delete: bool = typer.Option(
        False, "--delete", help="Run a delete op instead of upsert."
    ),
) -> None:
    """Build an ad-hoc Job and run it through the worker pipeline once.

    Bypasses the queue — does NOT enqueue. Useful for smoke-testing the
    Source adapter + pipeline path without spinning up the full pool.
    """
    asyncio.run(_run_once(get_config(), uri, db, delete))


async def _run_once(
    app_config: AppConfig, uri: str, db_path: Path | None, delete: bool
) -> None:
    db = _resolve_db_path(app_config, db_path)
    now = datetime.now(UTC)
    job = Job(
        id=f"adhoc-{uuid.uuid4()}",
        source_id="adhoc",
        uri=uri,
        op=JobOp.DELETE if delete else JobOp.UPSERT,
        status=JobStatus.CLAIMED,
        attempts=1,
        max_attempts=1,
        enqueued_at=now,
        scheduled_at=now,
        claimed_at=now,
        claimed_by="run-once",
    )

    async with HaikuRAG(db, config=app_config) as client:
        try:
            result = await run_job(client, job)
        except PermanentError as e:
            typer.echo(f"PERMANENT: {e}", err=True)
            raise typer.Exit(2) from e
        except TransientError as e:
            typer.echo(f"TRANSIENT: {e}", err=True)
            raise typer.Exit(1) from e

    if result.deleted:
        typer.echo(f"Deleted document at {uri}")
    else:
        typer.echo(
            f"Ingested {uri}: document_id={result.document_id} "
            f"revision={result.revision} md5={result.content_hash}"
        )
