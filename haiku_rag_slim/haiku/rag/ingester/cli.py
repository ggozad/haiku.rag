import asyncio
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
from haiku.rag.ingester.exceptions import PermanentError, TransientError  # noqa: E402
from haiku.rag.ingester.queue.migrations import open_queue  # noqa: E402
from haiku.rag.ingester.queue.models import Job, JobOp, JobStatus  # noqa: E402
from haiku.rag.ingester.workers.pipeline import run_job  # noqa: E402

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


@cli.command("run-once")
def run_once(
    uri: str = typer.Argument(..., help="URI to ingest (file://, http(s)://, s3://)."),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Path to haiku.rag.yaml."
    ),
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
    app_config = _load_config_with_override(config)
    asyncio.run(_run_once(app_config, uri, db, delete))


async def _run_once(
    app_config: AppConfig, uri: str, db_path: Path | None, delete: bool
) -> None:
    db = db_path or (app_config.storage.data_dir / "haiku.rag.lancedb")
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
