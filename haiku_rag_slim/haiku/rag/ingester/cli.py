import asyncio
import sys
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import typer
import yaml
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
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
from haiku.rag.ingester.batch import BatchManifest  # noqa: E402
from haiku.rag.ingester.exceptions import (  # noqa: E402
    DuplicateSourceError,
    NothingToIngestError,
    QueueError,
)
from haiku.rag.ingester.queue.migrations import (  # noqa: E402
    apply_migrations,
    make_engine,
)
from haiku.rag.ingester.queue.models import JobStatus  # noqa: E402
from haiku.rag.logging import configure_cli_logging  # noqa: E402
from haiku.rag.store.exceptions import (  # noqa: E402
    AmbiguousDatabaseError,
    ConfigMismatchError,
    MigrationRequiredError,
    ReadOnlyError,
    SourceUnavailableError,
    TagError,
    UnknownDatabaseError,
)

if TYPE_CHECKING:
    from haiku.rag.ingester.app import BatchProgress, BatchProgressCallback

_cli = typer.Typer(
    name="haiku-ingester",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="Production ingester for haiku.rag.",
)


_LONG_RUNNING = frozenset({"serve"})


@_cli.callback()
def main(
    ctx: typer.Context,
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to haiku.rag.yaml. Falls back to a discovered project YAML, then the process default.",
    ),
) -> None:
    """Top-level callback so every subcommand inherits --config without
    each one redeclaring it. Mirrors haiku-rag's CLI shape."""
    from haiku.rag.telemetry import attach_env_context
    from haiku.rag.telemetry import configure as configure_telemetry

    _load_config_with_override(config)
    # The file read again on a reload, through any symlink as it is then.
    found = config if config is not None else find_config_file(None)
    ctx.obj = found.absolute() if found is not None else None
    configure_cli_logging()
    configure_telemetry(service_name="haiku-ingester")
    if ctx.invoked_subcommand not in _LONG_RUNNING:
        ctx.with_resource(attach_env_context())


def cli() -> None:
    """Entry point that translates known errors into a clean exit."""
    try:
        _cli()
    except (
        AmbiguousDatabaseError,
        ConfigMismatchError,
        DuplicateSourceError,
        ImportError,
        MigrationRequiredError,
        NothingToIngestError,
        QueueError,
        ReadOnlyError,
        SourceUnavailableError,
        TagError,
        UnknownDatabaseError,
    ) as e:
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


def _resolve_queue_config(
    config: AppConfig, override: Path | None, dburi: str | None = None
) -> QueueConfig:
    """The configured queue, with `--queue` applied as a path override. The
    override is ignored when a dburi is set — the queue lives in a server.
    `--dburi` names a queue outside the configuration, as the account that
    owns its schema."""
    if dburi is not None:
        if override is not None:
            typer.echo("Error: --dburi cannot be combined with --queue", err=True)
            raise typer.Exit(2)
        return QueueConfig(dburi=dburi)
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
    engine = make_engine(queue)
    try:
        await apply_migrations(engine)
    finally:
        await engine.dispose()


_QUEUE_OPTION = typer.Option(
    None,
    "--queue",
    "-q",
    help="Override the queue DB path (defaults to ingester.queue.path).",
)
_DBURI_OPTION = typer.Option(
    None,
    "--dburi",
    envvar="HAIKU_RAG_QUEUE_DBURI",
    help="The queue to migrate, as the account that owns its schema, "
    "instead of the configured one.",
)


@queue_cli.command("init")
def queue_init(
    queue: Path | None = _QUEUE_OPTION, dburi: str | None = _DBURI_OPTION
) -> None:
    """Create the queue DB and apply the current schema. Idempotent."""
    queue_config = _resolve_queue_config(get_config(), queue, dburi)
    asyncio.run(_ensure_schema(queue_config))
    typer.echo(f"Queue initialized at {_queue_target(queue_config)}")


@queue_cli.command("migrate")
def queue_migrate(
    queue: Path | None = _QUEUE_OPTION, dburi: str | None = _DBURI_OPTION
) -> None:
    """Apply any pending schema migrations to an existing queue DB. Idempotent."""
    queue_config = _resolve_queue_config(get_config(), queue, dburi)
    asyncio.run(_ensure_schema(queue_config))
    typer.echo(f"Queue at {_queue_target(queue_config)} is up to date")


def _default_manifest_path() -> Path:
    datestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")
    return Path(f"manifest-{datestamp}.yaml")


def _write_manifest(manifest: BatchManifest, path: Path) -> None:
    data = manifest.model_dump(mode="json")
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


@contextmanager
def _batch_progress(
    description: str,
) -> "Iterator[BatchProgressCallback | None]":  # pragma: no cover
    console = Console(file=sys.stdout)
    if not console.is_terminal:
        yield None
        return

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    task_id = None

    def _update(snapshot: "BatchProgress") -> None:
        nonlocal task_id
        task_description = (
            f"{description} ({snapshot.succeeded} ok, {snapshot.dead} dead, "
            f"{snapshot.rejected} rejected)"
        )
        if task_id is None:
            task_id = progress.add_task(
                task_description,
                total=snapshot.total,
                completed=snapshot.completed,
            )
            return
        progress.update(
            task_id,
            description=task_description,
            total=snapshot.total,
            completed=snapshot.completed,
        )

    with progress:
        yield _update


def _load_manifest(path: Path) -> BatchManifest:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return BatchManifest.model_validate(data)


@_cli.command("serve")
def serve(
    ctx: typer.Context,
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
    root_path: str | None = typer.Option(
        None,
        "--root-path",
        help="Serve the control plane under a base path so it can be reverse-"
        "proxied behind a sub-path, e.g. /ingester (overrides "
        "ingester.api.root_path).",
    ),
    no_api: bool = typer.Option(
        False,
        "--no-api",
        help="Run pollers + workers without the HTTP control plane.",
    ),
) -> None:
    """Run the production ingester: pollers + workers (and the HTTP API
    unless --no-api is set). Blocks until SIGINT/SIGTERM. SIGHUP reloads the
    databases from the configuration file."""
    from haiku.rag.ingester.app import IngesterApp

    app_config = get_config()
    overrides = {
        field: value
        for field, value in {
            "host": host,
            "port": port,
            "root_path": root_path,
        }.items()
        if value is not None
    }
    for field, value in overrides.items():
        setattr(app_config.ingester.api, field, value)
    app = IngesterApp(config=app_config, config_path=ctx.obj, api_overrides=overrides)
    asyncio.run(app.serve(api=not no_api))


class RetrySelection(StrEnum):
    DEAD = "dead"
    REJECTED = "rejected"
    ALL = "all"


_RETRY_STATUSES = {
    RetrySelection.DEAD: {JobStatus.DEAD},
    RetrySelection.REJECTED: {JobStatus.REJECTED},
    RetrySelection.ALL: {JobStatus.DEAD, JobStatus.REJECTED},
}


@_cli.command("run-batch")
def run_batch(
    db_name: str | None = typer.Option(
        None,
        "--db-name",
        help="Run one database of ingester.databases, by name, instead of all.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Discover planned changes and write a YAML manifest without ingesting.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Dry-run manifest path (defaults to manifest-<datestamp>.yaml).",
    ),
    manifest: Path | None = typer.Option(
        None,
        "--manifest",
        help="Replay a dry-run manifest instead of running discovery.",
    ),
    retry: RetrySelection | None = typer.Option(
        None,
        "--retry",
        help="After the sweep, re-queue the latest dead or rejected job of each "
        "document not ingested since.",
    ),
) -> None:
    """Run one discover sweep across the sources of every ingested database,
    or of the one --db-name selects, drain the queue, then exit. New and
    changed resources are ingested, resources that vanished from a source
    are deleted. Exits non-zero if any job dead-letters or a
    source's sweep does not complete."""
    if manifest is not None and dry_run:
        typer.echo("Error: --manifest cannot be combined with --dry-run")
        raise typer.Exit(2)
    if retry is not None and (dry_run or manifest is not None):
        other = "--dry-run" if dry_run else "--manifest"
        typer.echo(f"Error: --retry cannot be combined with {other}")
        raise typer.Exit(2)
    if output is not None and not dry_run:
        typer.echo("Error: --output is only valid with --dry-run")
        raise typer.Exit(2)
    asyncio.run(
        _run_batch(
            get_config(),
            db_name,
            dry_run=dry_run,
            output=output,
            manifest_path=manifest,
            retry=_RETRY_STATUSES[retry] if retry is not None else set(),
        )
    )


async def _run_batch(
    app_config: AppConfig,
    db_name: str | None,
    *,
    dry_run: bool = False,
    output: Path | None = None,
    manifest_path: Path | None = None,
    retry: Collection[JobStatus] = frozenset(),
) -> None:
    from haiku.rag.ingester.app import IngesterApp

    app = IngesterApp(config=app_config, database_name=db_name)
    if dry_run:
        report = await app.run_batch_dry_run()
        if report.failed_sweeps:
            typer.echo(
                f"Sources that failed to sweep: {', '.join(report.failed_sweeps)}"
            )
            raise typer.Exit(1)
        manifest_path = output or _default_manifest_path()
        _write_manifest(report.manifest, manifest_path)
        upserts = sum(source.upsert_count for source in report.manifest.sources)
        deletes = sum(source.delete_count for source in report.manifest.sources)
        unchanged = sum(source.unchanged_count for source in report.manifest.sources)
        typer.echo(
            "Dry run complete: "
            f"{upserts} upsert, {deletes} delete, {unchanged} unchanged "
            f"-> {manifest_path}"
        )
        return

    if manifest_path is not None:
        try:
            manifest = _load_manifest(manifest_path)
            with _batch_progress("Replaying manifest") as progress_callback:
                if progress_callback is None:
                    report = await app.run_batch_from_manifest(manifest)
                else:
                    report = await app.run_batch_from_manifest(
                        manifest, progress_callback=progress_callback
                    )
        except ValueError as exc:
            typer.echo(f"Error: {exc}")
            raise typer.Exit(1) from exc
        typer.echo(
            "Manifest batch complete: "
            f"{report.succeeded} succeeded, {report.dead} dead, "
            f"{report.rejected} rejected"
        )
        if report.dead:
            raise typer.Exit(1)
        return

    with _batch_progress("Running batch") as progress_callback:
        if progress_callback is None:
            report = await app.run_batch(retry=retry)
        else:
            report = await app.run_batch(
                retry=retry, progress_callback=progress_callback
            )
    if retry:
        typer.echo(f"Retried {report.retried} job(s)")
    typer.echo(
        f"Batch complete: {report.succeeded} succeeded, {report.dead} dead, "
        f"{report.rejected} rejected"
    )
    if report.failed_sweeps:
        typer.echo(
            f"Sources that failed to sweep: {', '.join(report.failed_sweeps)}",
            err=True,
        )
    if report.dead or report.failed_sweeps:
        raise typer.Exit(1)
