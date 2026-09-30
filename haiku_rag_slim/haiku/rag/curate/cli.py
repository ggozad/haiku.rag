import asyncio
import sys
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
from haiku.rag.config.models import CurateStoreConfig  # noqa: E402
from haiku.rag.curate.app import (  # noqa: E402
    APIServerStopped,
    describe,
    ensure_store,
    run_sweep,
    serve,
)
from haiku.rag.curate.store.migrations import UnsupportedStoreError  # noqa: E402
from haiku.rag.curate.store.models import SweepStatus  # noqa: E402
from haiku.rag.logging import configure_cli_logging  # noqa: E402
from haiku.rag.sqlstore import display_target  # noqa: E402
from haiku.rag.store.exceptions import (  # noqa: E402
    AmbiguousDatabaseError,
    UnknownDatabaseError,
)

_cli = typer.Typer(
    name="haiku-curate",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="Corpus curation for haiku.rag.",
)

store_cli = typer.Typer(
    name="store",
    no_args_is_help=True,
    help="Operate curate's history store.",
)
_cli.add_typer(store_cli)


@_cli.callback()
def main(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to haiku.rag.yaml. Falls back to a discovered project YAML, then the process default.",
    ),
) -> None:
    from haiku.rag.telemetry import configure as configure_telemetry

    found = find_config_file(config)
    set_config(
        AppConfig.model_validate(load_yaml_config(found)) if found else AppConfig()
    )
    configure_cli_logging()
    configure_telemetry(service_name="haiku-curate")


def cli() -> None:
    """Entry point that turns scope, store and server errors into a clean exit."""
    try:
        _cli()
    except (
        AmbiguousDatabaseError,
        UnknownDatabaseError,
        UnsupportedStoreError,
        APIServerStopped,
    ) as e:
        typer.echo(f"Error: {e}", err=True)
        sys.exit(1)


@_cli.command("sweep")
def sweep_command() -> None:
    """Sweep every configured database once; exits 1 when any database fails."""
    results = asyncio.run(run_sweep(get_config()))
    for result in results:
        typer.echo(describe(result))
    if any(result.status is SweepStatus.ERROR for result in results):
        raise typer.Exit(code=1)


@_cli.command("serve")
def serve_command(
    no_sweep: bool = typer.Option(
        False,
        "--no-sweep",
        help="Serve the API only; sweep elsewhere, e.g. `haiku-curate sweep` in cron.",
    ),
) -> None:
    """Sweep every curate.sweep_interval_s and serve the API; stops on SIGINT/SIGTERM."""
    asyncio.run(serve(get_config(), sweeping=not no_sweep))


def _store_config(override: Path | None) -> CurateStoreConfig:
    store = get_config().curate.store
    if override is not None and store.dburi is None:
        return store.model_copy(update={"path": override.expanduser()})
    return store


_STORE_OPTION = typer.Option(
    None,
    "--store",
    "-s",
    help="Override the store path (defaults to curate.store.path).",
)


@store_cli.command("init")
def store_init(store: Path | None = _STORE_OPTION) -> None:
    """Create the store and apply the current schema. Idempotent."""
    config = _store_config(store)
    asyncio.run(ensure_store(config))
    typer.echo(f"Store initialized at {display_target(config.path, config.dburi)}")


@store_cli.command("migrate")
def store_migrate(store: Path | None = _STORE_OPTION) -> None:
    """Apply any pending schema migrations to the store. Idempotent."""
    config = _store_config(store)
    asyncio.run(ensure_store(config))
    typer.echo(f"Store at {display_target(config.path, config.dburi)} is up to date")
