import asyncio
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import typer
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.markup import escape

from evaluations.artifacts import download_dataset_db, upload_dataset_db
from evaluations.completion import complete_arm, window_start
from evaluations.config import DatasetSpec
from evaluations.datasets import DATASETS
from evaluations.experiment import code_revision, config_hash
from evaluations.pairing import check_pair_rows, orient, pair_outcomes, render
from evaluations.population import populate_db
from evaluations.preflight import run_preflight
from evaluations.qa import run_live_qa_benchmark, run_qa_benchmark
from evaluations.queue import Queue, default_logs_path, tmux_command
from evaluations.registry import Registry, default_registry_path, launch_record
from evaluations.results import default_results_path, find_results, read_results
from evaluations.retrieval import run_retrieval_benchmark
from evaluations.traces import case_outcomes, query_logfire
from haiku.rag.config import AppConfig, find_config_file, load_yaml_config
from haiku.rag.config.models import ModelConfig
from haiku.rag.logging import configure_cli_logging
from haiku.rag.telemetry import configure as configure_telemetry
from haiku.rag.utils import parse_model_option

load_dotenv(find_dotenv(usecwd=True))

# Scrubbing off: eval outputs are financial answers with words like "authorized"
# that trip Logfire's secret scrubber and redact the model's answer text.
configure_telemetry(service_name="evals", scrubbing=False)
configure_cli_logging()
console = Console()

REGISTRY_OPTION = typer.Option(
    None,
    "--registry",
    help="Registry file. Defaults to registry.sqlite in the evaluations data directory.",
)
RESULTS_OPTION = typer.Option(
    None,
    "--results",
    help="Per-case result files. Defaults to results/ in the evaluations data directory.",
)


async def evaluate_dataset(
    spec: DatasetSpec,
    config: AppConfig,
    skip_db: bool,
    skip_retrieval: bool,
    skip_qa: bool,
    limit: int | None,
    name: str | None,
    db_path: Path | None,
    vacuum_interval: int = 100,
    multimodal_only: bool = False,
    judge_model: ModelConfig | None = None,
    capability_model: ModelConfig | None = None,
    case_ids: set[str] | None = None,
    document_filter: str | None = None,
    results_dir: Path | None = None,
) -> None:
    if document_filter is not None:
        console.print(f"Document filter: {document_filter}", style="dim")

    if db_path is not None and config.lancedb.databases:
        raise ValueError(
            "--db PATH places the database where the configuration places none, "
            f"and this configuration names {', '.join(config.lancedb.databases)} "
            "in lancedb.databases. Drop --db to evaluate the configured set."
        )

    if not skip_db:
        if spec.uses_configured_databases(config, db_path):
            raise ValueError(
                "lancedb.databases places the databases this run reads, and "
                "population writes to one, so it would ingest into a database "
                "the run does not read. Pass --skip-db to evaluate the "
                "configured set, or --db PATH to populate and evaluate one."
            )
        console.print(f"Using dataset: {spec.key}", style="bold magenta")
        await populate_db(
            spec, config, db_path=db_path, vacuum_interval=vacuum_interval
        )

    if not skip_retrieval:
        console.print("Running retrieval benchmarks...", style="bold blue")
        await run_retrieval_benchmark(
            spec,
            config,
            limit=limit,
            name=name,
            db_path=db_path,
            multimodal_only=multimodal_only,
            document_filter=document_filter,
        )

    if not skip_qa:
        console.print("\nRunning QA benchmarks...", style="bold yellow")
        qa_benchmark = run_live_qa_benchmark if spec.live else run_qa_benchmark
        await qa_benchmark(
            spec,
            config,
            limit=limit,
            name=name,
            db_path=db_path,
            judge_model=judge_model,
            capability_model=capability_model,
            case_ids=case_ids,
            document_filter=document_filter,
            results_dir=results_dir,
        )


app = typer.Typer(help="Run retrieval and QA benchmarks for configured datasets.")


def _load_config(config_path: Path | None) -> AppConfig:
    """Load AppConfig from a file path or standard search path."""
    if config_path:
        if not config_path.exists():
            raise typer.BadParameter(f"Config file not found: {config_path}")
        console.print(f"Loading config from: {config_path}", style="dim")
        yaml_data = load_yaml_config(config_path)
        return AppConfig.model_validate(yaml_data)

    found = find_config_file(None)
    if found:
        console.print(f"Loading config from: {found}", style="dim")
        yaml_data = load_yaml_config(found)
        return AppConfig.model_validate(yaml_data)

    console.print("No config file found, using defaults", style="dim")
    return AppConfig()


def require_telemetry(no_telemetry: bool) -> None:
    """Exit 1 without LOGFIRE_TOKEN: Logfire ships spans only when it is set."""
    if no_telemetry or os.environ.get("LOGFIRE_TOKEN"):
        return
    console.print(
        "LOGFIRE_TOKEN is not set: this run would record no per-case results. "
        "Put the token in .env, or pass --no-telemetry to run without telemetry.",
        style="red",
    )
    raise typer.Exit(code=1)


def _print_run_identity(config: AppConfig) -> None:
    revision = code_revision()
    sha = revision["git_sha"] or "unknown"
    dirty = " (uncommitted changes)" if revision["git_dirty"] else ""
    console.print(
        f"Code: {sha}{dirty} | config hash: {config_hash(config)}", style="dim"
    )


def _load_case_ids(path: Path | None) -> set[str] | None:
    """Read a newline-delimited case-id file into a set (None when no path)."""
    if path is None:
        return None
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _resolve_dataset(dataset: str) -> DatasetSpec:
    """Resolve a dataset key to a DatasetSpec or raise BadParameter."""
    spec = DATASETS.get(dataset.lower())
    if spec is None:
        valid_datasets = ", ".join(sorted(DATASETS))
        raise typer.BadParameter(
            f"Unknown dataset '{dataset}'. Choose from: {valid_datasets}"
        )
    return spec


def _resolve_datasets(dataset: str) -> list[DatasetSpec]:
    """Resolve 'all' or a single dataset key to a list of DatasetSpecs.

    'all' yields one spec per database: query variants sharing a db_filename
    would otherwise be downloaded/uploaded twice.
    """
    if dataset.lower() == "all":
        seen: set[str] = set()
        specs: list[DatasetSpec] = []
        for spec in DATASETS.values():
            if spec.db_filename in seen:
                continue
            seen.add(spec.db_filename)
            specs.append(spec)
        return specs
    return [_resolve_dataset(dataset)]


@app.command()
def run(
    dataset: str = typer.Argument(..., help="Dataset key to evaluate."),
    config: Path | None = typer.Option(
        None, "--config", help="Path to haiku.rag YAML config file."
    ),
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Database path, where the configuration places no database.",
    ),
    skip_db: bool = typer.Option(
        False, "--skip-db", help="Skip updating the evaluation db."
    ),
    skip_retrieval: bool = typer.Option(
        False, "--skip-retrieval", help="Skip retrieval benchmark."
    ),
    skip_qa: bool = typer.Option(False, "--skip-qa", help="Skip QA benchmark."),
    limit: int | None = typer.Option(
        None, "--limit", help="Limit number of test cases for both retrieval and QA."
    ),
    name: str | None = typer.Option(None, "--name", help="Override evaluation name."),
    vacuum_interval: int = typer.Option(
        100, "--vacuum-interval", help="Vacuum every N documents during DB population."
    ),
    multimodal_only: bool = typer.Option(
        False,
        "--multimodal-only",
        help="Only evaluate queries requiring image understanding.",
    ),
    capability_model: str | None = typer.Option(
        None,
        "--capability-model",
        help="Capability model as 'provider:name'. Defaults to qa.model from the config.",
    ),
    document_filter: str | None = typer.Option(
        None,
        "--filter",
        "-f",
        help=(
            "SQL WHERE clause over document columns (id, uri, title, "
            "created_at, updated_at, metadata) restricting every benchmark "
            "search, e.g. \"uri LIKE '%arxiv%'\". metadata is stored as a "
            "string, so match it with LIKE."
        ),
    ),
    filter_ids: Path | None = typer.Option(
        None,
        "--filter-ids",
        help=(
            "Path to a newline-delimited file of QA case ids to run "
            "(failure-subset rerun). Filters QA only; retrieval is unaffected."
        ),
    ),
    no_telemetry: bool = typer.Option(
        False,
        "--no-telemetry",
        help=(
            "Run without Logfire telemetry. Without this flag a run refuses to "
            "start when LOGFIRE_TOKEN is not set."
        ),
    ),
    results: Path | None = RESULTS_OPTION,
) -> None:
    require_telemetry(no_telemetry)
    spec = _resolve_dataset(dataset)
    app_config = _load_config(config)
    _print_run_identity(app_config)
    judge_model_config = app_config.evaluations.judge
    capability_model_config = (
        parse_model_option(capability_model) if capability_model else None
    )

    asyncio.run(
        evaluate_dataset(
            spec=spec,
            config=app_config,
            skip_db=skip_db,
            skip_retrieval=skip_retrieval,
            skip_qa=skip_qa,
            limit=limit,
            name=name,
            db_path=db,
            vacuum_interval=vacuum_interval,
            multimodal_only=multimodal_only,
            judge_model=judge_model_config,
            capability_model=capability_model_config,
            case_ids=_load_case_ids(filter_ids),
            document_filter=document_filter,
            results_dir=results or default_results_path(),
        )
    )


def _registry(path: Path | None) -> Registry:
    return Registry(path or default_registry_path())


@app.command()
def preflight(
    arm: Path = typer.Argument(..., help="Arm file to check before it starts."),
    register: bool = typer.Option(
        False,
        "--register",
        help="On success, record the launch in the registry.",
    ),
    registry: Path | None = REGISTRY_OPTION,
) -> None:
    """Print every check an arm must pass; exit 1 when one fails."""
    result = asyncio.run(run_preflight(arm))
    for check in result.checks:
        colour, label = ("green", "ok  ") if check.ok else ("red", "FAIL")
        console.print(
            f"[{colour}]{label}[/{colour}] {check.name}: {escape(check.detail)}",
            soft_wrap=True,
        )
    if not result.ok:
        raise typer.Exit(code=1)
    if register:
        assert result.arm is not None and result.config is not None
        record = launch_record(
            result.arm,
            result.config,
            result.fingerprint,
            started_at=datetime.now(UTC).isoformat(timespec="seconds"),
            git_sha=result.git_sha,
        )
        _registry(registry).register_launch(record)
        console.print(f"registered {record.name}")


def _without_detach(argv: list[str]) -> list[str]:
    """`argv` with the --detach option and its value removed."""
    kept: list[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token == "--detach":
            skip = True
            continue
        if token.startswith("--detach="):
            continue
        kept.append(token)
    return kept


@app.command()
def queue(
    arms: list[Path] = typer.Argument(..., help="Arm files, run in this order."),
    registry: Path | None = REGISTRY_OPTION,
    logs: Path | None = typer.Option(
        None,
        "--logs",
        help="Log directory. Defaults to logs/ in the evaluations data directory.",
    ),
    repo: Path | None = typer.Option(
        None,
        "--repo",
        help="Checkout that provisions a missing worktree at the arm's sha.",
    ),
    env_file: Path | None = typer.Option(
        None, "--env", help=".env copied into a provisioned worktree."
    ),
    settle_minutes: float = typer.Option(
        15.0,
        "--settle-minutes",
        help="How long to wait for a run's spans to reach Logfire after it exits.",
    ),
    prefix: list[str] = typer.Option(
        ["uv", "run"],
        "--prefix",
        help="Command that runs `evaluations` in the worktree.",
    ),
    detach: str | None = typer.Option(
        None, "--detach", help="Run inside a detached tmux session of this name."
    ),
    results: Path | None = RESULTS_OPTION,
) -> None:
    """Run arms in order: preflight, smoke, register, run, complete."""
    if detach:
        argv = [sys.argv[0], *_without_detach(sys.argv[1:])]
        subprocess.run(tmux_command(detach, argv, Path.cwd()), check=True)
        console.print(f"queue running in tmux session {detach}")
        return
    runner = Queue(
        registry=_registry(registry),
        logs=logs or default_logs_path(),
        query=query_logfire,
        prefix=list(prefix),
        repo=repo,
        env_source=env_file,
        settle_seconds=settle_minutes * 60,
        results_dir=results or default_results_path(),
    )
    outcomes = runner.run(list(arms))
    for outcome in outcomes:
        console.print(
            f"{outcome.name}: {outcome.status}: {outcome.detail}",
            soft_wrap=True,
            highlight=False,
            markup=False,
        )
    if any(outcome.status != "valid" for outcome in outcomes):
        raise typer.Exit(code=1)


arms_app = typer.Typer(help="The registry of evaluation arms.")
app.add_typer(arms_app, name="arms")


@arms_app.command("list")
def arms_list(
    registry: Path | None = REGISTRY_OPTION,
    dataset: str | None = typer.Option(None, "--dataset", help="Only this dataset."),
    db: str | None = typer.Option(
        None, "--db", help="Only arms on this database path."
    ),
    status: str | None = typer.Option(
        None, "--status", help="Only launched, valid or void arms."
    ),
) -> None:
    """One line per arm, oldest first."""
    for record in _registry(registry).list(dataset=dataset, db_path=db, status=status):
        sha = (record.git_sha or "")[:12]
        accuracy = "" if record.accuracy is None else f" acc={record.accuracy:.4f}"
        cited = "" if record.cited_map is None else f" cited_map={record.cited_map:.4f}"
        cases = "" if record.cases is None else f" cases={record.cases}"
        db_name = Path(record.db_path).name if record.db_path else "-"
        console.print(
            f"{record.started_at}  {record.name}  {record.dataset}  {record.status}  "
            f"{sha}  {record.capability_model or '-'}  {db_name}{cases}{accuracy}{cited}",
            soft_wrap=True,
            highlight=False,
        )


@arms_app.command("show")
def arms_show(name: str, registry: Path | None = REGISTRY_OPTION) -> None:
    """Every field of one arm."""
    record = _registry(registry).get(name)
    if record is None:
        console.print(f"no arm named {name}", style="red")
        raise typer.Exit(code=1)
    for field_name, value in asdict(record).items():
        console.print(f"{field_name}: {value}", soft_wrap=True, highlight=False)


@arms_app.command("void")
def arms_void(
    name: str,
    reason: str = typer.Option(
        ..., "--reason", help="Why the arm's numbers must not be used."
    ),
    registry: Path | None = REGISTRY_OPTION,
) -> None:
    """Mark an arm void; its rows stay, its numbers are never paired."""
    try:
        _registry(registry).mark_void(name, reason)
    except ValueError as error:
        console.print(str(error), style="red")
        raise typer.Exit(code=1) from None
    console.print(f"{name} marked void: {reason}")


def _outcomes(record, key: str, since: str, results_dir: Path):
    """Per-case outcomes from the run's result file, else from Logfire."""
    path = find_results(results_dir, record.name)
    if path is not None:
        return read_results(path)[1]
    return case_outcomes(
        record.trace_id or "", key, query=query_logfire, min_timestamp=since
    )


@arms_app.command("pair")
def arms_pair(
    a: str,
    b: str,
    registry: Path | None = REGISTRY_OPTION,
    results: Path | None = RESULTS_OPTION,
) -> None:
    """The standard paired table for two registered arms. Treated and baseline
    come from the recorded comparator, never from argument order."""
    store = _registry(registry)
    records = []
    for name in (a, b):
        record = store.get(name)
        if record is None:
            console.print(f"no arm named {name}", style="red")
            raise typer.Exit(code=1)
        records.append(record)
    try:
        treated, baseline = orient(records[0], records[1])
    except ValueError as error:
        console.print(str(error), style="red")
        raise typer.Exit(code=1) from None
    problems = check_pair_rows(treated, baseline)
    if problems:
        for problem in problems:
            console.print(problem, style="red")
        raise typer.Exit(code=1)
    spec = DATASETS.get(treated.dataset)
    if spec is None:
        console.print(f"unknown dataset {treated.dataset}", style="red")
        raise typer.Exit(code=1)
    since = window_start(
        min(
            datetime.fromisoformat(record.started_at) for record in (treated, baseline)
        ).isoformat()
    )
    results_dir = results or default_results_path()
    outcomes = {
        record.name: _outcomes(record, spec.pair_key, since, results_dir)
        for record in (treated, baseline)
    }
    result = pair_outcomes(
        spec.pair_key,
        treated.name,
        outcomes[treated.name],
        baseline.name,
        outcomes[baseline.name],
    )
    console.print(
        render(result, decision_rule=treated.decision_rule),
        soft_wrap=True,
        highlight=False,
        markup=False,
    )


@arms_app.command("complete")
def arms_complete(
    name: str,
    trace: str | None = typer.Option(
        None, "--trace", help="Trace id, when the run name alone does not find it."
    ),
    registry: Path | None = REGISTRY_OPTION,
    results: Path | None = RESULTS_OPTION,
) -> None:
    """Fill an arm's result fields from its result file or its trace; void it
    when neither exists."""
    store = _registry(registry)
    try:
        summary = complete_arm(
            store,
            name,
            query=query_logfire,
            trace_id=trace,
            results_dir=results or default_results_path(),
        )
    except ValueError as error:
        console.print(str(error), style="red")
        raise typer.Exit(code=1) from None
    record = store.get(name)
    assert record is not None
    if summary is None:
        console.print(f"{name} marked void: {record.void_reason}", style="red")
        raise typer.Exit(code=1)
    console.print(
        f"{name} completed from trace {record.trace_id}: {summary.cases} cases, "
        f"accuracy {_rate(summary.accuracy)}, cite rate {_rate(summary.cite_rate)}, "
        f"cited_map {_rate(summary.cited_map)}, aborts {summary.aborts}",
        soft_wrap=True,
        highlight=False,
    )


def _rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


@arms_app.command("export")
def arms_export(path: Path, registry: Path | None = REGISTRY_OPTION) -> None:
    """Write the registry as JSONL, one arm per line, diffable."""
    count = _registry(registry).export_jsonl(path)
    console.print(f"exported {count} arms to {path}")


@arms_app.command("import")
def arms_import(path: Path, registry: Path | None = REGISTRY_OPTION) -> None:
    """Upsert arms from a JSONL export."""
    count = _registry(registry).import_jsonl(path)
    console.print(f"imported {count} arms from {path}")


@app.command()
def download(
    dataset: str = typer.Argument(..., help="Dataset key or 'all' to download all."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing database."),
) -> None:
    """Download pre-built evaluation database from HuggingFace."""
    for spec in _resolve_datasets(dataset):
        download_dataset_db(spec, force=force)


@app.command()
def upload(
    dataset: str = typer.Argument(..., help="Dataset key or 'all' to upload all."),
) -> None:
    """Upload evaluation database to HuggingFace (maintainer only)."""
    for spec in _resolve_datasets(dataset):
        upload_dataset_db(spec)


if __name__ == "__main__":
    app()
