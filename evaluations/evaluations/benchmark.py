import asyncio

from pathlib import Path
from typing import cast

import typer
from dotenv import find_dotenv, load_dotenv
from rich.console import Console

from evaluations.artifacts import download_dataset_db, upload_dataset_db
from evaluations.config import DatasetSpec
from evaluations.population import populate_db
from evaluations.qa import TARGETS, Target, run_live_qa_benchmark, run_qa_benchmark
from evaluations.retrieval import run_retrieval_benchmark
from evaluations.datasets import DATASETS
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
    retrieval_limit: int | None = None,
    target: Target = "rag-capability",
    capability_model: ModelConfig | None = None,
    case_ids: set[str] | None = None,
    document_filter: str | None = None,
) -> None:
    if document_filter is not None:
        console.print(f"Document filter: {document_filter}", style="dim")

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
            retrieval_limit=retrieval_limit,
        )

    if not skip_qa:
        console.print(
            f"\nRunning QA benchmarks (target={target})...", style="bold yellow"
        )
        qa_benchmark = run_live_qa_benchmark if spec.live else run_qa_benchmark
        await qa_benchmark(
            spec,
            config,
            limit=limit,
            name=name,
            db_path=db_path,
            judge_model=judge_model,
            target=target,
            capability_model=capability_model,
            case_ids=case_ids,
            document_filter=document_filter,
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
    db: Path | None = typer.Option(None, "--db", help="Override the database path."),
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
    retrieval_limit: int | None = typer.Option(
        None,
        "--retrieval-limit",
        help=(
            "Candidates each database fetches, overriding the dataset's. "
            "Sets how deep hybrid search looks before its results are scored."
        ),
    ),
    vacuum_interval: int = typer.Option(
        100, "--vacuum-interval", help="Vacuum every N documents during DB population."
    ),
    multimodal_only: bool = typer.Option(
        False,
        "--multimodal-only",
        help="Only evaluate queries requiring image understanding.",
    ),
    target: str = typer.Option(
        "rag-capability",
        "--target",
        help="What to benchmark: rag-capability | analysis-capability.",
    ),
    capability_model: str | None = typer.Option(
        None,
        "--capability-model",
        help=(
            "Capability model as 'provider:name'. Defaults to qa.model (or "
            "analysis.model when --target is analysis-capability) from the config."
        ),
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
) -> None:
    spec = _resolve_dataset(dataset)
    app_config = _load_config(config)
    if target not in TARGETS:
        raise typer.BadParameter(
            f"Unknown target {target!r}. Choose from: {', '.join(TARGETS)}"
        )
    target_value = cast(Target, target)
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
            retrieval_limit=retrieval_limit,
            target=target_value,
            capability_model=capability_model_config,
            case_ids=_load_case_ids(filter_ids),
            document_filter=document_filter,
        )
    )


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
