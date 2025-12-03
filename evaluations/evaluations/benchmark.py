import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import logfire
import typer
from dotenv import load_dotenv
from pydantic_evals import Case, Dataset as EvalDataset
from pydantic_evals.evaluators import LLMJudge
from pydantic_evals.reporting import ReportCaseFailure
from rich.console import Console
from rich.progress import Progress

from evaluations.config import DatasetSpec
from evaluations.datasets import DATASETS
from evaluations.evaluators import ANSWER_EQUIVALENCE_RUBRIC
from evaluations.prompts import WIX_SUPPORT_PROMPT
from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, find_config_file, load_yaml_config
from haiku.rag.config.models import ModelConfig
from haiku.rag.logging import configure_cli_logging
from haiku.rag.qa import get_qa_agent
from haiku.rag.utils import get_model

load_dotenv()

logfire.configure(send_to_logfire="if-token-present", service_name="evals")
logfire.instrument_pydantic_ai()
configure_cli_logging()
console = Console()


def build_experiment_metadata(
    dataset_key: str,
    test_cases: int,
    config: AppConfig,
    judge_config: ModelConfig,
) -> dict[str, Any]:
    """Build experiment metadata for Logfire tracking."""
    return {
        "dataset": dataset_key,
        "test_cases": test_cases,
        "embedder_provider": config.embeddings.model.provider,
        "embedder_model": config.embeddings.model.name,
        "embedder_dim": config.embeddings.model.vector_dim,
        "chunk_size": config.processing.chunk_size,
        "context_chunk_radius": config.processing.context_chunk_radius,
        "rerank_provider": config.reranking.model.provider
        if config.reranking.model
        else None,
        "rerank_model": config.reranking.model.name if config.reranking.model else None,
        "qa_provider": config.qa.model.provider,
        "qa_model": config.qa.model.name,
        "qa_temperature": config.qa.model.temperature,
        "qa_max_tokens": config.qa.model.max_tokens,
        "qa_enable_thinking": config.qa.model.enable_thinking,
        "judge_provider": judge_config.provider,
        "judge_model": judge_config.name,
        "judge_temperature": judge_config.temperature,
        "judge_max_tokens": judge_config.max_tokens,
        "judge_enable_thinking": judge_config.enable_thinking,
    }


async def populate_db(
    spec: DatasetSpec, config: AppConfig, db_path: Path | None = None
) -> None:
    db = spec.db_path(db_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    corpus = spec.document_loader()
    if spec.document_limit is not None:
        corpus = corpus.select(range(min(spec.document_limit, len(corpus))))

    with Progress() as progress:
        task = progress.add_task("[green]Populating database...", total=len(corpus))
        async with HaikuRAG(db, config=config) as rag:
            for doc in corpus:
                doc_mapping = cast(Mapping[str, Any], doc)
                payload = spec.document_mapper(doc_mapping)
                if payload is None:
                    progress.advance(task)
                    continue

                existing = await rag.get_document_by_uri(payload.uri)
                if existing is not None:
                    assert existing.id
                    chunks = await rag.chunk_repository.get_by_document_id(existing.id)
                    if chunks:
                        progress.advance(task)
                        continue
                    await rag.document_repository.delete(existing.id)

                await rag.create_document(
                    content=payload.content,
                    uri=payload.uri,
                    title=payload.title,
                    metadata=payload.metadata,
                )
                progress.advance(task)


async def run_retrieval_benchmark(
    spec: DatasetSpec,
    config: AppConfig,
    limit: int | None = None,
    name: str | None = None,
    db_path: Path | None = None,
) -> dict[str, float] | None:
    if spec.retrieval_loader is None or spec.retrieval_mapper is None:
        console.print("Skipping retrieval benchmark; no retrieval config.")
        return None

    corpus = spec.retrieval_loader()
    if limit is not None:
        corpus = corpus.select(range(min(limit, len(corpus))))

    cases = []
    with Progress() as progress:
        task = progress.add_task("[blue]Building retrieval cases...", total=len(corpus))
        for doc in corpus:
            doc_mapping = cast(Mapping[str, Any], doc)
            sample = spec.retrieval_mapper(doc_mapping)
            if sample is None or sample.skip:
                progress.advance(task)
                continue

            case = Case(
                inputs=sample.question,
                metadata={"relevant_uris": sample.expected_uris},
            )
            cases.append(case)
            progress.advance(task)

    if not cases:
        console.print("No retrieval cases to evaluate.")
        return None

    if spec.retrieval_evaluator is None:
        raise ValueError(f"No retrieval evaluator configured for dataset: {spec.key}")

    evaluator = spec.retrieval_evaluator
    metric_name = evaluator.__class__.__name__.replace("Evaluator", "").upper()

    dataset = EvalDataset(
        cases=cases,
        evaluators=[evaluator],
    )

    db = spec.db_path(db_path)
    async with HaikuRAG(db, config=config) as rag:

        async def retrieval_target(question: str) -> list[str]:
            chunks = await rag.search(query=question, limit=5)

            seen = set()
            uris = []
            for chunk, _ in chunks:
                if chunk.document_id is None:
                    continue
                doc = await rag.get_document_by_id(chunk.document_id)
                if doc and doc.uri and doc.uri not in seen:
                    uris.append(doc.uri)
                    seen.add(doc.uri)

            return uris

        eval_name = name if name is not None else f"{spec.key}_retrieval_evaluation"

        judge_config = ModelConfig(
            provider="ollama", name="gpt-oss", enable_thinking=False
        )
        experiment_metadata = build_experiment_metadata(
            dataset_key=spec.key,
            test_cases=len(cases),
            config=config,
            judge_config=judge_config,
        )

        report = await dataset.evaluate(
            retrieval_target,
            name=eval_name,
            max_concurrency=1,
            progress=True,
            metadata=experiment_metadata,
        )

    total_score = 0.0
    total_cases = 0
    for case in report.cases:
        if case.scores:
            for score_result in case.scores.values():
                total_score += score_result.value
                total_cases += 1

    mean_score = total_score / total_cases if total_cases > 0 else 0.0

    console.print("\n=== Retrieval Benchmark Results ===", style="bold cyan")
    console.print(f"Dataset: {spec.key}")
    console.print(f"Total queries: {len(cases)}")
    console.print(f"{metric_name}: {mean_score:.4f}")

    return {
        metric_name.lower(): mean_score,
        "queries": len(cases),
    }


async def run_qa_benchmark(
    spec: DatasetSpec,
    config: AppConfig,
    limit: int | None = None,
    name: str | None = None,
    db_path: Path | None = None,
) -> ReportCaseFailure[str, str, dict[str, str]] | None:
    corpus = spec.qa_loader()
    if limit is not None:
        corpus = corpus.select(range(min(limit, len(corpus))))

    cases = [
        spec.qa_case_builder(index, cast(Mapping[str, Any], doc))
        for index, doc in enumerate(corpus, start=1)
    ]

    judge_config = ModelConfig(provider="ollama", name="gpt-oss", enable_thinking=False)
    judge_model = get_model(judge_config, config)

    evaluation_dataset = EvalDataset[str, str, dict[str, str]](
        name=spec.key,
        cases=cases,
        evaluators=[
            LLMJudge(
                rubric=ANSWER_EQUIVALENCE_RUBRIC,
                include_input=True,
                include_expected_output=True,
                model=judge_model,
                assertion={
                    "evaluation_name": "answer_equivalent",
                    "include_reason": True,
                },
            ),
        ],
    )

    db = spec.db_path(db_path)
    async with HaikuRAG(db, config=config) as rag:
        system_prompt = WIX_SUPPORT_PROMPT if spec.key == "wix" else None
        qa = get_qa_agent(rag, system_prompt=system_prompt)

        async def answer_question(question: str) -> str:
            return await qa.answer(question)

        eval_name = name if name is not None else f"{spec.key}_qa_evaluation"

        experiment_metadata = build_experiment_metadata(
            dataset_key=spec.key,
            test_cases=len(cases),
            config=config,
            judge_config=judge_config,
        )

        report = await evaluation_dataset.evaluate(
            answer_question,
            name=eval_name,
            max_concurrency=1,
            progress=True,
            metadata=experiment_metadata,
        )

    passing_cases = sum(
        1
        for case in report.cases
        if case.assertions.get("answer_equivalent")
        and case.assertions["answer_equivalent"].value
    )
    total_processed = len(report.cases)
    failures = report.failures

    total_cases = total_processed
    accuracy = passing_cases / total_cases if total_cases > 0 else 0

    console.print("\n=== QA Benchmark Results ===", style="bold cyan")
    console.print(f"Total questions: {total_cases}")
    console.print(f"Correct answers: {passing_cases}")
    console.print(f"QA Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    if failures:
        console.print("[red]\nSummary of failures:[/red]")
        for failure in failures:
            console.print(f"Case: {failure.name}")
            console.print(f"Question: {failure.inputs}")
            console.print(f"Error: {failure.error_message}")
            console.print("")

    return failures[0] if failures else None


async def evaluate_dataset(
    spec: DatasetSpec,
    config: AppConfig,
    skip_db: bool,
    skip_retrieval: bool,
    skip_qa: bool,
    limit: int | None,
    name: str | None,
    db_path: Path | None,
) -> None:
    if not skip_db:
        console.print(f"Using dataset: {spec.key}", style="bold magenta")
        await populate_db(spec, config, db_path=db_path)

    if not skip_retrieval:
        console.print("Running retrieval benchmarks...", style="bold blue")
        await run_retrieval_benchmark(
            spec, config, limit=limit, name=name, db_path=db_path
        )

    if not skip_qa:
        console.print("\nRunning QA benchmarks...", style="bold yellow")
        await run_qa_benchmark(spec, config, limit=limit, name=name, db_path=db_path)


app = typer.Typer(help="Run retrieval and QA benchmarks for configured datasets.")


@app.command()
def run(
    dataset: str = typer.Argument(..., help="Dataset key to evaluate."),
    config: Path | None = typer.Option(
        None, "--config", help="Path to haiku.rag YAML config file."
    ),
    db: Path | None = typer.Option(None, "--db", help="Override the database path."),
    skip_db: bool = typer.Option(
        False, "--skip-db", help="Skip updateing the evaluation db."
    ),
    skip_retrieval: bool = typer.Option(
        False, "--skip-retrieval", help="Skip retrieval benchmark."
    ),
    skip_qa: bool = typer.Option(False, "--skip-qa", help="Skip QA benchmark."),
    limit: int | None = typer.Option(
        None, "--limit", help="Limit number of test cases for both retrieval and QA."
    ),
    name: str | None = typer.Option(None, "--name", help="Override evaluation name."),
) -> None:
    spec = DATASETS.get(dataset.lower())
    if spec is None:
        valid_datasets = ", ".join(sorted(DATASETS))
        raise typer.BadParameter(
            f"Unknown dataset '{dataset}'. Choose from: {valid_datasets}"
        )

    # Load config from file or use defaults
    if config:
        if not config.exists():
            raise typer.BadParameter(f"Config file not found: {config}")
        console.print(f"Loading config from: {config}", style="dim")
        yaml_data = load_yaml_config(config)
        app_config = AppConfig.model_validate(yaml_data)
    else:
        # Try to find config file using standard search path
        config_path = find_config_file(None)
        if config_path:
            console.print(f"Loading config from: {config_path}", style="dim")
            yaml_data = load_yaml_config(config_path)
            app_config = AppConfig.model_validate(yaml_data)
        else:
            console.print("No config file found, using defaults", style="dim")
            app_config = AppConfig()

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
        )
    )


if __name__ == "__main__":
    app()
