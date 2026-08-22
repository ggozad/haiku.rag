"""Retrieval benchmark: search the corpus and score the ranking."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from pydantic_evals import Case, Dataset as EvalDataset
from rich.console import Console
from rich.progress import Progress

from evaluations.config import DatasetSpec
from evaluations.experiment import build_experiment_metadata
from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig

console = Console()


async def run_retrieval_benchmark(
    spec: DatasetSpec,
    config: AppConfig,
    limit: int | None = None,
    name: str | None = None,
    db_path: Path | None = None,
    multimodal_only: bool = False,
    document_filter: str | None = None,
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

            # Filter for multimodal queries if requested
            if multimodal_only:
                if sample.source_type is None or "image" not in sample.source_type:
                    progress.advance(task)
                    continue

            case = Case(
                inputs=sample.question,
                metadata={
                    "relevant_uris": sample.expected_uris,
                    "source_type": sample.source_type,
                },
            )
            cases.append(case)
            progress.advance(task)

    if not cases:
        console.print("No retrieval cases to evaluate.")
        return None

    if not spec.retrieval_evaluators:
        raise ValueError(f"No retrieval evaluators configured for dataset: {spec.key}")

    dataset = EvalDataset(
        name=f"{spec.key}-retrieval",
        cases=cases,
        evaluators=list(spec.retrieval_evaluators),
    )

    db = None if spec.covers_a_set(config, db_path) else spec.db_path(db_path)
    async with HaikuRAG(db, config=config, read_only=True) as rag:

        async def retrieval_target(question: str) -> list[str]:
            chunks = await rag.search(
                query=question,
                limit=spec.retrieval_limit,
                include_images=False,
                filter=document_filter,
            )

            seen = set()
            identifiers = []
            for result in chunks:
                uri = result.document_uri
                if uri and uri not in seen:
                    identifiers.append(uri)
                    seen.add(uri)

            return identifiers

        eval_name = name if name is not None else f"{spec.key}_retrieval_evaluation"

        experiment_metadata = build_experiment_metadata(
            dataset_key=spec.key,
            test_cases=len(cases),
            config=config,
            document_filter=document_filter,
        )

        report = await dataset.evaluate(
            retrieval_target,
            name=eval_name,
            max_concurrency=1,
            progress=True,
            metadata=experiment_metadata,
        )

    per_metric: dict[str, list[float]] = {}
    for case in report.cases:
        for key, score_result in case.scores.items():
            per_metric.setdefault(key, []).append(score_result.value)

    console.print("\n=== Retrieval Benchmark Results ===", style="bold cyan")
    console.print(f"Dataset: {spec.key}")
    console.print(f"Total queries: {len(cases)}")
    results: dict[str, float] = {"queries": len(cases)}
    for key, values in per_metric.items():
        mean_score = sum(values) / len(values)
        metric_name = key.replace("Evaluator", "").upper()
        console.print(f"{metric_name}: {mean_score:.4f}")
        results[metric_name.lower()] = mean_score

    return results
