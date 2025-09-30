import asyncio
from collections.abc import Mapping
from typing import Any, cast

import logfire
import typer
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_evals import Dataset as EvalDataset
from pydantic_evals.evaluators import IsInstance, LLMJudge
from pydantic_evals.reporting import ReportCaseFailure
from rich.console import Console
from rich.progress import Progress

from evaluations.config import DatasetSpec, RetrievalSample
from evaluations.datasets import DATASETS
from evaluations.llm_judge import ANSWER_EQUIVALENCE_RUBRIC
from haiku.rag import logging  # noqa: F401
from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.logging import configure_cli_logging
from haiku.rag.qa import get_qa_agent

QA_JUDGE_MODEL = "qwen3"

logfire.configure(send_to_logfire="if-token-present", service_name="evals")
logfire.instrument_pydantic_ai()
configure_cli_logging()
console = Console()


async def populate_db(spec: DatasetSpec) -> None:
    spec.db_path.parent.mkdir(parents=True, exist_ok=True)
    corpus = spec.document_loader()
    if spec.document_limit is not None:
        corpus = corpus.select(range(min(spec.document_limit, len(corpus))))

    with Progress() as progress:
        task = progress.add_task("[green]Populating database...", total=len(corpus))
        async with HaikuRAG(spec.db_path) as rag:
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
            rag.store.vacuum()


def _is_relevant_match(retrieved_uri: str | None, sample: RetrievalSample) -> bool:
    return retrieved_uri is not None and retrieved_uri in sample.expected_uris


async def run_retrieval_benchmark(spec: DatasetSpec) -> dict[str, float] | None:
    if spec.retrieval_loader is None or spec.retrieval_mapper is None:
        console.print("Skipping retrieval benchmark; no retrieval config.")
        return None

    corpus = spec.retrieval_loader()

    recall_totals = {
        1: 0.0,
        3: 0.0,
        5: 0.0,
    }
    total_queries = 0

    with Progress() as progress:
        task = progress.add_task(
            "[blue]Running retrieval benchmark...", total=len(corpus)
        )
        async with HaikuRAG(spec.db_path) as rag:
            for doc in corpus:
                doc_mapping = cast(Mapping[str, Any], doc)
                sample = spec.retrieval_mapper(doc_mapping)
                if sample is None or sample.skip:
                    progress.advance(task)
                    continue

                matches = await rag.search(query=sample.question, limit=5)
                if not matches:
                    progress.advance(task)
                    continue

                total_queries += 1

                retrieved_uris: list[str] = []
                for chunk, _ in matches:
                    if chunk.document_id is None:
                        continue
                    retrieved_doc = await rag.get_document_by_id(chunk.document_id)
                    if retrieved_doc and retrieved_doc.uri:
                        retrieved_uris.append(retrieved_doc.uri)

                # Compute per-query recall@K by counting how many relevant
                # documents are retrieved within the first K results and
                # averaging these fractions across all queries.
                for cutoff in (1, 3, 5):
                    top_k = set(retrieved_uris[:cutoff])
                    relevant = set(sample.expected_uris)
                    if relevant:
                        matched = len(top_k & relevant)
                        recall_totals[cutoff] += matched / len(relevant)

                progress.advance(task)

    if total_queries == 0:
        console.print("No retrieval cases to evaluate.")
        return None

    recall_at_1 = recall_totals[1] / total_queries
    recall_at_3 = recall_totals[3] / total_queries
    recall_at_5 = recall_totals[5] / total_queries

    console.print("\n=== Retrieval Benchmark Results ===", style="bold cyan")
    console.print(f"Total queries: {total_queries}")
    console.print(f"Recall@1: {recall_at_1:.4f}")
    console.print(f"Recall@3: {recall_at_3:.4f}")
    console.print(f"Recall@5: {recall_at_5:.4f}")

    return {
        "recall@1": recall_at_1,
        "recall@3": recall_at_3,
        "recall@5": recall_at_5,
    }


async def run_qa_benchmark(
    spec: DatasetSpec, qa_limit: int | None = None
) -> ReportCaseFailure[str, str, dict[str, str]] | None:
    corpus = spec.qa_loader()
    if qa_limit is not None:
        corpus = corpus.select(range(min(qa_limit, len(corpus))))

    cases = [
        spec.qa_case_builder(index, cast(Mapping[str, Any], doc))
        for index, doc in enumerate(corpus, start=1)
    ]

    judge_model = OpenAIChatModel(
        model_name=QA_JUDGE_MODEL,
        provider=OllamaProvider(base_url=f"{Config.OLLAMA_BASE_URL}/v1"),
    )

    evaluation_dataset = EvalDataset[str, str, dict[str, str]](
        cases=cases,
        evaluators=[
            IsInstance(type_name="str"),
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

    total_processed = 0
    passing_cases = 0
    failures: list[ReportCaseFailure[str, str, dict[str, str]]] = []

    with Progress(console=console) as progress:
        qa_task = progress.add_task(
            "[yellow]Evaluating QA cases...",
            total=len(evaluation_dataset.cases),
        )

        async with HaikuRAG(spec.db_path) as rag:
            qa = get_qa_agent(rag)

            async def answer_question(question: str) -> str:
                return await qa.answer(question)

            for case in evaluation_dataset.cases:
                progress.console.print(f"\n[bold]Evaluating case:[/bold] {case.name}")

                single_case_dataset = EvalDataset[str, str, dict[str, str]](
                    cases=[case],
                    evaluators=evaluation_dataset.evaluators,
                )

                report = await single_case_dataset.evaluate(
                    answer_question,
                    name="qa_answer",
                    max_concurrency=1,
                    progress=False,
                )

                total_processed += 1

                if report.cases:
                    result_case = report.cases[0]

                    equivalence = result_case.assertions.get("answer_equivalent")
                    progress.console.print(f"Question: {result_case.inputs}")
                    progress.console.print(f"Expected: {result_case.expected_output}")
                    progress.console.print(f"Generated: {result_case.output}")
                    if equivalence is not None:
                        progress.console.print(
                            f"Equivalent: {equivalence.value}"
                            + (f" — {equivalence.reason}" if equivalence.reason else "")
                        )
                        if equivalence.value:
                            passing_cases += 1

                    progress.console.print("")

                if report.failures:
                    failures.extend(report.failures)
                    failure = report.failures[0]
                    progress.console.print(
                        "[red]Failure encountered during case evaluation:[/red]"
                    )
                    progress.console.print(f"Question: {failure.inputs}")
                    progress.console.print(f"Error: {failure.error_message}")
                    progress.console.print("")

                progress.console.print(
                    f"[green]Accuracy: {(passing_cases / total_processed):.4f} "
                    f"{passing_cases}/{total_processed}[/green]"
                )
                progress.advance(qa_task)

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
    skip_db: bool,
    skip_retrieval: bool,
    skip_qa: bool,
    qa_limit: int | None,
) -> None:
    if not skip_db:
        console.print(f"Using dataset: {spec.key}", style="bold magenta")
        await populate_db(spec)

    if not skip_retrieval:
        console.print("Running retrieval benchmarks...", style="bold blue")
        await run_retrieval_benchmark(spec)

    if not skip_qa:
        console.print("\nRunning QA benchmarks...", style="bold yellow")
        await run_qa_benchmark(spec, qa_limit=qa_limit)


app = typer.Typer(help="Run retrieval and QA benchmarks for configured datasets.")


@app.command()
def run(
    dataset: str = typer.Argument(..., help="Dataset key to evaluate."),
    skip_db: bool = typer.Option(
        False, "--skip-db", help="Skip updateing the evaluation db."
    ),
    skip_retrieval: bool = typer.Option(
        False, "--skip-retrieval", help="Skip retrieval benchmark."
    ),
    skip_qa: bool = typer.Option(False, "--skip-qa", help="Skip QA benchmark."),
    qa_limit: int | None = typer.Option(
        None, "--qa-limit", help="Limit number of QA cases."
    ),
) -> None:
    spec = DATASETS.get(dataset.lower())
    if spec is None:
        valid_datasets = ", ".join(sorted(DATASETS))
        raise typer.BadParameter(
            f"Unknown dataset '{dataset}'. Choose from: {valid_datasets}"
        )

    asyncio.run(
        evaluate_dataset(
            spec=spec,
            skip_db=skip_db,
            skip_retrieval=skip_retrieval,
            skip_qa=skip_qa,
            qa_limit=qa_limit,
        )
    )


if __name__ == "__main__":
    app()
