import asyncio
from pathlib import Path

import logfire
from datasets import Dataset, load_dataset
from llm_judge import ANSWER_EQUIVALENCE_RUBRIC
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_evals import Case
from pydantic_evals import Dataset as EvalDataset
from pydantic_evals.evaluators import IsInstance, LLMJudge
from pydantic_evals.reporting import ReportCaseFailure
from rich.console import Console
from rich.progress import Progress

from haiku.rag import logging  # noqa
from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.logging import configure_cli_logging
from haiku.rag.qa import get_qa_agent

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()
configure_cli_logging()
console = Console()

QA_JUDGE_MODEL: str = "qwen3"
db_path = Path(__file__).parent / "data" / "benchmark.lancedb"


async def populate_db():
    ds: Dataset = load_dataset("ServiceNow/repliqa")["repliqa_3"]  # type: ignore
    corpus = ds.filter(lambda doc: doc["document_topic"] == "News Stories")

    with Progress() as progress:
        task = progress.add_task("[green]Populating database...", total=len(corpus))

        async with HaikuRAG(db_path) as rag:
            for doc in corpus:
                uri = doc["document_id"]  # type: ignore
                existing_doc = await rag.get_document_by_uri(uri)
                if existing_doc is not None:
                    progress.advance(task)
                    continue

                await rag.create_document(
                    content=doc["document_extracted"],  # type: ignore
                    uri=uri,
                )
                progress.advance(task)
            rag.store.vacuum()


async def run_match_benchmark():
    ds: Dataset = load_dataset("ServiceNow/repliqa")["repliqa_3"]  # type: ignore
    corpus = ds.filter(lambda doc: doc["document_topic"] == "News Stories")

    correct_at_1 = 0
    correct_at_2 = 0
    correct_at_3 = 0
    total_queries = 0

    with Progress() as progress:
        task = progress.add_task(
            "[blue]Running retrieval benchmark...", total=len(corpus)
        )

        async with HaikuRAG(db_path) as rag:
            for doc in corpus:
                doc_id = doc["document_id"]  # type: ignore
                expected_answer = doc["answer"]  # type: ignore
                if expected_answer == "The answer is not found in the document.":
                    progress.advance(task)
                    continue
                matches = await rag.search(
                    query=doc["question"],  # type: ignore
                    limit=3,
                )

                total_queries += 1

                # Check position of correct document in results
                for position, (chunk, _) in enumerate(matches):
                    assert chunk.document_id is not None, (
                        "Chunk document_id should not be None"
                    )
                    retrieved = await rag.get_document_by_id(chunk.document_id)
                    if retrieved and retrieved.uri == doc_id:
                        if position == 0:  # First position
                            correct_at_1 += 1
                            correct_at_2 += 1
                            correct_at_3 += 1
                        elif position == 1:  # Second position
                            correct_at_2 += 1
                            correct_at_3 += 1
                        elif position == 2:  # Third position
                            correct_at_3 += 1
                        break

                progress.advance(task)

    # Calculate recall metrics
    recall_at_1 = correct_at_1 / total_queries
    recall_at_2 = correct_at_2 / total_queries
    recall_at_3 = correct_at_3 / total_queries

    console.print("\n=== Retrieval Benchmark Results ===", style="bold cyan")
    console.print(f"Total queries: {total_queries}")
    console.print(f"Recall@1: {recall_at_1:.4f}")
    console.print(f"Recall@2: {recall_at_2:.4f}")
    console.print(f"Recall@3: {recall_at_3:.4f}")

    return {"recall@1": recall_at_1, "recall@2": recall_at_2, "recall@3": recall_at_3}


async def run_qa_benchmark(k: int | None = None):
    """Run QA benchmarking on the corpus."""
    ds: Dataset = load_dataset("ServiceNow/repliqa")["repliqa_3"]  # type: ignore
    corpus = ds.filter(lambda doc: doc["document_topic"] == "News Stories")

    if k is not None:
        corpus = corpus.select(range(min(k, len(corpus))))

    cases: list[Case[str, str, dict[str, str]]] = []
    for index, doc in enumerate(corpus, start=1):
        question = doc["question"]  # type: ignore[index]
        expected_answer = doc["answer"]  # type: ignore[index]
        doc_id = doc["document_id"]  # type: ignore[index]
        case_name = f"{index}_{doc_id}" if doc_id is not None else f"case_{index}"

        cases.append(
            Case(
                name=case_name,
                inputs=question,
                expected_output=expected_answer,
                metadata={
                    "document_id": str(doc_id),
                    "case_index": str(index),
                },
            )
        )

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

    console.print("[yellow]Running QA benchmark...[/yellow]")

    total_processed = 0
    passing_cases = 0
    failures: list[ReportCaseFailure[str, str, dict[str, str]]] = []

    async with HaikuRAG(db_path) as rag:
        qa = get_qa_agent(rag)

        async def answer_question(question: str) -> str:
            return await qa.answer(question)

        for case in evaluation_dataset.cases:
            console.print(f"\n[bold]Evaluating case:[/bold] {case.name}")

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
                console.print(f"Question: {result_case.inputs}")
                console.print(f"Expected: {result_case.expected_output}")
                console.print(f"Generated: {result_case.output}")
                if equivalence is not None:
                    console.print(
                        f"Equivalent: {equivalence.value}"
                        + (f" — {equivalence.reason}" if equivalence.reason else "")
                    )
                    if equivalence.value:
                        passing_cases += 1
                console.print("")

            if report.failures:
                failures.extend(report.failures)
                failure = report.failures[0]
                console.print("[red]Failure encountered during case evaluation:[/red]")
                console.print(f"Question: {failure.inputs}")
                console.print(f"Error: {failure.error_message}")
                console.print("")

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


async def main():
    await populate_db()

    console.print("Running retrieval benchmarks...", style="bold blue")
    await run_match_benchmark()

    console.print("\nRunning QA benchmarks...", style="bold yellow")
    await run_qa_benchmark()


if __name__ == "__main__":
    asyncio.run(main())
