import asyncio
import shutil
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any, Literal, cast

import typer
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import HfApi, snapshot_download
from pydantic_evals import Case, Dataset as EvalDataset, set_eval_attribute
from pydantic_evals.evaluators import Evaluator
from pydantic_evals.reporting import ReportCaseFailure
from rich.console import Console
from rich.progress import Progress

from evaluations.config import ConversationInput, DatasetSpec
from evaluations.datasets import DATASETS
from evaluations.evaluators import (
    ANSWER_EQUIVALENCE_RUBRIC,
    REFUSAL_RUBRIC,
    ConversationEvaluator,
    RefusalJudge,
    TranscriptLLMJudge,
)
from evaluations.capability_runner import (
    CapabilityFactory,
    prefix_to_messages,
    run_capability_conversation,
    run_capability_question,
)
from haiku.rag.client import HaikuRAG
from haiku.rag.client.documents import DocumentImport
from haiku.rag.config import AppConfig, find_config_file, load_yaml_config
from haiku.rag.config.models import ModelConfig
from haiku.rag.logging import configure_cli_logging
from haiku.rag.telemetry import configure as configure_telemetry
from haiku.rag.utils import get_model, parse_model_option

Target = Literal["rag-capability", "analysis-capability"]
TARGETS: tuple[Target, ...] = ("rag-capability", "analysis-capability")

# Pinned judge model. Decoupled from `config.qa.model` so a user changing
# their QA model does not inadvertently change the judge — keeps cross-run
# comparisons stable. Override per-run with `--judge-model provider:name`.
DEFAULT_JUDGE_MODEL = ModelConfig(provider="ollama", name="qwen3.6")

load_dotenv(find_dotenv(usecwd=True))

HF_REPO_ID = "ggozad/haiku-rag-eval-dbs"

# Scrubbing off: eval outputs are financial answers with words like "authorized"
# that trip Logfire's secret scrubber and redact the model's answer text.
configure_telemetry(service_name="evals", scrubbing=False)
configure_cli_logging()
console = Console()


def build_experiment_metadata(
    dataset_key: str,
    test_cases: int,
    config: AppConfig,
    judge_config: ModelConfig | None = None,
    target: Target = "rag-capability",
    capability_config: ModelConfig | None = None,
) -> dict[str, Any]:
    """Build experiment metadata for Logfire tracking."""
    metadata: dict[str, Any] = {
        "dataset": dataset_key,
        "test_cases": test_cases,
        "target": target,
        "embedder_provider": config.embeddings.model.provider,
        "embedder_model": config.embeddings.model.name,
        "embedder_dim": config.embeddings.model.vector_dim,
        "chunk_size": config.processing.chunk_size,
        "search_limit": config.search.limit,
        "max_context_chars": config.search.max_context_chars,
        "rerank_provider": config.reranking.model.provider
        if config.reranking.model
        else None,
        "rerank_model": config.reranking.model.name if config.reranking.model else None,
        "qa_provider": config.qa.model.provider,
        "qa_model": config.qa.model.name,
        "qa_temperature": config.qa.model.temperature,
        "qa_max_tokens": config.qa.model.max_tokens,
        "qa_enable_thinking": config.qa.model.enable_thinking,
        "qa_max_searches": config.qa.max_searches,
    }
    if judge_config is not None:
        metadata.update(
            {
                "judge_provider": judge_config.provider,
                "judge_model": judge_config.name,
                "judge_temperature": judge_config.temperature,
                "judge_max_tokens": judge_config.max_tokens,
                "judge_enable_thinking": judge_config.enable_thinking,
            }
        )
    if capability_config is not None:
        metadata.update(
            {
                "capability_provider": capability_config.provider,
                "capability_model": capability_config.name,
                "capability_temperature": capability_config.temperature,
                "capability_max_tokens": capability_config.max_tokens,
                "capability_enable_thinking": capability_config.enable_thinking,
            }
        )
    return metadata


async def _ingest_batched(
    rag: HaikuRAG,
    spec: DatasetSpec,
    corpus,
    batch_size: int,
    on_document: Callable[[], None] = lambda: None,
) -> None:
    """Ingest inline-content documents via `import_documents` batches.

    Each batch writes the documents/chunks/document_items tables once and
    embeds every chunk in one batched pass. A URI is skipped on resume only
    when its document has chunks; a chunkless document (crash between the
    document and chunk writes) is deleted and re-imported.
    """
    uri_rows = await (
        rag.store.document_meta_table.query().select(["id", "uri"]).to_list()
    )
    chunk_rows = await rag.store.chunks_table.query().select(["document_id"]).to_list()
    chunked_ids = {row["document_id"] for row in chunk_rows}
    complete = {row["uri"] for row in uri_rows if row["id"] in chunked_ids}
    chunkless = {
        row["uri"]: row["id"] for row in uri_rows if row["id"] not in chunked_ids
    }

    batch: list[DocumentImport] = []
    for doc in corpus:
        payload = spec.document_mapper(cast(Mapping[str, Any], doc))
        if payload is None or payload.uri in complete:
            on_document()
            continue
        if payload.uri in chunkless:
            await rag.delete_document(chunkless[payload.uri])
        assert payload.content is not None, "batched ingest requires inline content"
        docling_document = await rag.convert(payload.content, format=payload.format)
        chunks = await rag.chunk(docling_document)
        batch.append(
            DocumentImport(
                docling_document=docling_document,
                chunks=chunks,
                uri=payload.uri,
                title=payload.title,
                metadata=payload.metadata or {},
            )
        )
        if len(batch) >= batch_size:
            await rag.import_documents(batch)
            batch = []
        on_document()

    if batch:
        await rag.import_documents(batch)


async def populate_db(
    spec: DatasetSpec,
    config: AppConfig,
    db_path: Path | None = None,
    vacuum_interval: int = 100,
) -> None:
    db = spec.db_path(db_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    corpus = spec.document_loader()
    if spec.document_limit is not None:
        corpus = corpus.select(range(min(spec.document_limit, len(corpus))))

    # Disable auto_vacuum - we'll vacuum periodically instead to prevent disk exhaustion
    config.storage.auto_vacuum = False

    with Progress() as progress:
        task = progress.add_task("[green]Populating database...", total=len(corpus))
        async with HaikuRAG(db, config=config, create=True) as rag:
            if spec.ingest_batch_size is not None:
                await _ingest_batched(
                    rag,
                    spec,
                    corpus,
                    batch_size=spec.ingest_batch_size,
                    on_document=lambda: progress.advance(task),
                )
                await rag.store.vacuum(retention_seconds=0)
                return

            docs_since_vacuum = 0
            for doc in corpus:
                doc_mapping = cast(Mapping[str, Any], doc)
                payload = spec.document_mapper(doc_mapping)
                if payload is None:
                    progress.advance(task)
                    continue

                # `payload.uri` is the canonical document identifier and is now
                # honored by both `create_document` and (via the `uri=` override)
                # `create_document_from_source`, so it's also the right key to
                # look up an existing document, regardless of whether the source
                # is a file path or inline content.
                existing = await rag.get_document_by_uri(payload.uri)
                if existing is not None:
                    assert existing.id
                    chunks = await rag.chunk_repository.get_by_document_id(existing.id)
                    if chunks:
                        progress.advance(task)
                        continue
                    await rag.document_repository.delete(existing.id)

                if payload.source_path is not None:
                    await rag.create_document_from_source(
                        source=payload.source_path,
                        title=payload.title,
                        metadata=payload.metadata,
                        uri=payload.uri,
                    )
                else:
                    assert payload.content is not None
                    await rag.create_document(
                        content=payload.content,
                        uri=payload.uri,
                        title=payload.title,
                        metadata=payload.metadata,
                        format=payload.format,
                    )
                docs_since_vacuum += 1
                progress.advance(task)

                # Periodic vacuum to prevent disk exhaustion
                if docs_since_vacuum >= vacuum_interval:
                    await rag.store.vacuum(retention_seconds=0)
                    docs_since_vacuum = 0

            # Final vacuum
            await rag.store.vacuum(retention_seconds=0)


async def run_retrieval_benchmark(
    spec: DatasetSpec,
    config: AppConfig,
    limit: int | None = None,
    name: str | None = None,
    db_path: Path | None = None,
    multimodal_only: bool = False,
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

    db = spec.db_path(db_path)
    async with HaikuRAG(db, config=config) as rag:

        async def retrieval_target(question: str) -> list[str]:
            chunks = await rag.search(query=question, limit=spec.retrieval_limit)

            seen = set()
            identifiers = []
            for result in chunks:
                if result.document_id is None:
                    continue
                doc = await rag.get_document_by_id(result.document_id)
                if doc is None:
                    continue
                # Use arxiv_id from metadata if present, otherwise use URI
                doc_id = doc.metadata.get("arxiv_id") if doc.metadata else None
                if doc_id is None:
                    doc_id = doc.uri
                if doc_id and doc_id not in seen:
                    identifiers.append(doc_id)
                    seen.add(doc_id)

            return identifiers

        eval_name = name if name is not None else f"{spec.key}_retrieval_evaluation"

        experiment_metadata = build_experiment_metadata(
            dataset_key=spec.key,
            test_cases=len(cases),
            config=config,
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


def _capability_factory_for_target(target: Target) -> CapabilityFactory:
    if target == "rag-capability":
        from haiku.rag.capabilities.rag import create_capability

        return create_capability
    if target == "analysis-capability":
        from haiku.rag.capabilities.analysis import create_capability

        return create_capability
    raise ValueError(f"target {target!r} is not a capability target")


def _attach_relevant_uris(
    cases: list[Case[str, str, dict[str, Any]]],
    spec: DatasetSpec,
    limit: int | None,
) -> None:
    """Augment QA cases with `relevant_uris` joined from retrieval samples.

    Mutates each case's metadata in place. Cases with no matching retrieval
    sample (by question) are left untouched.
    """
    if spec.retrieval_loader is None or spec.retrieval_mapper is None:
        return
    corpus = spec.retrieval_loader()
    if limit is not None:
        corpus = corpus.select(range(min(limit, len(corpus))))
    expected_by_question: dict[str, tuple[str, ...]] = {}
    for raw in corpus:
        sample = spec.retrieval_mapper(cast(Mapping[str, Any], raw))
        if sample is None or sample.skip:
            continue
        expected_by_question[sample.question] = sample.expected_uris
    for case in cases:
        if not isinstance(case.inputs, str):
            continue
        uris = expected_by_question.get(case.inputs)
        if uris is None:
            continue
        metadata = case.metadata if case.metadata is not None else {}
        metadata["relevant_uris"] = list(uris)
        case.metadata = metadata


def _resolve_capability_config(
    target: Target, config: AppConfig, capability_model: ModelConfig | None
) -> ModelConfig:
    if target == "analysis-capability":
        # Mirror the capability-code resolver: explicit analysis.model wins,
        # else fall back to qa.model.
        return capability_model or config.analysis.model or config.qa.model
    return capability_model or config.qa.model


def _live_summary(report_cases, report_failures=()) -> dict[str, float | int] | None:
    """Aggregate ConversationEvaluator scores across conversations.

    Micro rates weight every turn equally (sums across conversations); macro
    rates average per-conversation means, so short conversations don't get
    overweighted by micro nor long ones by macro. Failed conversations are
    operational exclusions: they count toward the attempted coverage figures
    but never toward the rates.
    """

    def _score(case, key: str):
        result = case.scores.get(key)
        return result.value if result is not None else None

    scored = [case for case in report_cases if _score(case, "turns_total") is not None]
    if not scored:
        return None

    failed_turns = sum(
        len(failure.inputs) if isinstance(failure.inputs, list) else 0
        for failure in report_failures
    )
    turns_total = sum(_score(case, "turns_total") for case in scored)
    turns_judged = sum(_score(case, "turns_judged") or 0 for case in scored)
    turns_passed = sum(_score(case, "turns_passed") for case in scored)
    summary: dict[str, float | int] = {
        "conversations": len(scored),
        "conversations_attempted": len(report_cases) + len(report_failures),
        "turns_total": turns_total,
        "turns_judged": turns_judged,
        "turns_attempted": turns_total + failed_turns,
        "micro_pass_rate": turns_passed / turns_judged if turns_judged else 0.0,
        "macro_pass_rate": sum(_score(case, "turn_pass_rate") for case in scored)
        / len(scored),
    }

    cited = [case for case in scored if _score(case, "cited_map") is not None]
    eligible = sum(_score(case, "cited_eligible") for case in scored)
    if cited and eligible:
        summary["cited_eligible"] = eligible
        summary["cited_map_micro"] = (
            sum(
                _score(case, "cited_map") * _score(case, "cited_eligible")
                for case in cited
            )
            / eligible
        )
        summary["cited_map_macro"] = sum(
            _score(case, "cited_map") for case in cited
        ) / len(cited)

    true_refusals = sum(_score(case, "true_refusals") or 0 for case in scored)
    false_refusals = sum(_score(case, "false_refusals") or 0 for case in scored)
    unanswerable = sum(_score(case, "unanswerable_turns") or 0 for case in scored)
    refusals = true_refusals + false_refusals
    summary["unanswerable_turns"] = unanswerable
    summary["refusals"] = refusals
    summary["refusal_precision"] = true_refusals / refusals if refusals else 0.0
    summary["refusal_recall"] = true_refusals / unanswerable if unanswerable else 0.0
    return summary


def _refusal_metrics(report_cases) -> tuple[float, float, int, int] | None:
    """Refusal precision/recall against answerability labels.

    Uses cases the refusal judge scored (ANSWERABLE/UNANSWERABLE turns).
    Returns (precision, recall, unanswerable_count, refusal_count), or None
    when no case was judged.
    """
    outcomes: list[tuple[str, bool]] = []
    for case in report_cases:
        refused = case.assertions.get("refused")
        label = (case.metadata or {}).get("answerability")
        if refused is None or label not in ("ANSWERABLE", "UNANSWERABLE"):
            continue
        outcomes.append((label, bool(refused.value)))
    if not outcomes:
        return None
    refusals = [(label, r) for label, r in outcomes if r]
    true_refusals = sum(1 for label, _ in refusals if label == "UNANSWERABLE")
    unanswerable = sum(1 for label, _ in outcomes if label == "UNANSWERABLE")
    precision = true_refusals / len(refusals) if refusals else 0.0
    recall = true_refusals / unanswerable if unanswerable else 0.0
    return precision, recall, unanswerable, len(refusals)


def _filter_qa_corpus(corpus, case_ids: set[str] | None):
    """Keep only rows whose ``id`` is in ``case_ids`` (failure-subset reruns).

    Returns the corpus unchanged when ``case_ids`` is None.
    """
    if case_ids is None:
        return corpus
    return corpus.filter(lambda row: row.get("id") in case_ids)


async def run_qa_benchmark(
    spec: DatasetSpec,
    config: AppConfig,
    limit: int | None = None,
    name: str | None = None,
    db_path: Path | None = None,
    judge_model: ModelConfig | None = None,
    target: Target = "rag-capability",
    capability_model: ModelConfig | None = None,
    case_ids: set[str] | None = None,
) -> ReportCaseFailure[str, str, dict[str, str]] | None:
    corpus = spec.qa_loader()
    corpus = _filter_qa_corpus(corpus, case_ids)
    if limit is not None:
        corpus = corpus.select(range(min(limit, len(corpus))))

    cases = [
        spec.qa_case_builder(index, cast(Mapping[str, Any], doc))
        for index, doc in enumerate(corpus, start=1)
    ]

    judge_config = judge_model or DEFAULT_JUDGE_MODEL
    capability_config = _resolve_capability_config(target, config, capability_model)
    db = spec.db_path(db_path)

    _attach_relevant_uris(cases, spec, limit)
    citation_evaluator = spec.citation_evaluator

    qa_evaluator = spec.qa_evaluator
    evaluators: list[Evaluator]
    if qa_evaluator is not None:
        evaluators = [qa_evaluator]
    else:
        evaluators = [
            TranscriptLLMJudge(
                rubric=ANSWER_EQUIVALENCE_RUBRIC,
                include_input=True,
                include_expected_output=True,
                model=get_model(judge_config, config),
                assertion={
                    "evaluation_name": "answer_equivalent",
                    "include_reason": True,
                },
            ),
        ]
    if citation_evaluator is not None:
        evaluators.append(citation_evaluator)
    if spec.evaluate_refusal:
        evaluators.append(
            RefusalJudge(
                rubric=REFUSAL_RUBRIC,
                model=get_model(judge_config, config),
                assertion={"evaluation_name": "refused", "include_reason": False},
            )
        )

    evaluation_dataset = EvalDataset[Any, str, dict[str, Any]](
        name=spec.key, cases=cases, evaluators=evaluators
    )

    eval_name = name if name is not None else f"{spec.key}_qa_evaluation"
    experiment_metadata = build_experiment_metadata(
        dataset_key=spec.key,
        test_cases=len(cases),
        config=config,
        judge_config=judge_config,
        target=target,
        capability_config=capability_config,
    )
    experiment_metadata.update(spec.experiment_metadata or {})

    async def _evaluate(answer_fn: Callable[[Any], Awaitable[str]]):
        return await evaluation_dataset.evaluate(
            answer_fn,
            name=eval_name,
            max_concurrency=1,
            progress=True,
            metadata=experiment_metadata,
        )

    capability_factory = _capability_factory_for_target(target)
    resolved_capability_model = get_model(capability_config, config)

    async def answer_question(inputs: str | ConversationInput) -> str:
        if isinstance(inputs, ConversationInput):
            question = inputs.question
            message_history = prefix_to_messages(inputs.prefix)
        else:
            question = inputs
            message_history = None
        result = await run_capability_question(
            capability_factory=capability_factory,
            db_path=db,
            config=config,
            question=question,
            capability_model=resolved_capability_model,
            message_history=message_history,
        )
        set_eval_attribute("cited_uris", result.cited_uris)
        return result.answer

    report = await _evaluate(answer_question)

    total_processed = len(report.cases)
    failures = report.failures
    if qa_evaluator is not None:
        score_key = qa_evaluator.get_default_evaluation_name()
        passing_cases = sum(
            1
            for case in report.cases
            if score_key in case.scores and case.scores[score_key].value >= 1.0
        )
        scoring = score_key
    else:
        passing_cases = sum(
            1
            for case in report.cases
            if case.assertions.get("answer_equivalent")
            and case.assertions["answer_equivalent"].value
        )
        scoring = "answer_equivalent"
    accuracy = passing_cases / total_processed if total_processed > 0 else 0

    console.print("\n=== QA Benchmark Results ===", style="bold cyan")
    console.print(f"Scoring: {scoring}")
    console.print(f"Total questions: {total_processed}")
    console.print(f"Correct answers: {passing_cases}")
    console.print(f"QA Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    if report.cases:
        mean_task_time = sum(case.task_duration for case in report.cases) / len(
            report.cases
        )
        console.print(f"Avg task time per case: {mean_task_time:.2f}s")

    if citation_evaluator is not None:
        score_key = citation_evaluator.get_default_evaluation_name()
        scores = [
            case.scores[score_key].value
            for case in report.cases
            if score_key in case.scores
        ]
        if scores:
            cited_count = sum(
                1 for case in report.cases if case.attributes.get("cited_uris")
            )
            mean_citations = sum(
                len(case.attributes.get("cited_uris") or []) for case in report.cases
            ) / len(report.cases)
            mean_score = sum(scores) / len(scores)
            console.print(
                f"\n=== Citation Retrieval ({score_key}) ===", style="bold cyan"
            )
            console.print(f"Mean {score_key}: {mean_score:.4f}")
            console.print(
                f"Eligible cases (gold passages known): {len(scores)}/{len(report.cases)}"
            )
            console.print(
                f"Cite rate (≥1 citation): {cited_count / len(report.cases):.2%}"
            )
            console.print(f"Mean citations per case: {mean_citations:.2f}")

    if spec.evaluate_refusal:
        metrics = _refusal_metrics(report.cases)
        if metrics is not None:
            precision, recall, unanswerable, refusals = metrics
            console.print(
                "\n=== Refusal vs answerability labels ===", style="bold cyan"
            )
            console.print(f"Refusal precision: {precision:.2%} | recall: {recall:.2%}")
            console.print(
                f"UNANSWERABLE turns: {unanswerable} | refusals: {refusals} "
                "(PARTIAL excluded)"
            )

    if failures:
        console.print("[red]\nSummary of failures:[/red]")
        for failure in failures:
            console.print(f"Case: {failure.name}")
            console.print(f"Question: {failure.inputs}")
            console.print(f"Error: {failure.error_message}")
            console.print("")

    return failures[0] if failures else None


async def run_live_qa_benchmark(
    spec: DatasetSpec,
    config: AppConfig,
    limit: int | None = None,
    name: str | None = None,
    db_path: Path | None = None,
    judge_model: ModelConfig | None = None,
    target: Target = "rag-capability",
    capability_model: ModelConfig | None = None,
    case_ids: set[str] | None = None,
) -> None:
    """Replay conversations turn by turn through one capability session.

    One case per conversation; ``limit`` counts conversations. Answers carry
    forward as real message history, so prior-turn compaction is exercised.
    """
    corpus = spec.qa_loader()
    corpus = _filter_qa_corpus(corpus, case_ids)
    if limit is not None:
        corpus = corpus.select(range(min(limit, len(corpus))))

    cases = [
        spec.qa_case_builder(index, cast(Mapping[str, Any], doc))
        for index, doc in enumerate(corpus, start=1)
    ]

    judge_config = judge_model or DEFAULT_JUDGE_MODEL
    capability_config = _resolve_capability_config(target, config, capability_model)
    db = spec.db_path(db_path)

    evaluation_dataset = EvalDataset[Any, Any, dict[str, Any]](
        name=spec.key,
        cases=cases,
        evaluators=[
            ConversationEvaluator(
                rubric=ANSWER_EQUIVALENCE_RUBRIC,
                model=get_model(judge_config, config),
            )
        ],
    )

    eval_name = name if name is not None else f"{spec.key}_qa_evaluation"
    experiment_metadata = build_experiment_metadata(
        dataset_key=spec.key,
        test_cases=len(cases),
        config=config,
        judge_config=judge_config,
        target=target,
        capability_config=capability_config,
    )
    experiment_metadata.update(spec.experiment_metadata or {})

    capability_factory = _capability_factory_for_target(target)
    resolved_capability_model = get_model(capability_config, config)

    async def answer_conversation(questions: list[str]) -> list[str]:
        results = await run_capability_conversation(
            capability_factory=capability_factory,
            db_path=db,
            config=config,
            questions=list(questions),
            capability_model=resolved_capability_model,
        )
        set_eval_attribute("turn_cited_uris", [r.cited_uris for r in results])
        return [r.answer for r in results]

    report = await evaluation_dataset.evaluate(
        answer_conversation,
        name=eval_name,
        max_concurrency=1,
        progress=True,
        metadata=experiment_metadata,
    )

    summary = _live_summary(report.cases, report.failures)
    console.print("\n=== Live Conversation Results ===", style="bold cyan")
    if summary is None:
        attempted = len(report.cases) + len(report.failures)
        console.print(f"No conversations were scored ({attempted} attempted).")
    else:
        console.print(
            f"Conversations scored: {summary['conversations']}"
            f"/{summary['conversations_attempted']} | turns scored: "
            f"{summary['turns_total']}/{summary['turns_attempted']}"
        )
        if summary["turns_judged"] < summary["turns_total"]:
            console.print(
                f"Turns judged: {summary['turns_judged']}/{summary['turns_total']} "
                "(per-turn judge errors excluded from rates)"
            )
        if report.failures:
            console.print(
                "Failed conversations are operational exclusions — "
                "not counted as wrong answers."
            )
        console.print(
            f"Answer pass rate — micro (per turn): {summary['micro_pass_rate']:.4f} | "
            f"macro (per conversation): {summary['macro_pass_rate']:.4f}"
        )
        if "cited_map_micro" in summary:
            console.print(
                f"cited_map — micro: {summary['cited_map_micro']:.4f} | "
                f"macro: {summary['cited_map_macro']:.4f} "
                f"(eligible turns: {summary['cited_eligible']})"
            )
        console.print(
            f"Refusal precision: {summary['refusal_precision']:.2%} | "
            f"recall: {summary['refusal_recall']:.2%} "
            f"(UNANSWERABLE turns: {summary['unanswerable_turns']}, "
            f"refusals: {summary['refusals']})"
        )
    if report.cases:
        mean_task_time = sum(case.task_duration for case in report.cases) / len(
            report.cases
        )
        turns = sum(len(case.output or []) for case in report.cases)
        per_turn = (
            sum(case.task_duration for case in report.cases) / turns if turns else 0.0
        )
        console.print(
            f"Avg task time: {mean_task_time:.2f}s per conversation | "
            f"{per_turn:.2f}s per turn"
        )

    if report.failures:
        console.print("[red]\nSummary of failures:[/red]")
        for failure in report.failures:
            console.print(f"Case: {failure.name}")
            console.print(f"Error: {failure.error_message}")
            console.print("")


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
    target: Target = "rag-capability",
    capability_model: ModelConfig | None = None,
    case_ids: set[str] | None = None,
) -> None:
    if not skip_db:
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
            target=target_value,
            capability_model=capability_model_config,
            case_ids=_load_case_ids(filter_ids),
        )
    )


@app.command()
def download(
    dataset: str = typer.Argument(..., help="Dataset key or 'all' to download all."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing database."),
) -> None:
    """Download pre-built evaluation database from HuggingFace."""
    specs = _resolve_datasets(dataset)

    for spec in specs:
        db = spec.db_path()
        if db.exists() and not force:
            console.print(
                f"[yellow]Skipping {spec.key}: database already exists at {db}[/yellow]"
            )
            console.print("Use --force to overwrite.")
            continue

        console.print(f"[blue]Downloading {spec.key}...[/blue]")

        try:
            downloaded_path = snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                allow_patterns=f"{spec.db_filename}/*",
            )
        except Exception as e:
            console.print(f"[red]Failed to download {spec.key}: {e}[/red]")
            continue

        # Check if the expected database exists in the downloaded snapshot
        source_path = Path(downloaded_path) / spec.db_filename
        if not source_path.exists():
            console.print(
                f"[red]Database {spec.key} not found in HuggingFace repo.[/red]"
            )
            console.print(
                f"[yellow]The database may not have been uploaded yet. "
                f"Try running 'evaluations build {spec.key}' to create it locally.[/yellow]"
            )
            continue

        # Remove existing database if force is set
        if db.exists():
            shutil.rmtree(db)

        # Copy from cache to target location
        db.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_path, db)

        console.print(f"[green]Downloaded {spec.key} to {db}[/green]")


@app.command()
def upload(
    dataset: str = typer.Argument(..., help="Dataset key or 'all' to upload all."),
) -> None:
    """Upload evaluation database to HuggingFace (maintainer only).

    Uses ``upload_large_folder`` for resumable, parallel transfer — important
    for the multi-GB ORB databases which would otherwise abort on any transient
    network failure under plain ``upload_folder``.

    ``upload_large_folder`` has no ``path_in_repo`` — it ships the contents of
    ``folder_path`` to the repo root. Stage the db under a temp parent with
    hardlinks so the basename becomes the remote path, leaving everything
    else at the root undisturbed.
    """
    import os
    import tempfile

    specs = _resolve_datasets(dataset)

    api = HfApi()

    for spec in specs:
        db = spec.db_path()
        if not db.exists():
            console.print(f"[red]Database not found at {db}[/red]")
            continue

        # Wipe the existing remote path so we don't accumulate orphaned files
        # from prior uploads. upload_large_folder doesn't accept delete_patterns,
        # so we do this as a separate commit. Safe to run if the path is missing.
        try:
            api.delete_folder(
                path_in_repo=spec.db_filename,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
            )
        except Exception:
            pass

        with tempfile.TemporaryDirectory() as staging:
            target = Path(staging) / spec.db_filename
            target.mkdir()
            for src in db.rglob("*"):
                if not src.is_file():
                    continue
                rel = src.relative_to(db)
                dest = target / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.link(src, dest)

            console.print(f"[blue]Uploading {spec.key} ({db})...[/blue]")
            api.upload_large_folder(
                folder_path=staging,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
            )

        console.print(f"[green]Uploaded {spec.key} to {HF_REPO_ID}[/green]")


if __name__ == "__main__":
    app()
