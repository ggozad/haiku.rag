"""QA benchmarks: single-question runs and live multi-turn conversations."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

from pydantic_evals import Case, Dataset as EvalDataset, set_eval_attribute
from pydantic_evals.evaluators import Evaluator
from pydantic_evals.reporting import ReportCaseFailure
from rich.console import Console

from evaluations.capability_runner import (
    CapabilityFactory,
    prefix_to_messages,
    run_capability_conversation,
    run_capability_question,
)
from evaluations.config import ConversationInput, DatasetSpec, ScopedQuestion
from evaluations.evaluators import (
    ANSWER_EQUIVALENCE_RUBRIC,
    REFUSAL_ELIGIBLE_LABELS,
    REFUSAL_RUBRIC,
    ConversationEvaluator,
    RefusalJudge,
    TranscriptLLMJudge,
)
from evaluations.experiment import DEFAULT_JUDGE_MODEL, build_experiment_metadata
from haiku.rag.config import AppConfig
from haiku.rag.config.models import ModelConfig
from haiku.rag.utils import get_model

console = Console()

Target = Literal["rag-capability", "analysis-capability"]
TARGETS: tuple[Target, ...] = ("rag-capability", "analysis-capability")


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
    if not any(isinstance(case.inputs, str) for case in cases):
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


def _live_summary(report_cases, report_failures) -> dict[str, float | int] | None:
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
    turns_judged = sum(_score(case, "turns_judged") for case in scored)
    turns_passed = sum(_score(case, "turns_passed") for case in scored)
    # A conversation with zero judged turns (its judge calls all failed)
    # reports turn_pass_rate 0.0; averaging that in would count a judge
    # outage as a failed conversation, against the exclusion policy.
    judged = [case for case in scored if _score(case, "turns_judged")]
    summary: dict[str, float | int] = {
        "conversations": len(scored),
        "conversations_attempted": len(report_cases) + len(report_failures),
        "turns_total": turns_total,
        "turns_judged": turns_judged,
        "turns_attempted": turns_total + failed_turns,
        "micro_pass_rate": turns_passed / turns_judged if turns_judged else 0.0,
        "macro_pass_rate": sum(_score(case, "turn_pass_rate") for case in judged)
        / len(judged)
        if judged
        else 0.0,
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

    true_refusals = sum(_score(case, "true_refusals") for case in scored)
    false_refusals = sum(_score(case, "false_refusals") for case in scored)
    unanswerable = sum(_score(case, "unanswerable_turns") for case in scored)
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
        if refused is None or label not in REFUSAL_ELIGIBLE_LABELS:
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

    Returns the corpus unchanged when ``case_ids`` is None. Matching nothing
    raises: a dataset keying its rows by another name leaves every case filtered
    out, and a run of no cases otherwise reports 0.0000 as though it were a score.
    """
    if case_ids is None:
        return corpus
    filtered = corpus.filter(lambda row: row.get("id") in case_ids)
    if len(filtered) == 0:
        raise ValueError(
            f"--filter-ids matched none of the {len(corpus)} cases. "
            "Check that the ids belong to this dataset and that its rows are "
            "keyed by `id`."
        )
    return filtered


class _QARun(NamedTuple):
    cases: list[Case[Any, Any, dict[str, Any]]]
    db: Path | None
    judge_config: ModelConfig
    eval_name: str
    experiment_metadata: dict[str, Any]
    capability_factory: CapabilityFactory
    capability_model: Any


def _prepare_qa_run(
    spec: DatasetSpec,
    config: AppConfig,
    limit: int | None,
    name: str | None,
    db_path: Path | None,
    judge_model: ModelConfig | None,
    target: Target,
    capability_model: ModelConfig | None,
    case_ids: set[str] | None,
    document_filter: str | None,
) -> _QARun:
    """Shared setup for the QA runners: cases, models, name and metadata."""
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

    eval_name = name if name is not None else f"{spec.key}_qa_evaluation"
    experiment_metadata = build_experiment_metadata(
        dataset_key=spec.key,
        test_cases=len(cases),
        config=config,
        judge_config=judge_config,
        target=target,
        capability_config=capability_config,
        document_filter=document_filter,
    )
    experiment_metadata.update(spec.experiment_metadata or {})

    return _QARun(
        cases=cases,
        db=None
        if spec.uses_configured_databases(config, db_path)
        else spec.db_path(db_path),
        judge_config=judge_config,
        eval_name=eval_name,
        experiment_metadata=experiment_metadata,
        capability_factory=_capability_factory_for_target(target),
        capability_model=get_model(capability_config, config),
    )


def _print_mean_task_time(report_cases, unit: str = "case") -> None:
    if not report_cases:
        return
    mean = sum(case.task_duration for case in report_cases) / len(report_cases)
    console.print(f"Avg task time per {unit}: {mean:.2f}s")


def _print_failures(failures, show_question: bool = False) -> None:
    if not failures:
        return
    console.print("[red]\nSummary of failures:[/red]")
    for failure in failures:
        console.print(f"Case: {failure.name}")
        if show_question:
            console.print(f"Question: {failure.inputs}")
        console.print(f"Error: {failure.error_message}")
        console.print("")


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
    document_filter: str | None = None,
) -> ReportCaseFailure[str, str, dict[str, str]] | None:
    run = _prepare_qa_run(
        spec,
        config,
        limit,
        name,
        db_path,
        judge_model,
        target,
        capability_model,
        case_ids,
        document_filter,
    )
    cases, judge_config = run.cases, run.judge_config

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
    # RefusalJudge scores only cases whose metadata carries an answerability
    # label; on unlabeled datasets it returns no score without a judge call.
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

    async def answer_question(inputs: str | ConversationInput | ScopedQuestion) -> str:
        sources: list[str] | None = None
        if isinstance(inputs, ConversationInput):
            question = inputs.question
            message_history = prefix_to_messages(inputs.prefix)
        elif isinstance(inputs, ScopedQuestion):
            question = inputs.question
            sources = inputs.sources
            message_history = None
        else:
            question = inputs
            message_history = None
        result = await run_capability_question(
            capability_factory=run.capability_factory,
            db_path=run.db,
            config=config,
            question=question,
            capability_model=run.capability_model,
            document_filter=document_filter,
            message_history=message_history,
            sources=sources,
        )
        set_eval_attribute("cited_uris", result.cited_uris)
        set_eval_attribute("cited_chunk_ids", result.cited_chunk_ids)
        set_eval_attribute("cited_sources", result.cited_sources)
        set_eval_attribute("searched_uris", result.searched_uris)
        set_eval_attribute("n_searches", result.n_searches)
        set_eval_attribute("n_search_calls", result.n_search_calls)
        set_eval_attribute("n_rejected_searches", result.n_rejected_searches)
        set_eval_attribute("n_failed_tools", result.n_failed_tools)
        set_eval_attribute("n_executions", result.n_executions)
        set_eval_attribute("n_requests", result.n_requests)
        set_eval_attribute("citation_status", result.citation_status)
        set_eval_attribute("executed_code", result.executed_code)
        return result.answer

    report = await evaluation_dataset.evaluate(
        answer_question,
        name=run.eval_name,
        max_concurrency=1,
        progress=True,
        metadata=run.experiment_metadata,
    )

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
    _print_mean_task_time(report.cases)

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

    if (metrics := _refusal_metrics(report.cases)) is not None:
        precision, recall, unanswerable, refusals = metrics
        console.print("\n=== Refusal vs answerability labels ===", style="bold cyan")
        console.print(f"Refusal precision: {precision:.2%} | recall: {recall:.2%}")
        console.print(
            f"UNANSWERABLE turns: {unanswerable} | refusals: {refusals} "
            "(PARTIAL excluded)"
        )

    if spec.report_hook is not None:
        spec.report_hook(list(report.cases))

    _print_failures(failures, show_question=True)

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
    document_filter: str | None = None,
) -> None:
    """Replay conversations turn by turn through one capability session.

    One case per conversation; ``limit`` counts conversations. Answers carry
    forward as real message history, so prior-turn compaction is exercised.
    """
    run = _prepare_qa_run(
        spec,
        config,
        limit,
        name,
        db_path,
        judge_model,
        target,
        capability_model,
        case_ids,
        document_filter,
    )

    evaluation_dataset = EvalDataset[Any, Any, dict[str, Any]](
        name=spec.key,
        cases=run.cases,
        evaluators=[
            ConversationEvaluator(
                rubric=ANSWER_EQUIVALENCE_RUBRIC,
                model=get_model(run.judge_config, config),
            )
        ],
    )

    async def answer_conversation(questions: list[str]) -> list[str]:
        results = await run_capability_conversation(
            capability_factory=run.capability_factory,
            db_path=run.db,
            config=config,
            questions=list(questions),
            capability_model=run.capability_model,
            document_filter=document_filter,
            compaction=spec.compaction,
        )
        set_eval_attribute("turn_cited_uris", [r.cited_uris for r in results])
        set_eval_attribute("turn_n_search_calls", [r.n_search_calls for r in results])
        set_eval_attribute(
            "turn_n_rejected_searches", [r.n_rejected_searches for r in results]
        )
        set_eval_attribute("turn_n_failed_tools", [r.n_failed_tools for r in results])
        set_eval_attribute("turn_n_requests", [r.n_requests for r in results])
        set_eval_attribute("turn_citation_status", [r.citation_status for r in results])
        return [r.answer for r in results]

    report = await evaluation_dataset.evaluate(
        answer_conversation,
        name=run.eval_name,
        max_concurrency=1,
        progress=True,
        metadata=run.experiment_metadata,
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

    _print_failures(report.failures)
