"""Per-case results of a run, written under the evaluations data directory so
a pairing or a completion needs no telemetry service."""

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic_evals.lifecycle import CaseLifecycle
from pydantic_evals.reporting import EvaluationReport, ReportCase, ReportCaseFailure

from evaluations.traces import CaseOutcome
from haiku.rag.utils import get_default_data_dir

RESULTS_ENV = "HAIKU_RAG_EVAL_RESULTS"

# A run name is one path component, and quotes into SQL as it stands.
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def check_run_name(name: str) -> str:
    """The name, when it is a run name: letters, digits, dot, dash and
    underscore, opening with a letter or digit."""
    if not RUN_NAME.match(name):
        raise ValueError(f"not a run name: {name!r}")
    return name


def default_results_path() -> Path:
    """`HAIKU_RAG_EVAL_RESULTS` when set, else results/ in the data directory.
    The queue sets the variable for the runs it starts, so a checkout that
    predates the `--results` option still writes where the queue reads."""
    configured = os.environ.get(RESULTS_ENV)
    if configured:
        return Path(configured)
    return get_default_data_dir() / "evaluations" / "results"


def _key(metadata: Any, pair_key: str) -> str | None:
    value = metadata.get(pair_key) if isinstance(metadata, dict) else None
    return None if value is None else str(value)


def _passed(case: ReportCase) -> bool | None:
    verdict = case.assertions.get("answer_equivalent")
    if verdict is not None:
        return bool(verdict.value)
    score = case.scores.get("number_match")
    if score is not None:
        return float(score.value) >= 1.0
    return None


def _case_row(case: ReportCase, pair_key: str, trace_id: str | None) -> dict[str, Any]:
    cited_map = case.scores.get("cited_map")
    verdict = case.assertions.get("answer_equivalent")
    return {
        "case_name": case.name,
        "key": _key(case.metadata, pair_key),
        "passed": _passed(case),
        "cited": bool(case.attributes.get("cited_uris")),
        "cited_map": None if cited_map is None else float(cited_map.value),
        "aborted": False,
        "trace_id": case.trace_id or trace_id,
        "answer": None if case.output is None else str(case.output),
        "reason": None if verdict is None else verdict.reason,
        "attributes": dict(case.attributes),
        "task_duration": case.task_duration,
    }


def _failure_row(
    failure: ReportCaseFailure, pair_key: str, trace_id: str | None
) -> dict[str, Any]:
    return {
        "case_name": failure.name,
        "key": _key(failure.metadata, pair_key),
        "passed": None,
        "cited": False,
        "cited_map": None,
        "aborted": True,
        "trace_id": failure.trace_id or trace_id,
        "answer": None,
        "reason": failure.error_message,
        "attributes": {},
        "task_duration": None,
    }


def partial_path(directory: Path, name: str) -> Path:
    """Where a run appends its cases while it is still running."""
    return directory / f"{name}.partial.jsonl"


def case_writer(directory: Path, *, name: str, pair_key: str) -> type[CaseLifecycle]:
    """A pydantic-evals lifecycle that appends one row per case as it finishes.

    A run that is killed or crashes keeps the cases it completed, and a watcher
    has something that grows without reading telemetry.
    """
    check_run_name(name)
    path = partial_path(directory, name)

    class Writer(CaseLifecycle):
        async def teardown(self, result: ReportCase | ReportCaseFailure | None) -> None:
            if result is None:
                return
            row = (
                _failure_row(result, pair_key, None)
                if isinstance(result, ReportCaseFailure)
                else _case_row(result, pair_key, None)
            )
            directory.mkdir(parents=True, exist_ok=True)
            with path.open("a") as out:
                out.write(json.dumps(row) + "\n")

    return Writer


def write_results(
    report: EvaluationReport, *, name: str, pair_key: str, directory: Path
) -> Path:
    """One JSON line per case, failures included as aborted, sorted by case
    name: the `CaseOutcome` fields, the trace id, the answer, the judge's
    reason and the run's per-case attributes. The file is
    `<name>.<trace id prefix>.jsonl`, or `notrace-<time>` without a trace."""
    check_run_name(name)
    rows = [_case_row(case, pair_key, report.trace_id) for case in report.cases]
    rows += [
        _failure_row(failure, pair_key, report.trace_id) for failure in report.failures
    ]
    rows.sort(key=lambda row: row["case_name"])
    tag = (
        report.trace_id[:12]
        if report.trace_id
        else "notrace-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.{tag}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    partial_path(directory, name).unlink(missing_ok=True)
    return path


def find_results(directory: Path, name: str) -> Path | None:
    """The result file of the run named `name`, the partial one when the run
    wrote no other. More than one finished file is an error."""
    check_run_name(name)
    matches = sorted(directory.glob(f"{name}.*.jsonl")) if directory.is_dir() else []
    partial = partial_path(directory, name)
    finished = [path for path in matches if path != partial]
    if len(finished) > 1:
        listed = ", ".join(path.name for path in finished)
        raise ValueError(
            f"{len(finished)} result files for {name!r} in {directory}: {listed}"
        )
    if finished:
        return finished[0]
    return partial if partial in matches else None


def read_results(path: Path) -> tuple[str | None, list[CaseOutcome]]:
    """The trace id the rows carry, and the outcomes."""
    trace_id: str | None = None
    outcomes: list[CaseOutcome] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        trace_id = row.get("trace_id") or trace_id
        outcomes.append(
            CaseOutcome(
                case_name=row["case_name"],
                key=row["key"],
                passed=row["passed"],
                cited=bool(row["cited"]),
                cited_map=row["cited_map"],
                aborted=bool(row["aborted"]),
            )
        )
    return trace_id, outcomes
