import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic_evals.lifecycle import CaseLifecycle
from pydantic_evals.reporting import EvaluationReport, ReportCase, ReportCaseFailure

from haiku.rag.utils import get_default_data_dir

RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass
class CaseOutcome:
    case_name: str
    key: str | None
    passed: bool | None
    cited: bool
    cited_map: float | None
    aborted: bool


def check_run_name(name: str) -> str:
    """The name, when it is one path component: letters, digits, dot, dash and
    underscore, opening with a letter or digit."""
    if not RUN_NAME.match(name):
        raise ValueError(f"not a run name: {name!r}")
    return name


def new_run_id() -> str:
    """Distinguishes concurrent runs that share a name."""
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"


def default_results_path() -> Path:
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


def partial_path(directory: Path, name: str, run_id: str) -> Path:
    """Where a run appends its cases while it is still running."""
    return directory / f"{name}.{run_id}.partial.jsonl"


def case_writer(
    directory: Path, *, name: str, pair_key: str, run_id: str
) -> type[CaseLifecycle]:
    """A pydantic-evals lifecycle appending one row per case as it finishes."""
    check_run_name(name)
    path = partial_path(directory, name, run_id)
    directory.mkdir(parents=True, exist_ok=True)

    class Writer(CaseLifecycle):
        async def teardown(self, result: ReportCase | ReportCaseFailure | None) -> None:
            if result is None:
                return
            row = (
                _failure_row(result, pair_key, None)
                if isinstance(result, ReportCaseFailure)
                else _case_row(result, pair_key, None)
            )
            with path.open("a") as out:
                out.write(json.dumps(row) + "\n")

    return Writer


def write_results(
    report: EvaluationReport,
    *,
    name: str,
    pair_key: str,
    directory: Path,
    run_id: str,
) -> Path:
    """One JSON line per case, failures included as aborted, sorted by case
    name, to `<name>.<trace id>.jsonl` (`notrace-<run id>` without a trace). Raises rather than overwrite a file, and removes the run's partial
    file."""
    check_run_name(name)
    rows = [_case_row(case, pair_key, report.trace_id) for case in report.cases]
    rows += [
        _failure_row(failure, pair_key, report.trace_id) for failure in report.failures
    ]
    rows.sort(key=lambda row: row["case_name"])
    tag = report.trace_id or f"notrace-{run_id}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.{tag}.jsonl"
    with path.open("x") as out:
        out.write("".join(json.dumps(row) + "\n" for row in rows))
    partial_path(directory, name, run_id).unlink(missing_ok=True)
    return path


def read_results(path: Path) -> list[CaseOutcome]:
    return [
        CaseOutcome(
            case_name=row["case_name"],
            key=row["key"],
            passed=row["passed"],
            cited=bool(row["cited"]),
            cited_map=row["cited_map"],
            aborted=bool(row["aborted"]),
        )
        for row in (
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        )
    ]
