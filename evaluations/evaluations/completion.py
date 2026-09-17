"""Complete a registry row from the trace an arm produced."""

import re
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from evaluations.datasets import DATASETS
from evaluations.pairing import ArmSummary, summarize
from evaluations.registry import Registry
from evaluations.traces import QueryFn, case_outcomes

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_RUN = re.compile(r"\bevaluations run\b")


def window_start(started_at: str, hours: int = 1) -> str:
    """A Logfire `min_timestamp` this many hours before a launch, in UTC."""
    started = datetime.fromisoformat(started_at)
    return (
        (started - timedelta(hours=hours))
        .astimezone(UTC)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def find_trace(run_name: str, since: str, *, query: QueryFn) -> str | None:
    """The trace of the experiment span named `run_name` started at or after
    `since`. None when there is none; more than one is an error."""
    if not _NAME.match(run_name):
        raise ValueError(f"not a run name: {run_name!r}")
    rows = query(
        "SELECT trace_id FROM records WHERE service_name = 'evals' "
        "AND span_name = 'evaluate {name}' "
        f"AND attributes->>'name' = '{run_name}' AND start_timestamp >= '{since}'",
        min_timestamp=since,
    )
    if not rows:
        return None
    if len(rows) > 1:
        found = ", ".join(str(row["trace_id"]) for row in rows)
        raise ValueError(
            f"{len(rows)} traces named {run_name!r} since {since}: {found}"
        )
    return str(rows[0]["trace_id"])


def _experiment_span(
    trace_id: str, since: str, *, query: QueryFn
) -> dict[str, Any] | None:
    rows = query(
        "SELECT trace_id, start_timestamp, end_timestamp FROM records "
        f"WHERE trace_id = '{trace_id}' AND span_name = 'evaluate {{name}}'",
        min_timestamp=since,
    )
    return rows[0] if rows else None


def complete_arm(
    registry: Registry,
    name: str,
    *,
    query: QueryFn,
    trace_id: str | None = None,
    void_reason: str | None = None,
) -> ArmSummary | None:
    """Fill an arm's result fields from its trace. Without a trace the arm is
    void with reason "no telemetry" and None is returned. `void_reason` voids
    the arm after recording its metrics, for a run that did not finish."""
    record = registry.get(name)
    if record is None:
        raise ValueError(f"no arm named {name!r}")
    since = window_start(record.started_at)
    if trace_id is None:
        trace_id = find_trace(name, since, query=query)
    if trace_id is None:
        registry.mark_void(name, "no telemetry")
        return None
    spec = DATASETS.get(record.dataset)
    key = spec.pair_key if spec is not None else "question_id"
    summary = summarize(
        name, case_outcomes(trace_id, key, query=query, min_timestamp=since)
    )
    wall_seconds = ended_at = None
    span = _experiment_span(trace_id, since, query=query)
    if span is not None:
        start = datetime.fromisoformat(str(span["start_timestamp"]))
        end = datetime.fromisoformat(str(span["end_timestamp"]))
        wall_seconds = (end - start).total_seconds()
        ended_at = str(span["end_timestamp"])
    registry.complete(
        name,
        trace_id=trace_id,
        cases=summary.cases,
        accuracy=summary.accuracy,
        cite_rate=summary.cite_rate,
        cited_map=summary.cited_map,
        aborts=summary.aborts,
        wall_seconds=wall_seconds,
        ended_at=ended_at,
    )
    if void_reason:
        registry.mark_void(name, void_reason)
    return summary


def count_evaluation_runs(command_lines: Iterable[str]) -> int:
    """How many `evaluations run` processes the given command lines show."""
    return sum(
        1
        for line in command_lines
        if _RUN.search(line) and not line.lstrip().startswith("grep")
    )


def running_evaluations() -> int:
    """`evaluations run` processes on this machine right now."""
    listing = subprocess.run(
        ["ps", "-eo", "args"], capture_output=True, text=True, check=True
    )
    return count_evaluation_runs(listing.stdout.splitlines())
