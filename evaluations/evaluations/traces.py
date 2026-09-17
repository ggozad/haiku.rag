"""Per-case outcomes read from Logfire."""

import json
import os
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The query API returns at most this many rows per query and does not say so.
PAGE = 100
_TRACE_ID = re.compile(r"^[0-9a-f]{32}$")
_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

QueryFn = Callable[..., list[dict[str, Any]]]


def read_key() -> str:
    key = os.environ.get("LOGFIRE_READ_KEY", "")
    path = Path.home() / ".logfire-read-key"
    if not key and path.is_file():
        key = path.read_text().strip()
    if not key:
        raise RuntimeError(
            "no Logfire read key: set LOGFIRE_READ_KEY or write ~/.logfire-read-key"
        )
    return key


def query_logfire(
    sql: str, *, min_timestamp: str, key: str | None = None
) -> list[dict[str, Any]]:
    """Run SQL against the Logfire query API. The key's prefix names the region."""
    key = key or read_key()
    region = key.split("_")[2] if key.startswith("pylf_") else "eu"
    request = urllib.request.Request(
        f"https://logfire-{region}.pydantic.dev/v2/query",
        data=json.dumps({"sql": sql, "min_timestamp": min_timestamp}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)["data"]


@dataclass
class CaseOutcome:
    case_name: str
    key: str | None
    passed: bool | None
    cited: bool
    cited_map: float | None
    aborted: bool


def _true(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def _passed(row: dict[str, Any]) -> bool | None:
    verdict = row.get("answer_equivalent")
    if verdict is not None:
        return _true(verdict)
    score = row.get("number_match")
    if score is not None:
        return float(score) >= 1.0
    return None


def _outcome(row: dict[str, Any]) -> CaseOutcome:
    key = row.get("pair_key")
    cited_map = row.get("cited_map")
    return CaseOutcome(
        case_name=str(row["case_name"]),
        key=None if key is None else str(key),
        passed=_passed(row),
        cited=int(row.get("n_cited") or 0) > 0,
        cited_map=None if cited_map is None else float(cited_map),
        aborted=_true(row.get("is_exception")),
    )


def case_outcomes(
    trace_id: str, key: str, *, query: QueryFn, min_timestamp: str
) -> list[CaseOutcome]:
    """Every case span of a trace, read page by page past the row cap and
    checked against the span count. `key` is the case-metadata field the
    dataset pairs on, read as text."""
    if not _TRACE_ID.match(trace_id):
        raise ValueError(f"not a trace id: {trace_id!r}")
    if not _KEY.match(key):
        raise ValueError(f"not a metadata key: {key!r}")
    scope = f"trace_id = '{trace_id}' AND span_name = 'case: {{case_name}}'"
    counted = query(
        f"SELECT count(*) AS n FROM records WHERE {scope}", min_timestamp=min_timestamp
    )
    expected = int(counted[0]["n"])
    outcomes: list[CaseOutcome] = []
    offset = 0
    while True:
        rows = query(
            f"SELECT attributes->>'case_name' AS case_name, "
            f"attributes->'metadata'->>'{key}' AS pair_key, "
            "attributes->'assertions'->'answer_equivalent'->>'value' AS answer_equivalent, "
            "attributes->'scores'->'number_match'->>'value' AS number_match, "
            "attributes->'scores'->'cited_map'->>'value' AS cited_map, "
            "json_length(attributes, 'attributes', 'cited_uris') AS n_cited, "
            "is_exception "
            f"FROM records WHERE {scope} ORDER BY case_name LIMIT {PAGE} OFFSET {offset}",
            min_timestamp=min_timestamp,
        )
        outcomes.extend(_outcome(row) for row in rows)
        if len(rows) < PAGE:
            break
        offset += PAGE
    if len(outcomes) != expected:
        raise ValueError(
            f"trace {trace_id} has {expected} case spans but {len(outcomes)} were read"
        )
    return outcomes
