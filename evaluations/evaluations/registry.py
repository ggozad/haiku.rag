"""The registry of evaluation arms: one row per arm, written by the harness."""

import json
import sqlite3
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any

from evaluations.arm import _ARM_FILE_SUFFIXES, ArmSpec, load_arm
from evaluations.datasets import DATASETS
from evaluations.experiment import DEFAULT_JUDGE_MODEL, config_hash
from haiku.rag.config.models import AppConfig
from haiku.rag.utils import get_default_data_dir, model_base_url


@dataclass
class ArmRecord:
    """One arm. Launch fields are filled when the arm starts, result fields
    when it completes. `status` is launched, valid or void; `source` says who
    wrote the row (harness, logfire, note, benchmarks, log)."""

    name: str
    dataset: str
    kind: str
    status: str
    started_at: str
    source: str
    void_reason: str | None = None
    git_sha: str | None = None
    config_path: str | None = None
    config_hash: str | None = None
    db_path: str | None = None
    db_documents: int | None = None
    db_chunks: int | None = None
    db_embedder: str | None = None
    db_version: str | None = None
    db_written_at: str | None = None
    capability_model: str | None = None
    capability_endpoint: str | None = None
    judge_model: str | None = None
    reranker: str | None = None
    embedder: str | None = None
    limit_cases: int | None = None
    filter_ids: str | None = None
    comparator: str | None = None
    differences: str | None = None
    decision_rule: str | None = None
    hypothesis: str | None = None
    operator: str | None = None
    concurrency: int | None = None
    trace_id: str | None = None
    cases: int | None = None
    accuracy: float | None = None
    cite_rate_all_cases: float | None = None
    cited_map: float | None = None
    aborts: int | None = None
    wall_seconds: float | None = None
    ended_at: str | None = None
    notes: str | None = None


_COLUMNS = [field.name for field in fields(ArmRecord)]
_DDL = ", ".join(
    "name TEXT PRIMARY KEY"
    if field.name == "name"
    else f"{field.name}{'' if field.default is not MISSING else ' NOT NULL'}"
    for field in fields(ArmRecord)
)
_INSERT = f"({', '.join(_COLUMNS)}) VALUES ({', '.join('?' * len(_COLUMNS))})"
# Every field added after a registry file was written. All are nullable, so an
# older file takes them as columns without a migration of its rows.
_ADDABLE = [field.name for field in fields(ArmRecord) if field.default is not MISSING]


def default_registry_path() -> Path:
    return get_default_data_dir() / "evaluations" / "registry.sqlite"


def _values(record: ArmRecord) -> tuple[Any, ...]:
    return tuple(getattr(record, column) for column in _COLUMNS)


def _record(row: sqlite3.Row) -> ArmRecord:
    return ArmRecord(**dict(zip(row.keys(), row, strict=True)))


class Registry:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        with self._db:
            self._db.execute(f"CREATE TABLE IF NOT EXISTS arms ({_DDL})")
            present = {
                row["name"] for row in self._db.execute("PRAGMA table_info(arms)")
            }
            for column in _ADDABLE:
                if column not in present:
                    self._db.execute(f"ALTER TABLE arms ADD COLUMN {column}")

    def register_launch(self, record: ArmRecord) -> None:
        try:
            with self._db:
                self._db.execute(f"INSERT INTO arms {_INSERT}", _values(record))
        except sqlite3.IntegrityError:
            raise ValueError(f"arm {record.name!r} is already registered") from None

    def get(self, name: str) -> ArmRecord | None:
        row = self._db.execute("SELECT * FROM arms WHERE name = ?", (name,)).fetchone()
        return _record(row) if row is not None else None

    def list(
        self,
        *,
        dataset: str | None = None,
        db_path: str | None = None,
        status: str | None = None,
    ) -> list[ArmRecord]:
        clauses: list[str] = []
        params: list[str] = []
        for column, value in (
            ("dataset", dataset),
            ("db_path", db_path),
            ("status", status),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._db.execute(
            f"SELECT * FROM arms{where} ORDER BY started_at, name", params
        ).fetchall()
        return [_record(row) for row in rows]

    def complete(
        self,
        name: str,
        *,
        trace_id: str | None,
        cases: int | None,
        accuracy: float | None,
        cite_rate_all_cases: float | None = None,
        cited_map: float | None = None,
        aborts: int | None = None,
        wall_seconds: float | None = None,
        ended_at: str | None = None,
    ) -> None:
        self._update(
            name,
            status="valid",
            trace_id=trace_id,
            cases=cases,
            accuracy=accuracy,
            cite_rate_all_cases=cite_rate_all_cases,
            cited_map=cited_map,
            aborts=aborts,
            wall_seconds=wall_seconds,
            ended_at=ended_at,
        )

    def mark_void(self, name: str, reason: str) -> None:
        self._update(name, status="void", void_reason=reason)

    def _update(self, name: str, **values: Any) -> None:
        assignments = ", ".join(f"{column} = ?" for column in values)
        with self._db:
            cursor = self._db.execute(
                f"UPDATE arms SET {assignments} WHERE name = ?",
                (*values.values(), name),
            )
        if cursor.rowcount == 0:
            raise ValueError(f"no arm named {name!r}")

    def export_jsonl(self, path: Path) -> int:
        """One JSON object per arm with sorted keys, ordered by start, so two
        exports of the same registry are byte-identical and diffable."""
        records = self.list()
        path.write_text(
            "".join(
                json.dumps(asdict(record), sort_keys=True) + "\n" for record in records
            )
        )
        return len(records)

    def import_jsonl(self, path: Path) -> int:
        """Upsert every row of an export by name."""
        count = 0
        with self._db:
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                record = ArmRecord(**json.loads(line))
                self._db.execute(
                    f"INSERT OR REPLACE INTO arms {_INSERT}", _values(record)
                )
                count += 1
        return count


def _comparator_name(comparator: str | None) -> str | None:
    """The arm name a comparator points at: a file's `name`, or the name itself."""
    if comparator is None:
        return None
    if comparator.endswith(_ARM_FILE_SUFFIXES):
        return load_arm(Path(comparator)).name
    return comparator


def _capability(config: AppConfig, kind: str) -> tuple[str | None, str | None]:
    """The model a QA arm runs its capability on and the endpoint it opens.
    A retrieval or build arm runs no capability."""
    if kind != "qa":
        return None, None
    return config.qa.model.name, model_base_url(config.qa.model, config)


def _kind(flags: list[str]) -> str:
    skip_qa = "--skip-qa" in flags
    skip_retrieval = "--skip-retrieval" in flags
    if skip_qa and skip_retrieval:
        return "build"
    return "retrieval" if skip_qa else "qa"


def _judge(arm: ArmSpec, config: AppConfig, kind: str) -> str | None:
    if kind != "qa":
        return None
    spec = DATASETS.get(arm.dataset)
    if spec is not None and spec.qa_evaluator is not None:
        return None
    return (config.evaluations.judge or DEFAULT_JUDGE_MODEL).name


def launch_record(
    arm: ArmSpec,
    config: AppConfig,
    fingerprint: dict[str, Any] | None,
    *,
    started_at: str,
    git_sha: str | None = None,
) -> ArmRecord:
    """The row an arm gets when it launches, from its file, its resolved
    config and the fingerprint of the database it reads."""
    kind = _kind(arm.flags)
    capability_model, capability_endpoint = _capability(config, kind)
    embed = config.embeddings.model
    corpus = fingerprint or {}
    db_embedder = (
        None
        if corpus.get("db_embedder_model") is None
        else f"{corpus['db_embedder_provider']}/{corpus['db_embedder_model']} "
        f"dim {corpus['db_embedder_dim']}"
    )
    return ArmRecord(
        name=arm.name,
        dataset=arm.dataset,
        kind=kind,
        status="launched",
        started_at=started_at,
        source="harness",
        git_sha=git_sha or arm.sha,
        config_path=str(arm.config),
        config_hash=config_hash(config),
        db_path=corpus.get("db_path"),
        db_documents=corpus.get("db_documents"),
        db_chunks=corpus.get("db_chunks"),
        db_embedder=db_embedder,
        db_version=corpus.get("db_version"),
        db_written_at=corpus.get("db_written_at"),
        capability_model=capability_model,
        capability_endpoint=capability_endpoint,
        judge_model=_judge(arm, config, kind),
        reranker=config.reranking.model.name if config.reranking.model else None,
        embedder=f"{embed.provider}/{embed.name} dim {embed.vector_dim}",
        limit_cases=arm.limit,
        filter_ids=None if arm.filter_ids is None else str(arm.filter_ids),
        comparator=_comparator_name(arm.comparator),
        differences=None if arm.comparator is None else json.dumps(arm.differences),
        decision_rule=arm.decision_rule,
        hypothesis=arm.hypothesis,
        operator=arm.operator,
    )
