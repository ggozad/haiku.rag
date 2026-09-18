"""Checks an arm must pass before it may start."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values
from pydantic import ValidationError

from evaluations.arm import (
    _ARM_FILE_SUFFIXES,
    ArmSpec,
    flag_options,
    load_arm,
    same_commit,
)
from evaluations.datasets import DATASETS
from evaluations.experiment import code_revision, config_hash, corpus_fingerprint
from evaluations.registry import ArmRecord, Registry
from haiku.rag.config import load_yaml_config
from haiku.rag.config.models import AppConfig
from haiku.rag.utils import locate_database


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class Preflight:
    """The checks an arm was put through, and what they resolved on the way:
    the arm, its config, the database it reads and the commit it runs from."""

    checks: list[Check] = field(default_factory=list)
    arm: ArmSpec | None = None
    config: AppConfig | None = None
    db: Path | None = None
    fingerprint: dict[str, Any] | None = None
    git_sha: str | None = None

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)


def _load_config(path: Path) -> AppConfig:
    return AppConfig.model_validate(load_yaml_config(path))


def _check_dataset(arm: ArmSpec) -> Check:
    if arm.dataset in DATASETS:
        return Check("dataset", True, arm.dataset)
    return Check(
        "dataset",
        False,
        f"unknown dataset {arm.dataset}; choose from {', '.join(sorted(DATASETS))}",
    )


def _check_worktree(arm: ArmSpec) -> tuple[Check, str | None]:
    if not arm.worktree.is_dir():
        return Check("worktree", False, f"{arm.worktree} does not exist"), None
    revision = code_revision(arm.worktree)
    sha = revision["git_sha"]
    if sha is None:
        return Check("worktree", False, f"{arm.worktree} is not a git checkout"), None
    if not sha.startswith(arm.sha):
        return Check(
            "worktree", False, f"{arm.worktree} is at {sha}, the arm pins {arm.sha}"
        ), None
    if revision["git_dirty"]:
        return Check(
            "worktree", False, f"{sha} at {arm.worktree} has uncommitted changes"
        ), sha
    return Check("worktree", True, f"{sha} at {arm.worktree}"), sha


def _check_telemetry(arm: ArmSpec) -> Check:
    if "--no-telemetry" in arm.flags:
        spec = DATASETS.get(arm.dataset)
        if "--skip-qa" in arm.flags:
            return Check(
                "telemetry",
                False,
                "--no-telemetry with --skip-qa records nothing: the QA run is "
                "what writes a result file",
            )
        if spec is not None and spec.live:
            return Check(
                "telemetry",
                False,
                f"--no-telemetry on {arm.dataset}: a live conversation run "
                "writes no result file",
            )
        return Check("telemetry", True, "--no-telemetry: the result file is the record")
    env_file = arm.worktree / ".env"
    token = dotenv_values(env_file).get("LOGFIRE_TOKEN") if env_file.is_file() else None
    if not token:
        return Check(
            "telemetry",
            False,
            f"{env_file} has no LOGFIRE_TOKEN; the run would record nothing",
        )
    return Check("telemetry", True, f"LOGFIRE_TOKEN present in {env_file}")


def _check_config(arm: ArmSpec) -> tuple[Check, AppConfig | None]:
    if not arm.config.is_file():
        return Check("config", False, f"{arm.config} does not exist"), None
    # A preflight reports a loader failure, whatever raised it.
    try:
        config = _load_config(arm.config)
    except Exception as error:
        return Check("config", False, f"{arm.config}: {error}"), None
    return Check("config", True, f"{arm.config} hash {config_hash(config)}"), config


async def _database_problem(
    db: Path, config: AppConfig
) -> tuple[str | None, dict[str, Any] | None]:
    """Why `db` cannot back this config, and what it holds."""
    if not db.exists():
        return f"{db} does not exist", None
    fingerprint = await corpus_fingerprint(db, config)
    if fingerprint["db_documents"] is None:
        return f"{db} holds no haiku.rag tables", fingerprint
    model = config.embeddings.model
    stored = (
        fingerprint["db_embedder_provider"],
        fingerprint["db_embedder_model"],
        fingerprint["db_embedder_dim"],
    )
    wanted = (model.provider, model.name, model.vector_dim)
    if stored != wanted:
        return (
            f"{db} stores embedder {stored[0]}/{stored[1]} dim {stored[2]}, "
            f"the config names {wanted[0]}/{wanted[1]} dim {wanted[2]}",
            fingerprint,
        )
    return None, fingerprint


async def _check_database(
    arm: ArmSpec, config: AppConfig | None
) -> tuple[Check, Path | None, dict[str, Any] | None]:
    if config is None:
        return Check("database", False, "config did not validate"), None, None
    if config.lancedb.databases:
        names = ", ".join(sorted(config.lancedb.databases))
        if arm.db is not None:
            return (
                Check(
                    "database",
                    False,
                    f"lancedb.databases names {names} and the arm names {arm.db}; "
                    "a run refuses the two together",
                ),
                None,
                None,
            )
        if "--skip-db" not in arm.flags:
            return (
                Check(
                    "database",
                    False,
                    f"lancedb.databases names {names}, which a run reads but "
                    "population does not write; the arm needs --skip-db",
                ),
                None,
                None,
            )
        problems = []
        remote = []
        for name, location in sorted(config.lancedb.databases.items()):
            placed = locate_database(location)
            if not isinstance(placed, Path):
                remote.append(name)
                continue
            # A relative location resolves where the run reads it, its worktree.
            problem, _ = await _database_problem(
                arm.worktree / placed.expanduser(), config
            )
            if problem is not None:
                problems.append(f"{name}: {problem}")
        if problems:
            return Check("database", False, "; ".join(problems)), None, None
        detail = f"configured set: {names}"
        if remote:
            detail += f"; {', '.join(remote)} not opened, a URI carries no path"
        return Check("database", True, detail), None, None
    spec = DATASETS.get(arm.dataset)
    if arm.db is not None:
        db = arm.db
    elif spec is not None:
        db = spec.db_path(None)
    else:
        return (
            Check("database", False, "no db and no dataset default to place one"),
            None,
            None,
        )
    problem, fingerprint = await _database_problem(db, config)
    if problem is not None:
        return Check("database", False, problem), db, fingerprint
    assert fingerprint is not None
    model = config.embeddings.model
    return (
        Check(
            "database",
            True,
            f"{db}: {fingerprint['db_documents']} documents, "
            f"{fingerprint['db_chunks']} chunks, embedder {model.provider}/{model.name} "
            f"dim {model.vector_dim}, schema {fingerprint['db_version']}",
        ),
        db,
        fingerprint,
    )


def _check_filter_files(arm: ArmSpec) -> Check:
    files = [path for path in (arm.filter_ids, arm.smoke_ids) if path is not None]
    if not files:
        return Check("filter ids", True, "none")
    ok = True
    details: list[str] = []
    for path in files:
        if not path.is_file():
            ok = False
            details.append(f"{path} does not exist")
            continue
        count = sum(1 for line in path.read_text().splitlines() if line.strip())
        if count == 0:
            ok = False
            details.append(f"{path} is empty")
        else:
            details.append(f"{path}: {count} ids")
    return Check("filter ids", ok, "; ".join(details))


def _flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(data, dict):
        flat: dict[str, Any] = {}
        for key, value in data.items():
            flat.update(_flatten(value, f"{prefix}{key}."))
        return flat
    return {prefix[:-1]: data}


def arm_differences(
    arm: ArmSpec,
    other: ArmSpec,
    config: AppConfig | None,
    other_config: AppConfig | None,
) -> list[str]:
    """Every way two arms differ, named as the operator must name them:
    arm fields by field, flags by option, config keys by dotted path."""
    names: list[str] = []
    if arm.dataset != other.dataset:
        names.append("dataset")
    if not same_commit(arm.sha, other.sha):
        names.append("sha")
    if arm.limit != other.limit:
        names.append("limit")
    for field_name in ("db", "filter_ids"):
        if str(getattr(arm, field_name)) != str(getattr(other, field_name)):
            names.append(field_name)
    ours, theirs = flag_options(arm.flags), flag_options(other.flags)
    names += sorted(
        option
        for option in set(ours) | set(theirs)
        if ours.get(option) != theirs.get(option)
    )
    if config is not None and other_config is not None:
        flat = _flatten(config.model_dump(mode="json"))
        other_flat = _flatten(other_config.model_dump(mode="json"))
        names += sorted(
            key
            for key in set(flat) | set(other_flat)
            if flat.get(key) != other_flat.get(key)
        )
    return names


def record_differences(
    arm: ArmSpec,
    config: AppConfig | None,
    record: ArmRecord,
    other_config: AppConfig | None,
) -> tuple[list[str], list[str]]:
    """Differences from a registered arm, and what its row cannot show.

    A row records no flags, so a flag difference is invisible to it. Config
    keys are visible only while the config the row names is still on disk.
    """
    names: list[str] = []
    blind: list[str] = ["flags, which a row does not record"]
    if arm.dataset != record.dataset:
        names.append("dataset")
    if not record.git_sha:
        blind.append("sha, which the row leaves unrecorded")
    elif not same_commit(arm.sha, record.git_sha):
        names.append("sha")
    if arm.limit != record.limit_cases:
        names.append("limit")
    if str(arm.db) != str(record.db_path):
        names.append("db")
    if str(arm.filter_ids) != str(record.filter_ids):
        names.append("filter_ids")
    if config is not None and other_config is not None:
        flat = _flatten(config.model_dump(mode="json"))
        other_flat = _flatten(other_config.model_dump(mode="json"))
        names += sorted(
            key
            for key in set(flat) | set(other_flat)
            if flat.get(key) != other_flat.get(key)
        )
    return names, blind


def _unverifiable(name: str, blind: list[str]) -> bool:
    """Whether a named difference is one the comparison could not check."""
    if not blind:
        return False
    kind = "flags" if name.startswith("-") else "config keys" if "." in name else name
    return any(reason.startswith(kind) for reason in blind)


def _check_comparator(
    arm: ArmSpec, config: AppConfig | None, registry: "Registry | None"
) -> Check:
    assert arm.comparator is not None
    blind: list[str] = []
    problems: list[str] = []
    if arm.comparator.endswith(_ARM_FILE_SUFFIXES):
        path = Path(arm.comparator)
        if not path.is_file():
            return Check("comparator", False, f"{path} does not exist")
        try:
            other = load_arm(path)
        except (ValidationError, ValueError, yaml.YAMLError) as error:
            return Check("comparator", False, f"{path}: {error}")
        other_config_check, other_config = _check_config(other)
        other_name = other.name
        actual = set(arm_differences(arm, other, config, other_config))
        if other_config is None:
            problems.append(f"configs not compared: {other_config_check.detail}")
    else:
        if registry is None:
            return Check(
                "comparator",
                False,
                f"{arm.comparator} names a registered arm and there is no registry "
                "to read it from",
            )
        record = registry.get(arm.comparator)
        if record is None:
            return Check(
                "comparator", False, f"no arm named {arm.comparator} in the registry"
            )
        other_name = record.name
        other_config: AppConfig | None = None
        if record.config_path and Path(record.config_path).is_file():
            _, other_config = _check_config(
                arm.model_copy(update={"config": Path(record.config_path)})
            )
        found, blind = record_differences(arm, config, record, other_config)
        actual = set(found)
        if other_config is None:
            if config is not None and config_hash(config) != (record.config_hash or ""):
                problems.append(
                    f"the configs differ and {record.config_path} is gone, so the "
                    "keys cannot be named"
                )
            else:
                blind.append(f"config keys, since {record.config_path} is gone")

    named = set(arm.differences)
    # A difference the row cannot show is taken on trust: the operator names it
    # and nothing here can contradict them.
    trusted = {name for name in named if _unverifiable(name, blind)}
    if unnamed := sorted(actual - named):
        problems.append("unnamed differences: " + ", ".join(unnamed))
    if stale := sorted(named - actual - trusted):
        problems.append("named but not different: " + ", ".join(stale))
    if config is None:
        problems.append("configs not compared: this arm's config did not validate")
    if problems:
        return Check("comparator", False, "; ".join(problems))
    listed = ", ".join(sorted(actual)) or "nothing (a null pair)"
    detail = f"differs from {other_name} in: {listed}"
    if blind:
        detail += "; not compared: " + ", ".join(blind)
    return Check("comparator", True, detail)


async def run_preflight(arm_path: Path, registry: Registry | None = None) -> Preflight:
    """Every check an arm must pass before it starts, in the order an operator
    reads them. An arm file that does not validate is the only check."""
    try:
        arm = load_arm(arm_path)
    except (ValidationError, ValueError, OSError, yaml.YAMLError) as error:
        return Preflight(checks=[Check("arm file", False, f"{arm_path}: {error}")])
    result = Preflight(arm=arm)
    result.checks.append(
        Check("arm file", True, f"{arm.name}: {' '.join(arm.command())}")
    )
    result.checks.append(_check_dataset(arm))
    worktree_check, result.git_sha = _check_worktree(arm)
    result.checks.append(worktree_check)
    result.checks.append(_check_telemetry(arm))
    config_check, result.config = _check_config(arm)
    result.checks.append(config_check)
    database_check, result.db, result.fingerprint = await _check_database(
        arm, result.config
    )
    result.checks.append(database_check)
    result.checks.append(_check_filter_files(arm))
    if arm.comparator is not None:
        result.checks.append(_check_comparator(arm, result.config, registry))
    return result
