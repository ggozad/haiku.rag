"""Checks an arm must pass before it may start."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values
from pydantic import ValidationError

from evaluations.arm import ArmSpec, load_arm
from evaluations.datasets import DATASETS
from evaluations.experiment import code_revision, config_hash, corpus_fingerprint
from haiku.rag.config import load_yaml_config
from haiku.rag.config.models import AppConfig


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


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


def _check_worktree(arm: ArmSpec) -> Check:
    if not arm.worktree.is_dir():
        return Check("worktree", False, f"{arm.worktree} does not exist")
    revision = code_revision(arm.worktree)
    sha = revision["git_sha"]
    if sha is None:
        return Check("worktree", False, f"{arm.worktree} is not a git checkout")
    if not sha.startswith(arm.sha):
        return Check(
            "worktree", False, f"{arm.worktree} is at {sha}, the arm pins {arm.sha}"
        )
    if revision["git_dirty"]:
        return Check(
            "worktree", False, f"{sha} at {arm.worktree} has uncommitted changes"
        )
    return Check("worktree", True, f"{sha} at {arm.worktree}")


def _check_telemetry(arm: ArmSpec) -> Check:
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


async def _check_database(arm: ArmSpec, config: AppConfig | None) -> Check:
    if config is None:
        return Check("database", False, "config did not validate")
    if arm.db is None and config.lancedb.databases:
        names = ", ".join(sorted(config.lancedb.databases))
        return Check("database", True, f"configured set: {names}")
    spec = DATASETS.get(arm.dataset)
    if arm.db is not None:
        db = arm.db
    elif spec is not None:
        db = spec.db_path(None)
    else:
        return Check("database", False, "no db and no dataset default to place one")
    if not db.exists():
        return Check("database", False, f"{db} does not exist")
    fingerprint = await corpus_fingerprint(db, config)
    if fingerprint["db_documents"] is None:
        return Check("database", False, f"{db} holds no haiku.rag tables")
    model = config.embeddings.model
    stored = (
        fingerprint["db_embedder_provider"],
        fingerprint["db_embedder_model"],
        fingerprint["db_embedder_dim"],
    )
    wanted = (model.provider, model.name, model.vector_dim)
    if stored != wanted:
        return Check(
            "database",
            False,
            f"{db} stores embedder {stored[0]}/{stored[1]} dim {stored[2]}, "
            f"the config names {wanted[0]}/{wanted[1]} dim {wanted[2]}",
        )
    return Check(
        "database",
        True,
        f"{db}: {fingerprint['db_documents']} documents, "
        f"{fingerprint['db_chunks']} chunks, embedder {model.provider}/{model.name} "
        f"dim {model.vector_dim}, schema {fingerprint['db_version']}",
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


def _options(flags: list[str]) -> dict[str, tuple[str, ...]]:
    """Group flag tokens by the option they belong to, so a difference is
    named by the option and not by a bare value."""
    grouped: dict[str, list[str]] = {}
    current = ""
    for token in flags:
        if token.startswith("-"):
            current = token
            grouped.setdefault(current, [])
        else:
            grouped.setdefault(current, []).append(token)
    return {option: tuple(values) for option, values in grouped.items()}


def _same_commit(a: str, b: str) -> bool:
    return a.startswith(b) or b.startswith(a)


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
    if not _same_commit(arm.sha, other.sha):
        names.append("sha")
    if arm.limit != other.limit:
        names.append("limit")
    for field in ("db", "filter_ids"):
        if str(getattr(arm, field)) != str(getattr(other, field)):
            names.append(field)
    ours, theirs = _options(arm.flags), _options(other.flags)
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


def _check_comparator(arm: ArmSpec, config: AppConfig | None) -> Check:
    assert arm.comparator is not None
    if not arm.comparator.is_file():
        return Check("comparator", False, f"{arm.comparator} does not exist")
    try:
        other = load_arm(arm.comparator)
    except (ValidationError, ValueError, yaml.YAMLError) as error:
        return Check("comparator", False, f"{arm.comparator}: {error}")
    other_config_check, other_config = _check_config(other)

    actual = set(arm_differences(arm, other, config, other_config))
    named = set(arm.differences)
    problems: list[str] = []
    if unnamed := sorted(actual - named):
        problems.append("unnamed differences: " + ", ".join(unnamed))
    if stale := sorted(named - actual):
        problems.append("named but not different: " + ", ".join(stale))
    if config is None or other_config is None:
        problems.append(
            "configs not compared: "
            + (
                other_config_check.detail
                if other_config is None
                else "this arm's config did not validate"
            )
        )
    if problems:
        return Check("comparator", False, "; ".join(problems))
    listed = ", ".join(sorted(actual)) or "nothing (a null pair)"
    return Check("comparator", True, f"differs from {other.name} in: {listed}")


async def run_preflight(arm_path: Path) -> list[Check]:
    """Every check an arm must pass before it starts, in the order an operator
    reads them. An arm file that does not validate is the only check."""
    try:
        arm = load_arm(arm_path)
    except (ValidationError, ValueError, OSError, yaml.YAMLError) as error:
        return [Check("arm file", False, f"{arm_path}: {error}")]
    checks = [Check("arm file", True, f"{arm.name}: {' '.join(arm.command())}")]
    checks.append(_check_dataset(arm))
    checks.append(_check_worktree(arm))
    checks.append(_check_telemetry(arm))
    config_check, config = _check_config(arm)
    checks.append(config_check)
    checks.append(await _check_database(arm, config))
    checks.append(_check_filter_files(arm))
    if arm.comparator is not None:
        checks.append(_check_comparator(arm, config))
    return checks
