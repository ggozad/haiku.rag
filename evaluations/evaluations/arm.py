"""An evaluation arm declared in a file: what runs, from which checkout, against what."""

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# The `evaluations run` options `ArmSpec.command` fills from the arm's own
# fields. A flag repeating one of these wins over the pinned value.
PINNED_OPTIONS = ("--config", "--name", "--db", "--limit", "--filter-ids")


class ArmSpec(BaseModel):
    """One evaluation arm.

    Paths are resolved by `load_arm` against the file that declares them.
    `flags` pass through to `evaluations run` unchanged and may not repeat an
    option the arm's own fields fill (`PINNED_OPTIONS`). `comparator` names the
    arm this arm is paired against, as a file when it ends in `.yaml` or `.yml`
    and as a registered arm name otherwise, `hypothesis` the claim the pair tests,
    and `differences` names every way the two arms differ; the preflight stops
    the launch on a difference it does not find in that list.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    dataset: str
    worktree: Path
    sha: str = Field(pattern=r"^[0-9a-f]{12,40}$")
    config: Path
    db: Path | None = None
    limit: int | None = Field(default=None, gt=0)
    filter_ids: Path | None = None
    flags: list[str] = []
    smoke_ids: Path | None = None
    comparator: str | None = None
    differences: list[str] = []
    decision_rule: str | None = None
    hypothesis: str | None = None
    operator: str | None = None

    @model_validator(mode="after")
    def _flags_leave_the_pinned_options_alone(self) -> Self:
        clash = sorted(set(flag_options(self.flags)) & set(PINNED_OPTIONS))
        if clash:
            raise ValueError(
                f"flags may not set {', '.join(clash)}: the arm file pins it, "
                "and a flag would run something else than the registry records"
            )
        return self

    @model_validator(mode="after")
    def _pairing_fields_agree(self) -> Self:
        if self.comparator is not None and self.decision_rule is None:
            raise ValueError("an arm with a comparator needs a decision_rule")
        if self.comparator is not None and not self.hypothesis:
            raise ValueError("an arm with a comparator needs a hypothesis")
        if self.differences and self.comparator is None:
            raise ValueError("differences need a comparator to differ from")
        return self

    def command(self, smoke: bool = False) -> list[str]:
        """The `evaluations run` invocation this arm stands for. The smoke
        variant runs `smoke_ids` under the name `<name>-smoke`."""
        argv = [
            "evaluations",
            "run",
            self.dataset,
            "--config",
            str(self.config),
            "--name",
            f"{self.name}-smoke" if smoke else self.name,
        ]
        if self.db is not None:
            argv += ["--db", str(self.db)]
        if smoke:
            if self.smoke_ids is None:
                raise ValueError(f"{self.name} has no smoke_ids")
            argv += ["--filter-ids", str(self.smoke_ids)]
        else:
            if self.limit is not None:
                argv += ["--limit", str(self.limit)]
            if self.filter_ids is not None:
                argv += ["--filter-ids", str(self.filter_ids)]
        return argv + list(self.flags)


def flag_options(flags: list[str]) -> dict[str, tuple[str, ...]]:
    """Flag tokens grouped by the option they belong to, `--option value` and
    `--option=value` alike."""
    grouped: dict[str, list[str]] = {}
    current = ""
    for token in flags:
        if token.startswith("-"):
            current, _, attached = token.partition("=")
            grouped.setdefault(current, [])
            if attached:
                grouped[current].append(attached)
        else:
            grouped.setdefault(current, []).append(token)
    return {option: tuple(values) for option, values in grouped.items()}


def same_commit(a: str, b: str) -> bool:
    """Whether two sha prefixes name one commit. An unknown sha names none."""
    return bool(a) and bool(b) and (a.startswith(b) or b.startswith(a))


_PATH_FIELDS = ("worktree", "config", "db", "filter_ids", "smoke_ids")
# A comparator written as a file resolves like one; anything else is the name
# of an arm the registry already holds.
_ARM_FILE_SUFFIXES = (".yaml", ".yml")


def load_arm(path: Path) -> ArmSpec:
    """Read an arm file. Relative paths resolve against the file's directory
    and `~` expands."""
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a mapping")
    base = path.absolute().parent
    for field in _PATH_FIELDS:
        value = data.get(field)
        if value is not None:
            data[field] = str(base / Path(str(value)).expanduser())
    comparator = data.get("comparator")
    if comparator is not None and str(comparator).endswith(_ARM_FILE_SUFFIXES):
        data["comparator"] = str(base / Path(str(comparator)).expanduser())
    return ArmSpec.model_validate(data)
