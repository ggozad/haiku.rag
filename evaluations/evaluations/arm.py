"""An evaluation arm declared in a file: what runs, from which checkout, against what."""

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ArmSpec(BaseModel):
    """One evaluation arm.

    Paths are resolved by `load_arm` against the file that declares them.
    `flags` pass through to `evaluations run` unchanged. `comparator` names the
    arm file this arm is paired against, and `differences` names every way the
    two arms differ; the preflight stops the launch on a difference it does not
    find in that list.
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
    comparator: Path | None = None
    differences: list[str] = []
    decision_rule: str | None = None
    operator: str | None = None
    deadline_hours: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _pairing_fields_agree(self) -> Self:
        if self.comparator is not None and self.decision_rule is None:
            raise ValueError("an arm with a comparator needs a decision_rule")
        if self.differences and self.comparator is None:
            raise ValueError("differences need a comparator to differ from")
        return self

    def command(self) -> list[str]:
        """The `evaluations run` invocation this arm stands for."""
        argv = [
            "evaluations",
            "run",
            self.dataset,
            "--config",
            str(self.config),
            "--name",
            self.name,
        ]
        if self.db is not None:
            argv += ["--db", str(self.db)]
        if self.limit is not None:
            argv += ["--limit", str(self.limit)]
        if self.filter_ids is not None:
            argv += ["--filter-ids", str(self.filter_ids)]
        return argv + list(self.flags)


_PATH_FIELDS = ("worktree", "config", "db", "filter_ids", "smoke_ids", "comparator")


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
    return ArmSpec.model_validate(data)
