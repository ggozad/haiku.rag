import re
from pathlib import Path

import pytest

SKILLS = sorted(
    (Path(__file__).parents[2] / ".claude" / "skills").glob("eval-*/SKILL.md")
)
PATTERNS = {
    "commit sha": r"\b[0-9a-f]{12}\b",
    "trace id": r"\b[0-9a-f]{32}\b",
    "date": r"\b20[0-9]{2}-[01][0-9]-[0-3][0-9]\b",
}


def test_the_operator_skills_are_present() -> None:
    assert [path.parent.name for path in SKILLS] == ["eval-analysis", "eval-launch"]


@pytest.mark.parametrize("path", SKILLS, ids=lambda path: path.parent.name)
def test_a_skill_holds_procedure_only(path: Path) -> None:
    """Facts about one run belong in the registry or the reference document,
    so a skill carries no commit sha, trace id or date above its change log."""
    body = path.read_text().split("## Change log")[0]
    for name, pattern in PATTERNS.items():
        assert re.search(pattern, body) is None, f"{path.parent.name} carries a {name}"
