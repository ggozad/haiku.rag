from collections import Counter
from dataclasses import dataclass
from typing import Any

# A sandbox surface is counted as touched when the model's own Python names it.
# Coverage is measured, not required: a surface nothing reaches is a finding about
# whether it earns its place in the VFS, not a failure of the run.
SURFACE_MARKERS: dict[str, tuple[str, ...]] = {
    "list_documents()": ("list_documents(",),
    "in-code search()": ("search(",),
    "content.txt": ("content.txt",),
    "items.jsonl": ("items.jsonl",),
    "toc.json": ("toc.json",),
    "metadata.json": ("metadata.json",),
}

# Families where exactly one database is correct, so a citation elsewhere is an
# attribution error rather than a difference of opinion.
SINGLE_SOURCE_FAMILIES = ("B1", "B3", "B4")


def _flag(case: Any, name: str) -> bool | None:
    """Read a boolean assertion, or a score recorded as a number."""
    assertion = (getattr(case, "assertions", None) or {}).get(name)
    if assertion is not None:
        return bool(assertion.value)
    score = (getattr(case, "scores", None) or {}).get(name)
    if score is not None:
        return bool(score.value >= 1.0)
    return None


def _meta(case: Any, key: str, default: Any = None) -> Any:
    return (getattr(case, "metadata", None) or {}).get(key, default)


@dataclass
class Gate:
    name: str
    checked: int
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures

    def line(self) -> str:
        state = "PASS" if self.passed else f"FAIL ({len(self.failures)})"
        detail = "" if self.passed else "  " + ", ".join(self.failures[:6])
        return f"{state:<12} {self.name} over {self.checked} case(s){detail}"


def scope_gate(cases: list[Any]) -> Gate:
    """No scoped case cites a database outside its scope. Includes the empty
    scope, which may cite nothing at all."""
    checked, failures = 0, []
    for case in cases:
        if _meta(case, "scope") is None:
            continue
        checked += 1
        if _flag(case, "scope_honoured") is False:
            failures.append(case.name)
    return Gate("scope leakage", checked, failures)


def attribution_gate(cases: list[Any]) -> Gate:
    """Every single-source case cites exactly the database that holds the answer.
    `cited_map` scores URIs and would pass a wrongly attributed answer."""
    checked, failures = 0, []
    for case in cases:
        if _meta(case, "family") not in SINGLE_SOURCE_FAMILIES:
            continue
        if _meta(case, "expected_sources") is None:
            continue
        checked += 1
        if _flag(case, "attribution_correct") is False:
            failures.append(case.name)
    return Gate("attribution errors", checked, failures)


def family_rates(cases: list[Any]) -> dict[str, tuple[int, int]]:
    """Answered-correctly over scored, per family. Families with no numeric or
    ordered-text expectation (the refusal families) are absent rather than 0/0."""
    passed: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for case in cases:
        family = _meta(case, "family")
        flag = _flag(case, "answer_correct")
        if family is None or flag is None:
            continue
        total[family] += 1
        passed[family] += int(flag)
    return {family: (passed[family], total[family]) for family in sorted(total)}


def surface_coverage(cases: list[Any]) -> dict[str, int]:
    """How many cases reached each sandbox surface, from the recorded Python."""
    counts = dict.fromkeys(SURFACE_MARKERS, 0)
    for case in cases:
        code = " ".join(
            (getattr(case, "attributes", None) or {}).get("executed_code") or []
        )
        if not code:
            continue
        for surface, markers in SURFACE_MARKERS.items():
            if any(marker in code for marker in markers):
                counts[surface] += 1
    return counts


def render(cases: list[Any]) -> tuple[str, bool]:
    """The gate report. Returns the text and whether both hard gates passed."""
    gates = [scope_gate(cases), attribution_gate(cases)]
    lines = ["", "=== Multi-database gates ===", ""]
    lines += [gate.line() for gate in gates]

    lines += ["", "=== Answer rates by family ===", ""]
    for family, (passed, total) in family_rates(cases).items():
        share = f"{passed / total:.0%}" if total else "-"
        lines.append(f"{family:<4} {passed:>3}/{total:<3} {share}")

    coverage = surface_coverage(cases)
    if any(coverage.values()):
        lines += ["", "=== Sandbox surfaces reached ===", ""]
        for surface, count in coverage.items():
            note = "" if count else "   (never reached)"
            lines.append(f"{surface:<18} {count:>3} case(s){note}")

    all_passed = all(gate.passed for gate in gates)
    lines += ["", f"GATES: {'PASSED' if all_passed else 'FAILED'}", ""]
    return "\n".join(lines), all_passed
