"""Paired comparison of two arms on the same cases."""

import json
from dataclasses import dataclass
from math import comb

from evaluations.arm import same_commit
from evaluations.registry import ArmRecord
from evaluations.traces import CaseOutcome


def _two_sided_binomial(k: int, n: int) -> float:
    """P(X <= k) doubled, X ~ Binomial(n, 1/2), capped at 1."""
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    return _two_sided_binomial(min(b, c), b + c)


def sign_test(up: int, down: int) -> float:
    """Two-sided exact sign test p-value, ties excluded."""
    return _two_sided_binomial(min(up, down), up + down)


def resolution(discordant: int, alpha: float = 0.05) -> int | None:
    """The smallest |b - c| the exact test rejects at `alpha` with this many
    discordant pairs, or None when no split reaches it."""
    for delta in range(discordant % 2, discordant + 1, 2):
        if _two_sided_binomial((discordant - delta) // 2, discordant) < alpha:
            return delta
    return None


@dataclass
class ArmSummary:
    name: str
    cases: int
    judged: int
    passed: int
    accuracy: float | None
    floor: float | None
    cite_rate: float | None
    cited_map: float | None
    aborts: int
    unjudged: int


def summarize(name: str, outcomes: list[CaseOutcome]) -> ArmSummary:
    """Accuracy over judged cases, floor over all cases, cite rate and mean
    cited_map over the cases that carry one."""
    cases = len(outcomes)
    judged = [outcome for outcome in outcomes if outcome.passed is not None]
    passed = sum(1 for outcome in judged if outcome.passed)
    maps = [outcome.cited_map for outcome in outcomes if outcome.cited_map is not None]
    return ArmSummary(
        name=name,
        cases=cases,
        judged=len(judged),
        passed=passed,
        accuracy=passed / len(judged) if judged else None,
        floor=passed / cases if cases else None,
        cite_rate=sum(1 for outcome in outcomes if outcome.cited) / cases
        if cases
        else None,
        cited_map=sum(maps) / len(maps) if maps else None,
        aborts=sum(1 for outcome in outcomes if outcome.aborted),
        unjudged=cases - len(judged),
    )


@dataclass
class PairResult:
    key: str
    treated: ArmSummary
    baseline: ArmSummary
    paired: int
    only_treated: int
    only_baseline: int
    judged_pairs: int
    b: int
    c: int
    mcnemar_p: float
    up: int
    down: int
    ties: int
    sign_p: float


def _by_key(name: str, key: str, outcomes: list[CaseOutcome]) -> dict[str, CaseOutcome]:
    missing = sum(1 for outcome in outcomes if outcome.key is None)
    if missing:
        raise ValueError(f"{name}: {missing} case(s) carry no {key}; cannot pair on it")
    by_key: dict[str, CaseOutcome] = {}
    duplicates = 0
    for outcome in outcomes:
        assert outcome.key is not None
        if outcome.key in by_key:
            duplicates += 1
        by_key[outcome.key] = outcome
    if duplicates:
        raise ValueError(f"{name}: {duplicates} duplicate value(s) of {key}")
    return by_key


def pair_outcomes(
    key: str,
    treated_name: str,
    treated: list[CaseOutcome],
    baseline_name: str,
    baseline: list[CaseOutcome],
) -> PairResult:
    """Join two arms on `key`. Discordance counts pairs judged on both sides;
    the sign test counts pairs with a cited_map on both sides."""
    ours = _by_key(treated_name, key, treated)
    theirs = _by_key(baseline_name, key, baseline)
    shared = sorted(set(ours) & set(theirs))
    b = c = up = down = ties = judged_pairs = 0
    for case_key in shared:
        x, y = ours[case_key], theirs[case_key]
        if x.passed is not None and y.passed is not None:
            judged_pairs += 1
            if x.passed and not y.passed:
                b += 1
            elif y.passed and not x.passed:
                c += 1
        if x.cited_map is not None and y.cited_map is not None:
            if x.cited_map > y.cited_map:
                up += 1
            elif x.cited_map < y.cited_map:
                down += 1
            else:
                ties += 1
    return PairResult(
        key=key,
        treated=summarize(treated_name, treated),
        baseline=summarize(baseline_name, baseline),
        paired=len(shared),
        only_treated=len(set(ours) - set(theirs)),
        only_baseline=len(set(theirs) - set(ours)),
        judged_pairs=judged_pairs,
        b=b,
        c=c,
        mcnemar_p=mcnemar_exact(b, c),
        up=up,
        down=down,
        ties=ties,
        sign_p=sign_test(up, down),
    )


def _rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _only(count: int, name: str) -> str:
    return f", {count} case{'s' if count != 1 else ''} only in {name}"


def render(result: PairResult, decision_rule: str | None = None) -> str:
    """The standard paired table."""
    head = f"paired on {result.key}: {result.paired} cases on both sides"
    if result.only_treated:
        head += _only(result.only_treated, result.treated.name)
    if result.only_baseline:
        head += _only(result.only_baseline, result.baseline.name)
    width = max(len(result.treated.name), len(result.baseline.name), 3)
    lines = [
        head,
        f"{'arm':<{width}}  cases  accuracy    floor  cite rate  cited_map  aborts  unjudged",
    ]
    for arm in (result.treated, result.baseline):
        lines.append(
            f"{arm.name:<{width}}  {arm.cases:>5}  {_rate(arm.accuracy):>8}  "
            f"{_rate(arm.floor):>7}  {_rate(arm.cite_rate):>9}  {_rate(arm.cited_map):>9}  "
            f"{arm.aborts:>6}  {arm.unjudged:>8}"
        )
    treated, baseline = result.treated.name, result.baseline.name
    lines.append(
        f"discordant: {treated} pass / {baseline} fail {result.b}, "
        f"{treated} fail / {baseline} pass {result.c}; "
        f"McNemar exact p = {result.mcnemar_p:.4f} over {result.judged_pairs} judged pairs"
    )
    lines.append(
        f"cited_map sign test: {result.up} up, {result.down} down, {result.ties} ties; "
        f"p = {result.sign_p:.4f}"
    )
    discordant = result.b + result.c
    delta = resolution(discordant)
    if delta is None:
        lines.append(
            f"resolution: {discordant} discordant pairs; no split reaches p < 0.05"
        )
    else:
        share = (
            f" ({delta / result.paired:.1%} of {result.paired} paired cases)"
            if result.paired
            else ""
        )
        lines.append(
            f"resolution: {discordant} discordant pairs; the exact test rejects at "
            f"|b - c| >= {delta}{share}"
        )
    if decision_rule:
        lines.append(f"decision rule: {decision_rule}")
    return "\n".join(lines)


def orient(a: ArmRecord, b: ArmRecord) -> tuple[ArmRecord, ArmRecord]:
    """(treated, baseline) from the recorded comparator, never from argument order."""
    a_names_b = a.comparator == b.name
    b_names_a = b.comparator == a.name
    if a_names_b and not b_names_a:
        return a, b
    if b_names_a and not a_names_b:
        return b, a
    if a_names_b and b_names_a:
        raise ValueError(f"{a.name} and {b.name} each name the other as comparator")
    raise ValueError(f"neither {a.name} nor {b.name} names the other as comparator")


_ARM_FIELDS = {"sha", "limit", "db", "dataset", "filter_ids"}


def check_pair_rows(treated: ArmRecord, baseline: ArmRecord) -> list[str]:
    """Everything that stops a pair before any case is fetched: a void arm, a
    missing trace, different datasets, and named differences that the two
    rows do not show or unnamed ones they do."""
    problems: list[str] = []
    for record in (treated, baseline):
        if record.status == "void":
            problems.append(f"{record.name} is void: {record.void_reason}")
        if not record.trace_id:
            problems.append(f"{record.name} has no trace id")
    if treated.dataset != baseline.dataset:
        problems.append(f"datasets differ: {treated.dataset} vs {baseline.dataset}")

    named = set(json.loads(treated.differences)) if treated.differences else set()
    commits_differ = not same_commit(treated.git_sha or "", baseline.git_sha or "")
    if "sha" in named and not commits_differ:
        problems.append("sha is named as a difference but the commits agree")
    if commits_differ and "sha" not in named:
        problems.append(
            f"commits differ ({(treated.git_sha or '')[:12]} vs "
            f"{(baseline.git_sha or '')[:12]}) and sha is not named"
        )
    config_keys = sorted(
        name for name in named if name not in _ARM_FIELDS and not name.startswith("-")
    )
    hashes_differ = treated.config_hash != baseline.config_hash
    if config_keys and not hashes_differ:
        problems.append(
            f"config keys named ({', '.join(config_keys)}) but the config hashes agree"
        )
    if hashes_differ and not config_keys:
        problems.append("config hashes differ and no config key is named")
    if ("limit" in named) != (treated.limit_cases != baseline.limit_cases):
        problems.append("limit named and equal, or different and not named")
    if ("db" in named) != (treated.db_path != baseline.db_path):
        problems.append("db named and equal, or different and not named")
    return problems
