from dataclasses import dataclass
from math import comb

from evaluations.results import CaseOutcome


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
    cite_rate_all_cases: float | None
    cited_map: float | None
    aborts: int
    unjudged: int


def summarize(name: str, outcomes: list[CaseOutcome]) -> ArmSummary:
    """Accuracy over judged cases, floor and cite rate over all cases, mean
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
        cite_rate_all_cases=sum(1 for outcome in outcomes if outcome.cited) / cases
        if cases
        else None,
        cited_map=sum(maps) / len(maps) if maps else None,
        aborts=sum(1 for outcome in outcomes if outcome.aborted),
        unjudged=cases - len(judged),
    )


@dataclass
class PairResult:
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


def _by_key(name: str, outcomes: list[CaseOutcome]) -> dict[str, CaseOutcome]:
    missing = sum(1 for outcome in outcomes if outcome.key is None)
    if missing:
        raise ValueError(f"{name}: {missing} case(s) carry no pairing key")
    by_key: dict[str, CaseOutcome] = {}
    duplicates = 0
    for outcome in outcomes:
        assert outcome.key is not None
        if outcome.key in by_key:
            duplicates += 1
        by_key[outcome.key] = outcome
    if duplicates:
        raise ValueError(f"{name}: {duplicates} duplicate pairing key(s)")
    return by_key


def pair_outcomes(
    treated_name: str,
    treated: list[CaseOutcome],
    baseline_name: str,
    baseline: list[CaseOutcome],
) -> PairResult:
    """Join two arms on their pairing keys. Discordance counts pairs judged on
    both sides; the sign test counts pairs with a cited_map on both sides."""
    ours = _by_key(treated_name, treated)
    theirs = _by_key(baseline_name, baseline)
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


def render(result: PairResult) -> str:
    """The standard paired table."""
    head = f"{result.paired} cases on both sides"
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
            f"{_rate(arm.floor):>7}  {_rate(arm.cite_rate_all_cases):>9}  {_rate(arm.cited_map):>9}  "
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
        share = f" ({delta / result.paired:.1%} of {result.paired} paired cases)"
        lines.append(
            f"resolution: {discordant} discordant pairs; the exact test rejects at "
            f"|b - c| >= {delta}{share}"
        )
    lines.append(
        "denominators: accuracy over judged cases; floor and cite rate over all "
        "cases; cited_map over the cases that carry one"
    )
    return "\n".join(lines)
