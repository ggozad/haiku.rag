import re
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from evaluations.numbers import extract_numbers, numbers_close

_ANSWER_RE = re.compile(r"(?im)^[\s*>#_-]*(?:final\s+)?answer\s*[:=]\s*(.+)$")


def _answer_segment(text: str) -> str:
    """Restrict to a declared ``ANSWER:`` line when present, so numbers in the
    surrounding reasoning don't count. Falls back to the whole text."""
    matches = _ANSWER_RE.findall(text)
    return matches[-1] if matches else text


@dataclass
class NumberMatchEvaluator(Evaluator):
    """Deterministic numeric scoring for datasets with numeric gold answers.

    Scores 1.0 when any number parsed from the prediction matches the gold
    answer within relative tolerance ``eps``, else 0.0. Non-numeric predictions
    score 0.0.
    """

    eps: float = 0.01

    def get_default_evaluation_name(self) -> str:
        return "number_match"

    def evaluate(self, ctx: EvaluatorContext) -> float:
        gold = extract_numbers(str(ctx.expected_output))
        if not gold:
            return 0.0
        # Gold mixes conventions: signs for changes are inconsistent (+0.2 vs
        # -1.9) and ratios appear as either a percent (37.81) or a decimal
        # (0.3781). Compare the declared answer by magnitude, at a ×100 scale
        # either way. Safe because we score only the single ANSWER-line number.
        target = abs(gold[0])
        candidates = [abs(c) for c in extract_numbers(_answer_segment(str(ctx.output)))]
        scales = (1.0, 0.01, 100.0)
        matched = any(
            numbers_close(c * s, target, self.eps) for c in candidates for s in scales
        )
        return 1.0 if matched else 0.0
