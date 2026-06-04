from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from evaluations.numbers import extract_numbers, numbers_close


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
        target = gold[0]
        candidates = extract_numbers(str(ctx.output))
        return (
            1.0 if any(numbers_close(c, target, self.eps) for c in candidates) else 0.0
        )
