import math
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


def _relevant_and_retrieved(ctx: EvaluatorContext) -> tuple[set[str], list[str]]:
    if ctx.metadata is None:
        return set(), []
    return set(ctx.metadata.get("relevant_uris", [])), list(ctx.output)


@dataclass
class RecallEvaluator(Evaluator):
    """Recall@k: fraction of relevant documents retrieved in the top k."""

    k: int

    def get_default_evaluation_name(self) -> str:
        return f"recall_{self.k}"

    def evaluate(self, ctx: EvaluatorContext) -> float:
        relevant, retrieved = _relevant_and_retrieved(ctx)
        if not relevant:
            return 0.0
        found = sum(1 for uri in retrieved[: self.k] if uri in relevant)
        return found / len(relevant)


@dataclass
class NDCGEvaluator(Evaluator):
    """Binary nDCG@k: DCG of relevant documents in the top k over the ideal DCG.

    Gains are binary (relevant or not), matching qrels without graded scores.
    """

    k: int

    def get_default_evaluation_name(self) -> str:
        return f"ndcg_{self.k}"

    def evaluate(self, ctx: EvaluatorContext) -> float:
        relevant, retrieved = _relevant_and_retrieved(ctx)
        if not relevant:
            return 0.0
        dcg = sum(
            1 / math.log2(rank + 1)
            for rank, uri in enumerate(retrieved[: self.k], start=1)
            if uri in relevant
        )
        ideal = sum(
            1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), self.k) + 1)
        )
        return dcg / ideal
