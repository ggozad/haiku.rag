from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.evaluators.evaluator import EvaluatorOutput


def _cited_uris(ctx: EvaluatorContext) -> list[str]:
    return list(ctx.attributes.get("cited_uris") or [])


def _relevant_uris(ctx: EvaluatorContext) -> set[str]:
    if ctx.metadata is None:
        return set()
    return set(ctx.metadata.get("relevant_uris", []))


def average_precision(cited: list[str], relevant: set[str]) -> float:
    """AP of the cited URIs against the relevant set (0.0 when nothing hits)."""
    precisions: list[float] = []
    found = 0
    for rank, uri in enumerate(cited, start=1):
        if uri in relevant:
            found += 1
            precisions.append(found / rank)
    if not precisions:
        return 0.0
    return sum(precisions) / len(relevant)


@dataclass
class CitationMAPEvaluator(Evaluator):
    """Average precision over the URIs the capability cited via the `cite` tool.

    Reads ``cited_uris`` from ``ctx.attributes`` (recorded during the task run
    via :func:`pydantic_evals.set_eval_attribute`) and ``relevant_uris`` from
    ``ctx.metadata``. Cases without relevant URIs (e.g. unanswerable turns)
    are ineligible and produce no score.
    """

    def get_default_evaluation_name(self) -> str:
        return "cited_map"

    def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
        relevant = _relevant_uris(ctx)
        if not relevant:
            return {}
        return average_precision(_cited_uris(ctx), relevant)
