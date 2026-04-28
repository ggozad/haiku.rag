from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


def _cited_uris(ctx: EvaluatorContext) -> list[str]:
    return list(ctx.attributes.get("cited_uris") or [])


def _relevant_uris(ctx: EvaluatorContext) -> set[str]:
    if ctx.metadata is None:
        return set()
    return set(ctx.metadata.get("relevant_uris", []))


@dataclass
class CitationMRREvaluator(Evaluator):
    """Reciprocal rank over the URIs the skill cited via the `cite` tool.

    Reads ``cited_uris`` from ``ctx.attributes`` (recorded during the task run
    via :func:`pydantic_evals.set_eval_attribute`) and ``relevant_uris`` from
    ``ctx.metadata``. Returns ``1.0/rank`` of the first cited URI that is in
    the relevant set, or ``0.0`` if none match.

    Use for single-document datasets, mirroring :class:`MRREvaluator`.
    """

    evaluation_name: str = "cited_mrr"

    def evaluate(self, ctx: EvaluatorContext) -> float:
        relevant = _relevant_uris(ctx)
        for rank, uri in enumerate(_cited_uris(ctx), start=1):
            if uri in relevant:
                return 1.0 / rank
        return 0.0


@dataclass
class CitationMAPEvaluator(Evaluator):
    """Average precision over the URIs the skill cited via the `cite` tool.

    Same input shape as :class:`CitationMRREvaluator`; use for multi-document
    datasets, mirroring :class:`MAPEvaluator`.
    """

    evaluation_name: str = "cited_map"

    def evaluate(self, ctx: EvaluatorContext) -> float:
        relevant = _relevant_uris(ctx)
        if not relevant:
            return 0.0
        precisions: list[float] = []
        found = 0
        for rank, uri in enumerate(_cited_uris(ctx), start=1):
            if uri in relevant:
                found += 1
                precisions.append(found / rank)
        if not precisions:
            return 0.0
        return sum(precisions) / len(relevant)
