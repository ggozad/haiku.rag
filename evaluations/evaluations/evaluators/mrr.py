from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class MRREvaluator(Evaluator):
    """
    Mean Reciprocal Rank evaluator for single-document retrieval.

    MRR = 1/rank where rank is the position of the first relevant document.
    Returns 0 if no relevant document is found.

    Appropriate for retrieval tasks where each query has exactly one relevant document.
    """

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """
        Calculate reciprocal rank for a single query.

        Expected context:
        - ctx.metadata['relevant_uris']: set/list of relevant document URIs
        - ctx.output: list of retrieved document URIs (ordered by rank)

        Returns:
            float: 1/rank of first relevant doc, or 0.0 if not found
        """
        if ctx.metadata is None:
            return 0.0
        relevant_uris = set(ctx.metadata.get("relevant_uris", []))
        retrieved_uris = ctx.output

        for rank, uri in enumerate(retrieved_uris, start=1):
            if uri in relevant_uris:
                return 1.0 / rank

        return 0.0
