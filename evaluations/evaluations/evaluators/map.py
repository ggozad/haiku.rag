from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from evaluations.evaluators.citation import average_precision


@dataclass
class MAPEvaluator(Evaluator):
    """
    Mean Average Precision evaluator for multi-document retrieval.

    AP = (sum of P@k for each relevant doc) / total relevant docs
    where P@k is precision at position k.

    Appropriate for retrieval tasks where queries have multiple relevant documents.
    """

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """
        Calculate average precision for a single query.

        Expected context:
        - ctx.metadata['relevant_uris']: set/list of relevant document URIs
        - ctx.output: list of retrieved document URIs (ordered by rank)

        Returns:
            float: Average precision (0.0-1.0)
        """
        if ctx.metadata is None:
            return 0.0
        relevant_uris = set(ctx.metadata.get("relevant_uris", []))
        if not relevant_uris:
            return 0.0
        return average_precision(list(ctx.output), relevant_uris)
