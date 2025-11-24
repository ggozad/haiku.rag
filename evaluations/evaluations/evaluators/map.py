from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


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
        retrieved_uris = ctx.output

        if not relevant_uris:
            return 0.0

        num_relevant = len(relevant_uris)
        precisions = []
        num_relevant_found = 0

        for rank, uri in enumerate(retrieved_uris, start=1):
            if uri in relevant_uris:
                num_relevant_found += 1
                precision_at_k = num_relevant_found / rank
                precisions.append(precision_at_k)

        if not precisions:
            return 0.0

        return sum(precisions) / num_relevant
