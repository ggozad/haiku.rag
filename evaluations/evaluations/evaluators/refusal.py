from dataclasses import dataclass

from pydantic_evals.evaluators import EvaluatorContext, LLMJudge
from pydantic_evals.evaluators.evaluator import EvaluatorOutput

REFUSAL_RUBRIC = (
    "The output declines to answer: it states that the information is not "
    "available, cannot be found, or is not known, rather than providing a "
    "substantive answer."
)

_ELIGIBLE_LABELS = ("ANSWERABLE", "UNANSWERABLE")


@dataclass
class RefusalJudge(LLMJudge):
    """Judges whether the output declines to answer.

    Runs only on turns whose answerability label makes refusal meaningful
    (ANSWERABLE/UNANSWERABLE); other turns produce no evaluation and cost no
    judge call.
    """

    async def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
        label = (ctx.metadata or {}).get("answerability")
        if label not in _ELIGIBLE_LABELS:
            return {}
        return await super().evaluate(ctx)
