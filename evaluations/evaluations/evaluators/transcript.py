from dataclasses import dataclass, replace

from pydantic_evals.evaluators import EvaluatorContext, LLMJudge
from pydantic_evals.evaluators.evaluator import EvaluatorOutput

from evaluations.config import ConversationInput


@dataclass
class TranscriptLLMJudge(LLMJudge):
    """LLMJudge that shows conversation inputs as a readable transcript.

    pydantic-evals serializes custom input models as JSON in the judge prompt;
    a ConversationInput is rendered as `speaker: text` lines instead. Plain
    string inputs pass through unchanged.
    """

    async def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
        if isinstance(ctx.inputs, ConversationInput):
            ctx = replace(ctx, inputs=ctx.inputs.transcript)
        return await super().evaluate(ctx)
