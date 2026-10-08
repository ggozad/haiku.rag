import asyncio
import re
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext
from pydantic_evals.evaluators.evaluator import EvaluatorOutput

# Verbatim from EnterpriseRAG-Bench (MIT), src/prompts/answer_evaluation.py.
FACT_VALIDATOR_PROMPT = """
You are an answer validator. Given an answer and a statement, determine if the answer is consistent with and contains the information in the statement. \
The answer may contain more details or richer information than the statement but as long as it does not contradict the statement, this is valid. \
If there are negative statements such as "The answer must not say...", it is valid if the answer mentions the statement with caveats or qualifications. \
It is valid if additional context is shared for completeness however hallucinations are not allowed. \
Output a simple yes or no for if the answer is consistent with and contains the information in the statement.

## Answer
```
{answer}
```

## Statement
```
{statement}
```

CRITICAL: output only a simple yes if the answer is consistent with the statement or a no if the answer does not contain the information in the statement or contradicts the statement.
""".strip()


@dataclass
class AnswerFactsJudge(Evaluator):
    """Scores `answer_completeness`: the fraction of the case's `answer_facts` the
    output supports, judging each fact in its own call. Cases without facts get
    no score and cost no judge call."""

    model: Model | str

    async def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
        facts = (ctx.metadata or {}).get("answer_facts")
        if not facts:
            return {}
        verdicts = await asyncio.gather(
            *(self._supports(str(ctx.output), fact) for fact in facts)
        )
        missing = [fact for fact, supported in zip(facts, verdicts) if not supported]
        return {
            "answer_completeness": EvaluationReason(
                value=1 - len(missing) / len(facts),
                reason="Missing: " + " | ".join(missing) if missing else None,
            )
        }

    async def _supports(self, answer: str, fact: str) -> bool:
        """A "yes" on the first line of the reply, as their scorer reads it."""
        prompt = FACT_VALIDATOR_PROMPT.format(answer=answer, statement=fact)
        reply = (await Agent(self.model).run(prompt)).output.strip()
        first_line = reply.splitlines()[0] if reply else ""
        return re.search(r"\byes\b", first_line, re.IGNORECASE) is not None

    def build_serialization_arguments(self) -> dict[str, Any]:
        arguments = super().build_serialization_arguments()
        if isinstance(self.model, Model):
            arguments["model"] = self.model.model_id
        return arguments
