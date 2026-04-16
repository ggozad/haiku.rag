from pydantic import BaseModel
from pydantic_ai import Agent

from haiku.rag.config.models import AppConfig, ModelConfig
from haiku.rag.utils import get_model

ANSWER_EQUIVALENCE_RUBRIC = """You are evaluating whether a generated answer is equivalent to an expected answer for a given question.

EVALUATION CRITERIA:
Rate as EQUIVALENT if:
✓ The generated answer contains the core factual information from the expected answer
✓ The generated answer directly addresses the question asked
✓ The key claims and conclusions are consistent
✓ The generated answer may include additional correct details not in the expected answer — this is fine

Rate as NOT EQUIVALENT if:
✗ The generated answer contradicts facts in the expected answer
✗ The generated answer fails to address the core question
✗ Key information from the expected answer is missing in a way that changes the meaning
✗ The answers lead to different conclusions or actions

GUIDELINES:
- The evaluation is asymmetric: judge the generated answer against the expected answer, not the other way around
- A generated answer that is MORE detailed or comprehensive than the expected answer is EQUIVALENT, as long as it doesn't contradict it
- If the expected answer is incomplete or narrow, do not penalize the generated answer for being broader
- Ignore differences in phrasing, style, or formatting
- Focus on whether a user would get the correct guidance from the generated answer
- Be tolerant of different levels of detail if the core answer is preserved
"""


class LLMJudgeResponseSchema(BaseModel):
    equivalent: bool


class LLMJudge:
    """LLM-as-judge for evaluating answer equivalence using Pydantic AI."""

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        config: AppConfig | None = None,
    ):
        if model_config is None:
            effective_config = config or AppConfig()
            model_config = effective_config.qa.model
        model_obj = get_model(model_config, config)

        # Create Pydantic AI agent
        self._agent: Agent[None, LLMJudgeResponseSchema] = Agent(  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
            model=model_obj,
            output_type=LLMJudgeResponseSchema,
            system_prompt=ANSWER_EQUIVALENCE_RUBRIC,
            retries=3,
        )

    async def judge_answers(
        self, question: str, answer: str, expected_answer: str
    ) -> bool:
        """
        Judge whether two answers are equivalent for a given question.

        Args:
            question: The original question
            answer: The generated answer to evaluate
            expected_answer: The reference/expected answer

        Returns:
            bool indicating if answers are equivalent
        """

        prompt = f"""QUESTION: {question}

GENERATED ANSWER: {answer}

EXPECTED ANSWER: {expected_answer}"""

        result = await self._agent.run(prompt)
        return result.output.equivalent
