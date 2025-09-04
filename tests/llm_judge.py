from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from haiku.rag.config import Config

# Shared rubric/prompt for answer equivalence evaluation
ANSWER_EQUIVALENCE_RUBRIC = """You are evaluating whether two answers to the same question are semantically equivalent.

EVALUATION CRITERIA:
Rate as EQUIVALENT if:
✓ Both answers contain the same core factual information
✓ Both directly address the question asked
✓ The key claims and conclusions are consistent
✓ Any additional detail in one answer doesn't contradict the other

Rate as NOT EQUIVALENT if:
✗ Factual contradictions exist between the answers
✗ One answer fails to address the core question
✗ Key information is missing that changes the meaning
✗ The answers lead to different conclusions or implications

GUIDELINES:
- Ignore minor differences in phrasing, style, or formatting
- Focus on semantic meaning rather than exact wording
- Consider both answers correct if they convey the same essential information
- Be tolerant of different levels of detail if the core answer is preserved
- Evaluate based on what a person asking this question would need to know
/no_think"""


class LLMJudgeResponseSchema(BaseModel):
    equivalent: bool


class LLMJudge:
    """LLM-as-judge for evaluating answer equivalence using Pydantic AI."""

    def __init__(self, model: str = "qwen3"):
        # Create Ollama model
        ollama_model = OpenAIChatModel(
            model_name=model,
            provider=OllamaProvider(base_url=f"{Config.OLLAMA_BASE_URL}/v1"),
        )

        # Create Pydantic AI agent
        self._agent = Agent(
            model=ollama_model,
            output_type=LLMJudgeResponseSchema,
            system_prompt=ANSWER_EQUIVALENCE_RUBRIC,
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
