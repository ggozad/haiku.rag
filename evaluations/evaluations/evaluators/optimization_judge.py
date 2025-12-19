from pydantic import BaseModel, Field
from pydantic_ai import Agent

from haiku.rag.config.models import AppConfig, ModelConfig
from haiku.rag.utils import get_model

OPTIMIZATION_RUBRIC = """You are evaluating an answer for both CORRECTNESS and HELPFULNESS.

SCORING CRITERIA:

**Correctness (is the core information accurate?):**
- Does the answer contain the same factual information as the expected answer?
- Are there any contradictions or errors?
- Does it address the question asked?

**Helpfulness (is the answer useful to someone asking this question?):**
- Is the answer complete enough to be useful?
- Does it provide sufficient context or explanation?
- Would a user feel their question was adequately answered?

SCORING GUIDE:
- **0.0**: Wrong answer - factually incorrect or doesn't address the question
- **0.25**: Partially correct but missing key information
- **0.5**: Correct but too terse to be helpful (e.g., single word when explanation needed)
- **0.75**: Correct and adequate - answers the question sufficiently
- **1.0**: Correct and helpful - complete, well-explained, useful response

GUIDELINES:
- Correctness is primary - a wrong answer cannot score above 0.25
- A terse but correct answer should score around 0.5
- Reward answers that would genuinely help someone understand the topic
- Consider the question type: factoid questions may need less explanation than how-to questions
"""


class OptimizationScore(BaseModel):
    reasoning: str = Field(description="Brief explanation of the score")
    score: float = Field(ge=0.0, le=1.0, description="Score from 0.0 to 1.0")


class OptimizationJudge:
    """LLM judge for optimization that returns continuous scores rewarding helpfulness."""

    def __init__(self, model: str = "gpt-oss", config: AppConfig | None = None):
        model_config = ModelConfig(provider="ollama", name=model, enable_thinking=False)
        model_obj = get_model(model_config, config)

        self._agent = Agent(
            model=model_obj,
            output_type=OptimizationScore,
            system_prompt=OPTIMIZATION_RUBRIC,
            retries=3,
        )

    async def score_answer(
        self, question: str, answer: str, expected_answer: str
    ) -> float:
        """
        Score an answer on both correctness and helpfulness.

        Args:
            question: The original question
            answer: The generated answer to evaluate
            expected_answer: The reference/expected answer

        Returns:
            float score from 0.0 to 1.0
        """
        prompt = f"""QUESTION: {question}

GENERATED ANSWER: {answer}

EXPECTED ANSWER: {expected_answer}

Score the generated answer on correctness and helpfulness."""

        result = await self._agent.run(prompt)
        return result.output.score
