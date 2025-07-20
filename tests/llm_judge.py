import json

from ollama import AsyncClient
from pydantic import BaseModel

from haiku.rag.config import Config


class LLMJudgeResponseSchema(BaseModel):
    equivalent: bool


class LLMJudge:
    """LLM-as-judge for evaluating answer equivalence using Ollama."""

    def __init__(self, model: str = Config.QA_MODEL):
        self.model = model
        self.client = AsyncClient(host=Config.OLLAMA_BASE_URL)

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
            Dictionary with judgment result:
            - equivalent: bool indicating if answers are equivalent
            - explanation: str explaining the reasoning
            - score: str rating from 1-5
        """

        prompt = f"""You are an expert evaluator determining whether two answers to the same question are semantically equivalent.

QUESTION: {question}

GENERATED ANSWER: {answer}

EXPECTED ANSWER: {expected_answer}

EVALUATION CRITERIA:
Rate as EQUIVALENT (true) if:
✓ Both answers contain the same core factual information
✓ Both directly address the question asked
✓ The key claims and conclusions are consistent
✓ Any additional detail in one answer doesn't contradict the other

Rate as NOT EQUIVALENT (false) if:
✗ Factual contradictions exist between the answers
✗ One answer fails to address the core question
✗ Key information is missing from one answer that changes the meaning
✗ The answers lead to different conclusions or implications

GUIDELINES:
- Ignore minor differences in phrasing, style, or formatting
- Focus on semantic meaning rather than exact wording
- Consider both answers correct if they convey the same essential information
- Be tolerant of different levels of detail if the core answer is preserved
- Evaluate based on what a person asking this question would need to know

Respond with JSON containing only: {{"equivalent": true}} or {{"equivalent": false}}"""

        response = await self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format=LLMJudgeResponseSchema.model_json_schema(),
            think=False,
        )

        answer = response["message"]["content"].strip()
        try:
            res = json.loads(answer)
            assert "equivalent" in res, "Response must contain 'equivalent' key"
            return res["equivalent"]
        except json.JSONDecodeError:
            assert False, "Response is not valid JSON"
