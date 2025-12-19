import asyncio

import dspy
import nest_asyncio

from evaluations.evaluators import OptimizationJudge
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.qa import get_qa_agent

# Allow nested event loops (needed because DSPy runs its own event loop)
nest_asyncio.apply()


class QAMetric:
    """
    DSPy metric that evaluates QA performance using the real QuestionAnswerAgent.

    This metric:
    1. Runs the actual pydantic-ai QuestionAnswerAgent with the candidate prompt
    2. Uses OptimizationJudge to score on correctness AND helpfulness
    3. Returns a continuous score from 0.0 to 1.0
    """

    def __init__(
        self,
        rag: HaikuRAG,
        config: AppConfig,
        judge: OptimizationJudge,
    ):
        self.rag = rag
        self.config = config
        self.judge = judge
        self._current_prompt: str | None = None

    def set_prompt(self, prompt: str) -> None:
        """Set the candidate prompt to evaluate."""
        self._current_prompt = prompt

    def __call__(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: object = None,
    ) -> float:
        """
        Evaluate the candidate prompt on a single example.

        Args:
            example: DSPy example with 'question' and 'expected_answer' fields
            prediction: DSPy prediction (ignored - we run our own agent)
            trace: Optional trace (unused)

        Returns:
            Score from 0.0 to 1.0 based on correctness and helpfulness
        """
        if self._current_prompt is None:
            raise RuntimeError("No prompt set. Call set_prompt() before evaluation.")

        question = example.question
        expected_answer = example.expected_answer

        # Run the real QuestionAnswerAgent with the candidate prompt
        qa_agent = get_qa_agent(
            self.rag,
            config=self.config,
            system_prompt=self._current_prompt,
        )

        answer, _ = asyncio.get_event_loop().run_until_complete(
            qa_agent.answer(question)
        )

        # Use OptimizationJudge to score on correctness and helpfulness
        score = asyncio.get_event_loop().run_until_complete(
            self.judge.score_answer(question, answer, expected_answer)
        )

        return score
