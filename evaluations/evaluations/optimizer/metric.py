import asyncio

import dspy

from evaluations.evaluators import LLMJudge
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.qa import get_qa_agent


class QAMetric:
    """
    DSPy metric that evaluates QA performance using the real QuestionAnswerAgent.

    This metric:
    1. Runs the actual pydantic-ai QuestionAnswerAgent with the candidate prompt
    2. Uses LLMJudge to evaluate answer equivalence
    3. Returns 1.0 for equivalent answers, 0.0 otherwise
    """

    def __init__(
        self,
        rag: HaikuRAG,
        config: AppConfig,
        judge: LLMJudge,
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
            1.0 if answer is equivalent, 0.0 otherwise
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

        # Use LLMJudge to evaluate
        is_equivalent = asyncio.get_event_loop().run_until_complete(
            self.judge.judge_answers(question, answer, expected_answer)
        )

        return 1.0 if is_equivalent else 0.0
