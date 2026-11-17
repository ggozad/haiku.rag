import re
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

FINAL_ANSWER = "Final Answer:"

# Video-MME-v2's extract_characters_regex_v2 (evaluation/test_video_mme_v2.py).
_ANSWER_PREFIXES = (
    FINAL_ANSWER,
    "The best answer is",
    "The correct answer is",
    "The answer is",
    "The answer",
    "The best option is",
    "The correct option is",
    "Best answer:",
    "Best option:",
    "Answer:",
    "Option:",
)


def extract_letter(text: str) -> str:
    """The option letter, read after the last `Final Answer:` when there is one."""
    if FINAL_ANSWER in text:
        text = text.rsplit(FINAL_ANSWER, 1)[1]
    text = text.strip()
    for prefix in _ANSWER_PREFIXES:
        text = text.replace(prefix, "")
    match = re.search(r"[A-H]", text)
    return match[0] if match else ""


@dataclass
class LetterMatchEvaluator(Evaluator):
    """Scores 1.0 when the answer's option letter is the gold letter."""

    def get_default_evaluation_name(self) -> str:
        return "letter_match"

    def evaluate(self, ctx: EvaluatorContext) -> float:
        return 1.0 if extract_letter(str(ctx.output)) == ctx.expected_output else 0.0
