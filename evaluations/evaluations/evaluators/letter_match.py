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


def extract_letter(text: str, standalone: bool = False) -> str:
    """The option letter, read after the last `Final Answer:` when there is one.

    Without `standalone` it is the first capital A to H, as the benchmark reads
    it, so "Cannot determine" reads as C. With it, the answer must start with
    the letter, alone or in brackets or bold."""
    if FINAL_ANSWER in text:
        text = text.rsplit(FINAL_ANSWER, 1)[1]
    text = text.strip()
    for prefix in _ANSWER_PREFIXES:
        text = text.replace(prefix, "")
    pattern = r"^[\s*(\[]*([A-H])(?![A-Za-z])" if standalone else r"([A-H])"
    match = re.search(pattern, text)
    return match[1] if match else ""


@dataclass
class LetterMatchEvaluator(Evaluator):
    """Scores 1.0 when the answer's option letter is the gold letter."""

    standalone: bool = False

    def get_default_evaluation_name(self) -> str:
        return "letter_match"

    def evaluate(self, ctx: EvaluatorContext) -> float:
        letter = extract_letter(str(ctx.output), self.standalone)
        return 1.0 if letter == ctx.expected_output else 0.0
