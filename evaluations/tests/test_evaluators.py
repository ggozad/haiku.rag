import math
from unittest.mock import MagicMock

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals.evaluators import EvaluatorContext

from evaluations.config import ConversationInput, Turn
from evaluations.evaluators import REFUSAL_RUBRIC, RefusalJudge
from evaluations.evaluators.map import MAPEvaluator
from evaluations.evaluators.number_match import NumberMatchEvaluator
from evaluations.evaluators.retrieval import NDCGEvaluator, RecallEvaluator


def _retrieval_ctx(relevant_uris: list[str], retrieved_uris: list[str]) -> MagicMock:
    ctx = MagicMock()
    ctx.metadata = {"relevant_uris": relevant_uris}
    ctx.output = retrieved_uris
    return ctx


def _ctx_without_metadata() -> MagicMock:
    ctx = MagicMock()
    ctx.metadata = None
    ctx.output = ["a"]
    return ctx


class TestMAPEvaluator:
    @pytest.mark.parametrize(
        ("relevant", "retrieved", "expected"),
        [
            (["doc1"], ["doc1", "doc2", "doc3"], 1.0),
            # P@1=1/1, P@2=2/2 → AP = (1+1)/2
            (["doc1", "doc2"], ["doc1", "doc2", "doc3"], 1.0),
            # P@2=1/2 → AP = 0.5/1
            (["doc2"], ["doc1", "doc2", "doc3"], 0.5),
            # P@1=1/1, P@3=2/3 → AP = (1 + 2/3)/2
            (["doc1", "doc3"], ["doc1", "doc2", "doc3"], (1.0 + 2 / 3) / 2),
            (["doc_x"], ["doc1", "doc2", "doc3"], 0.0),
            (["doc1"], [], 0.0),
            ([], ["doc1", "doc2"], 0.0),
        ],
        ids=[
            "perfect-single-doc",
            "perfect-two-docs",
            "one-relevant-at-second-position",
            "two-relevant-with-gap",
            "no-relevant-found",
            "empty-retrieved",
            "empty-relevant-uris",
        ],
    )
    def test_average_precision(
        self, relevant: list[str], retrieved: list[str], expected: float
    ) -> None:
        ctx = _retrieval_ctx(relevant, retrieved)
        assert MAPEvaluator().evaluate(ctx) == expected

    def test_none_metadata(self) -> None:
        assert MAPEvaluator().evaluate(_ctx_without_metadata()) == 0.0


class TestRecallEvaluator:
    def test_evaluation_name_includes_k(self) -> None:
        assert RecallEvaluator(k=5).get_default_evaluation_name() == "recall_5"
        assert RecallEvaluator(k=10).get_default_evaluation_name() == "recall_10"

    @pytest.mark.parametrize(
        ("k", "relevant", "retrieved", "expected"),
        [
            (5, ["a", "b"], ["a", "b", "c"], 1.0),
            (3, ["a", "b"], ["a", "c", "d"], 0.5),
            (5, ["a"], ["b", "c", "d", "e", "f", "a"], 0.0),
            (10, ["a"], ["b", "c", "d", "e", "f", "a"], 1.0),
            (5, [], ["a"], 0.0),
        ],
        ids=[
            "all-relevant-within-k",
            "partial",
            "relevant-beyond-k-not-counted",
            "relevant-within-larger-k",
            "empty-relevant",
        ],
    )
    def test_recall(
        self, k: int, relevant: list[str], retrieved: list[str], expected: float
    ) -> None:
        assert RecallEvaluator(k=k).evaluate(_retrieval_ctx(relevant, retrieved)) == (
            expected
        )

    def test_none_metadata(self) -> None:
        assert RecallEvaluator(k=5).evaluate(_ctx_without_metadata()) == 0.0


class TestNDCGEvaluator:
    def test_evaluation_name_includes_k(self) -> None:
        assert NDCGEvaluator(k=5).get_default_evaluation_name() == "ndcg_5"

    @pytest.mark.parametrize(
        ("k", "relevant", "retrieved", "expected"),
        [
            (5, ["a", "b"], ["a", "b", "c"], 1.0),
            # DCG = 1/log2(3), IDCG = 1/log2(2) = 1
            (5, ["a"], ["b", "a"], 1 / math.log2(3)),
            # DCG = 1 + 1/log2(4) = 1.5, IDCG = 1 + 1/log2(3)
            (3, ["a", "b"], ["a", "c", "b"], 1.5 / (1 + 1 / math.log2(3))),
            (5, ["a"], ["b", "c", "d", "e", "f", "a"], 0.0),
            # IDCG uses only the top-k ideal ranks
            (2, ["a", "b", "c"], ["a", "b"], 1.0),
            (5, [], ["a"], 0.0),
        ],
        ids=[
            "perfect-ranking",
            "single-relevant-at-rank-two",
            "two-relevant-with-gap",
            "relevant-beyond-k-not-counted",
            "ideal-dcg-capped-at-k",
            "empty-relevant",
        ],
    )
    def test_ndcg(
        self, k: int, relevant: list[str], retrieved: list[str], expected: float
    ) -> None:
        ctx = _retrieval_ctx(relevant, retrieved)
        assert NDCGEvaluator(k=k).evaluate(ctx) == pytest.approx(expected)

    def test_none_metadata(self) -> None:
        assert NDCGEvaluator(k=5).evaluate(_ctx_without_metadata()) == 0.0


def _evaluator_ctx(inputs: object, metadata: dict | None = None) -> EvaluatorContext:
    return EvaluatorContext(
        name="case",
        inputs=inputs,
        metadata=metadata,
        expected_output="expected",
        output="answer",
        duration=0.0,
        _span_tree=MagicMock(),
        attributes={},
        metrics={},
    )


class RecordingJudgeModel(FunctionModel):
    """A judge model that passes every case and records the prompts it was sent."""

    def __init__(self) -> None:
        super().__init__(self._grade)
        self.prompts: list[str] = []

    def _grade(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        part = request.parts[-1]
        assert isinstance(part, UserPromptPart) and isinstance(part.content, str)
        self.prompts.append(part.content)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {"reason": "ok", "pass": True, "score": 1.0},
                )
            ]
        )


class TestTranscriptLLMJudge:
    @pytest.mark.parametrize(
        ("inputs", "rendered"),
        [
            (
                ConversationInput(
                    turns=[
                        Turn(speaker="user", text="q1"),
                        Turn(speaker="agent", text="a1"),
                        Turn(speaker="user", text="q2"),
                    ]
                ),
                "user: q1\nagent: a1\nuser: q2",
            ),
            ("plain question", "plain question"),
        ],
        ids=["conversation-as-transcript", "string-passes-through"],
    )
    async def test_renders_the_input(self, inputs: object, rendered: str) -> None:
        from evaluations.evaluators import TranscriptLLMJudge

        model = RecordingJudgeModel()
        judge = TranscriptLLMJudge(
            rubric="rubric",
            include_input=True,
            include_expected_output=True,
            model=model,
        )
        await judge.evaluate(_evaluator_ctx(inputs))

        assert len(model.prompts) == 1
        assert f"<Input>\n{rendered}\n</Input>" in model.prompts[0]


class TestRefusalJudge:
    def _judge(self, model: RecordingJudgeModel) -> RefusalJudge:
        return RefusalJudge(
            rubric=REFUSAL_RUBRIC,
            model=model,
            assertion={"evaluation_name": "refused", "include_reason": False},
        )

    @pytest.mark.parametrize("label", ["ANSWERABLE", "UNANSWERABLE"])
    async def test_judges_eligible_labels(self, label: str) -> None:
        model = RecordingJudgeModel()
        result = await self._judge(model).evaluate(
            _evaluator_ctx("q", metadata={"answerability": label})
        )

        assert len(model.prompts) == 1
        assert result == {"refused": True}

    @pytest.mark.parametrize(
        "metadata", [{"answerability": "PARTIAL"}, {"answerability": None}, {}, None]
    )
    async def test_ineligible_turns_skip_the_judge(self, metadata) -> None:
        model = RecordingJudgeModel()
        result = await self._judge(model).evaluate(_evaluator_ctx("q", metadata))

        assert model.prompts == []
        assert result == {}


class TestNumberMatchEvaluator:
    @pytest.mark.parametrize(
        ("gold", "output", "score"),
        [
            ("127.4", "127.4", 1.0),
            ("127.4", "about $127.40 per transaction", 1.0),
            ("127.4", "the answer is 150", 0.0),
            ("50.3", "In 2008 it grew from 27.0 to 50.3 percent", 1.0),
            ("0.935", "the cumulative total return was 93.5%", 1.0),
            ("24.691358024691358", "approximately 24.69% of production", 1.0),
            ("127.4", "I cannot determine the value", 0.0),
            ("not a number", "127.4", 0.0),
            ("-12.3", "the change was (12.3)", 1.0),
            # gold stores this decrease as +0.2, the model declares -0.2
            ("0.1999999999999993", "declined 0.2 pp\nANSWER: -0.2", 1.0),
            # the model declares the percentage without a % sign
            ("0.3781", "growth was 37.81%\nANSWER: 37.81", 1.0),
            ("30.443", "ANSWER: 2330.8%", 0.0),
            # gold is in thousands, the model gives the full-dollar figure
            ("4575515.0", "...\nANSWER: 4575515000", 1.0),
            ("46.30434782608695", "fair value per share\nANSWER: 46304.35", 1.0),
            # gold matches a distractor in the body, the declared answer is wrong
            ("0.728", "Finished goods were 72.8% of inventory.\nANSWER: 82.8%", 0.0),
            ("0.935", "The graph shows growth to 193.5.\n\nANSWER: 93.5%", 1.0),
            ("127.4", "The average works out to $127.40 each.", 1.0),
        ],
        ids=[
            "exact",
            "within-tolerance",
            "outside-tolerance",
            "picks-matching-candidate-among-many",
            "percent-answer-matches-decimal-gold",
            "percent-answer-matches-percent-gold",
            "non-numeric-prediction",
            "non-numeric-gold",
            "negative-match",
            "sign-insensitive-against-inconsistent-gold",
            "bare-percent-matches-decimal-gold",
            "scale-mismatch-does-not-flip-genuine-error",
            "thousands-convention",
            "thousands-convention-reversed",
            "answer-line-ignores-reasoning-distractors",
            "answer-line-used-when-correct",
            "falls-back-to-full-text-without-answer-line",
        ],
    )
    def test_score(self, gold: str, output: str, score: float) -> None:
        ctx = MagicMock()
        ctx.expected_output = gold
        ctx.output = output
        assert NumberMatchEvaluator().evaluate(ctx) == score
