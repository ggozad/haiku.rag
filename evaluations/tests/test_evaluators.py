import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_evals.evaluators import EvaluatorContext

from evaluations.evaluators import REFUSAL_RUBRIC, RefusalJudge

from evaluations.evaluators.map import MAPEvaluator
from evaluations.evaluators.number_match import NumberMatchEvaluator
from evaluations.evaluators.retrieval import NDCGEvaluator, RecallEvaluator


class TestMAPEvaluator:
    def setup_method(self) -> None:
        self.evaluator = MAPEvaluator()

    def _make_ctx(
        self, relevant_uris: list[str], retrieved_uris: list[str]
    ) -> MagicMock:
        ctx = MagicMock()
        ctx.metadata = {"relevant_uris": relevant_uris}
        ctx.output = retrieved_uris
        return ctx

    def test_perfect_single_doc(self) -> None:
        ctx = self._make_ctx(["doc1"], ["doc1", "doc2", "doc3"])
        assert self.evaluator.evaluate(ctx) == 1.0

    def test_perfect_two_docs(self) -> None:
        # Both relevant at positions 1 and 2: P@1=1/1, P@2=2/2 → AP = (1+1)/2 = 1.0
        ctx = self._make_ctx(["doc1", "doc2"], ["doc1", "doc2", "doc3"])
        assert self.evaluator.evaluate(ctx) == 1.0

    def test_one_relevant_at_second_position(self) -> None:
        # 1 relevant doc at position 2: P@2=1/2 → AP = 0.5/1 = 0.5
        ctx = self._make_ctx(["doc2"], ["doc1", "doc2", "doc3"])
        assert self.evaluator.evaluate(ctx) == 0.5

    def test_two_relevant_with_gap(self) -> None:
        # Relevant at positions 1 and 3: P@1=1/1, P@3=2/3 → AP = (1 + 2/3)/2
        ctx = self._make_ctx(["doc1", "doc3"], ["doc1", "doc2", "doc3"])
        expected = (1.0 + 2 / 3) / 2
        assert self.evaluator.evaluate(ctx) == pytest.approx(expected)

    def test_no_relevant_found(self) -> None:
        ctx = self._make_ctx(["doc_x"], ["doc1", "doc2", "doc3"])
        assert self.evaluator.evaluate(ctx) == 0.0

    def test_empty_retrieved(self) -> None:
        ctx = self._make_ctx(["doc1"], [])
        assert self.evaluator.evaluate(ctx) == 0.0

    def test_none_metadata(self) -> None:
        ctx = MagicMock()
        ctx.metadata = None
        ctx.output = ["doc1"]
        assert self.evaluator.evaluate(ctx) == 0.0

    def test_empty_relevant_uris(self) -> None:
        ctx = self._make_ctx([], ["doc1", "doc2"])
        assert self.evaluator.evaluate(ctx) == 0.0


def _retrieval_ctx(relevant_uris: list[str], retrieved_uris: list[str]) -> MagicMock:
    ctx = MagicMock()
    ctx.metadata = {"relevant_uris": relevant_uris}
    ctx.output = retrieved_uris
    return ctx


class TestRecallEvaluator:
    def test_evaluation_name_includes_k(self) -> None:
        assert RecallEvaluator(k=5).get_default_evaluation_name() == "recall_5"
        assert RecallEvaluator(k=10).get_default_evaluation_name() == "recall_10"

    def test_all_relevant_within_k(self) -> None:
        ctx = _retrieval_ctx(["a", "b"], ["a", "b", "c"])
        assert RecallEvaluator(k=5).evaluate(ctx) == 1.0

    def test_partial_recall(self) -> None:
        ctx = _retrieval_ctx(["a", "b"], ["a", "c", "d"])
        assert RecallEvaluator(k=3).evaluate(ctx) == 0.5

    def test_relevant_beyond_k_not_counted(self) -> None:
        ctx = _retrieval_ctx(["a"], ["b", "c", "d", "e", "f", "a"])
        assert RecallEvaluator(k=5).evaluate(ctx) == 0.0
        assert RecallEvaluator(k=10).evaluate(ctx) == 1.0

    def test_empty_relevant(self) -> None:
        ctx = _retrieval_ctx([], ["a"])
        assert RecallEvaluator(k=5).evaluate(ctx) == 0.0

    def test_none_metadata(self) -> None:
        ctx = MagicMock()
        ctx.metadata = None
        ctx.output = ["a"]
        assert RecallEvaluator(k=5).evaluate(ctx) == 0.0


class TestNDCGEvaluator:
    def test_evaluation_name_includes_k(self) -> None:
        assert NDCGEvaluator(k=5).get_default_evaluation_name() == "ndcg_5"

    def test_perfect_ranking(self) -> None:
        ctx = _retrieval_ctx(["a", "b"], ["a", "b", "c"])
        assert NDCGEvaluator(k=5).evaluate(ctx) == pytest.approx(1.0)

    def test_single_relevant_at_rank_two(self) -> None:
        # DCG = 1/log2(3); IDCG = 1/log2(2) = 1
        ctx = _retrieval_ctx(["a"], ["b", "a"])
        expected = 1 / math.log2(3)
        assert NDCGEvaluator(k=5).evaluate(ctx) == pytest.approx(expected)

    def test_two_relevant_with_gap(self) -> None:
        # Relevant at ranks 1 and 3: DCG = 1 + 1/log2(4) = 1.5
        # IDCG = 1 + 1/log2(3)
        ctx = _retrieval_ctx(["a", "b"], ["a", "c", "b"])
        expected = 1.5 / (1 + 1 / math.log2(3))
        assert NDCGEvaluator(k=3).evaluate(ctx) == pytest.approx(expected)

    def test_relevant_beyond_k_not_counted(self) -> None:
        ctx = _retrieval_ctx(["a"], ["b", "c", "d", "e", "f", "a"])
        assert NDCGEvaluator(k=5).evaluate(ctx) == 0.0

    def test_ideal_dcg_capped_at_k(self) -> None:
        # 3 relevant but k=2: IDCG uses only the top-2 ideal ranks, so a
        # retrieval with both top-2 slots relevant scores 1.0.
        ctx = _retrieval_ctx(["a", "b", "c"], ["a", "b"])
        assert NDCGEvaluator(k=2).evaluate(ctx) == pytest.approx(1.0)

    def test_empty_relevant(self) -> None:
        ctx = _retrieval_ctx([], ["a"])
        assert NDCGEvaluator(k=5).evaluate(ctx) == 0.0

    def test_none_metadata(self) -> None:
        ctx = MagicMock()
        ctx.metadata = None
        ctx.output = ["a"]
        assert NDCGEvaluator(k=5).evaluate(ctx) == 0.0


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


class TestTranscriptLLMJudge:
    @pytest.mark.asyncio
    async def test_conversation_inputs_judged_as_transcript(self) -> None:
        from evaluations.config import ConversationInput, Turn
        from evaluations.evaluators import TranscriptLLMJudge

        judge = TranscriptLLMJudge(
            rubric="rubric",
            include_input=True,
            include_expected_output=True,
            model="test",
        )
        conversation = ConversationInput(
            turns=[
                Turn(speaker="user", text="q1"),
                Turn(speaker="agent", text="a1"),
                Turn(speaker="user", text="q2"),
            ]
        )
        grading = MagicMock(score=None, pass_=True, reason="ok")
        with patch(
            "pydantic_evals.evaluators.llm_as_a_judge.judge_input_output_expected",
            new_callable=AsyncMock,
            return_value=grading,
        ) as judge_call:
            await judge.evaluate(_evaluator_ctx(conversation))

        assert judge_call.await_args is not None
        assert judge_call.await_args.args[0] == "user: q1\nagent: a1\nuser: q2"

    @pytest.mark.asyncio
    async def test_string_inputs_pass_through(self) -> None:
        from evaluations.evaluators import TranscriptLLMJudge

        judge = TranscriptLLMJudge(
            rubric="rubric",
            include_input=True,
            include_expected_output=True,
            model="test",
        )
        grading = MagicMock(score=None, pass_=True, reason="ok")
        with patch(
            "pydantic_evals.evaluators.llm_as_a_judge.judge_input_output_expected",
            new_callable=AsyncMock,
            return_value=grading,
        ) as judge_call:
            await judge.evaluate(_evaluator_ctx("plain question"))

        assert judge_call.await_args is not None
        assert judge_call.await_args.args[0] == "plain question"


class TestRefusalJudge:
    def _judge(self) -> RefusalJudge:
        return RefusalJudge(
            rubric=REFUSAL_RUBRIC,
            model="test",
            assertion={"evaluation_name": "refused", "include_reason": False},
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("label", ["ANSWERABLE", "UNANSWERABLE"])
    async def test_judges_eligible_labels(self, label: str) -> None:
        grading = MagicMock(score=None, pass_=True, reason=None)
        with patch(
            "pydantic_evals.evaluators.llm_as_a_judge.judge_output",
            new_callable=AsyncMock,
            return_value=grading,
        ) as judge_call:
            result = await self._judge().evaluate(
                _evaluator_ctx("q", metadata={"answerability": label})
            )

        judge_call.assert_awaited_once()
        assert result == {"refused": True}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "metadata", [{"answerability": "PARTIAL"}, {"answerability": None}, {}, None]
    )
    async def test_ineligible_turns_skip_the_judge(self, metadata) -> None:
        with patch(
            "pydantic_evals.evaluators.llm_as_a_judge.judge_output",
            new_callable=AsyncMock,
        ) as judge_call:
            result = await self._judge().evaluate(_evaluator_ctx("q", metadata))

        judge_call.assert_not_awaited()
        assert result == {}


class TestNumberMatchEvaluator:
    def setup_method(self) -> None:
        self.evaluator = NumberMatchEvaluator()

    def _make_ctx(self, expected: str, output: str) -> MagicMock:
        ctx = MagicMock()
        ctx.expected_output = expected
        ctx.output = output
        return ctx

    def test_exact(self) -> None:
        assert self.evaluator.evaluate(self._make_ctx("127.4", "127.4")) == 1.0

    def test_within_tolerance(self) -> None:
        ctx = self._make_ctx("127.4", "about $127.40 per transaction")
        assert self.evaluator.evaluate(ctx) == 1.0

    def test_outside_tolerance(self) -> None:
        ctx = self._make_ctx("127.4", "the answer is 150")
        assert self.evaluator.evaluate(ctx) == 0.0

    def test_picks_matching_candidate_among_many(self) -> None:
        ctx = self._make_ctx("50.3", "In 2008 it grew from 27.0 to 50.3 percent")
        assert self.evaluator.evaluate(ctx) == 1.0

    def test_percent_answer_matches_decimal_gold(self) -> None:
        ctx = self._make_ctx("0.935", "the cumulative total return was 93.5%")
        assert self.evaluator.evaluate(ctx) == 1.0

    def test_percent_answer_matches_percent_gold(self) -> None:
        ctx = self._make_ctx("24.691358024691358", "approximately 24.69% of production")
        assert self.evaluator.evaluate(ctx) == 1.0

    def test_non_numeric_prediction(self) -> None:
        ctx = self._make_ctx("127.4", "I cannot determine the value")
        assert self.evaluator.evaluate(ctx) == 0.0

    def test_non_numeric_gold(self) -> None:
        ctx = self._make_ctx("not a number", "127.4")
        assert self.evaluator.evaluate(ctx) == 0.0

    def test_negative_match(self) -> None:
        ctx = self._make_ctx("-12.3", "the change was (12.3)")
        assert self.evaluator.evaluate(ctx) == 1.0

    def test_sign_insensitive_against_inconsistent_gold(self) -> None:
        # gold stores this decrease as +0.2; model declares the signed -0.2
        ctx = self._make_ctx("0.1999999999999993", "declined 0.2 pp\nANSWER: -0.2")
        assert self.evaluator.evaluate(ctx) == 1.0

    def test_bare_percent_matches_decimal_gold(self) -> None:
        # model declares the percentage without a % sign; gold is the decimal
        ctx = self._make_ctx("0.3781", "growth was 37.81%\nANSWER: 37.81")
        assert self.evaluator.evaluate(ctx) == 1.0

    def test_scale_mismatch_does_not_flip_genuine_error(self) -> None:
        ctx = self._make_ctx("30.443", "ANSWER: 2330.8%")
        assert self.evaluator.evaluate(ctx) == 0.0

    def test_thousands_convention(self) -> None:
        # gold is in thousands; model gives the full-dollar figure
        ctx = self._make_ctx("4575515.0", "...\nANSWER: 4575515000")
        assert self.evaluator.evaluate(ctx) == 1.0

    def test_thousands_convention_reversed(self) -> None:
        ctx = self._make_ctx(
            "46.30434782608695", "fair value per share\nANSWER: 46304.35"
        )
        assert self.evaluator.evaluate(ctx) == 1.0

    def test_answer_line_ignores_reasoning_distractors(self) -> None:
        # gold matches a distractor in the body, but the declared answer is wrong
        ctx = self._make_ctx(
            "0.728",
            "Finished goods were 72.8% of inventory.\nANSWER: 82.8%",
        )
        assert self.evaluator.evaluate(ctx) == 0.0

    def test_answer_line_used_when_correct(self) -> None:
        ctx = self._make_ctx(
            "0.935",
            "The graph shows growth to 193.5.\n\nANSWER: 93.5%",
        )
        assert self.evaluator.evaluate(ctx) == 1.0

    def test_falls_back_to_full_text_without_answer_line(self) -> None:
        ctx = self._make_ctx("127.4", "The average works out to $127.40 each.")
        assert self.evaluator.evaluate(ctx) == 1.0
