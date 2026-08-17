from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_evals.evaluators import EvaluatorContext
from pydantic_evals.evaluators.evaluator import EvaluationReason

from evaluations.evaluators.conversation import ConversationEvaluator


def _ctx(
    questions: list[str],
    answers: list[str],
    turns: list[dict],
    turn_cited_uris: list[list[str]] | None = None,
) -> EvaluatorContext:
    return EvaluatorContext(
        name="conv",
        inputs=questions,
        metadata={"conversation_id": "conv1", "turns": turns},
        expected_output=None,
        output=answers,
        duration=0.0,
        _span_tree=MagicMock(),
        attributes={"turn_cited_uris": turn_cited_uris or [[] for _ in answers]},
        metrics={},
    )


def _grading(pass_: bool) -> MagicMock:
    return MagicMock(score=None, pass_=pass_, reason=None)


class TestConversationEvaluator:
    @pytest.mark.asyncio
    async def test_per_turn_scores_and_aggregates(self) -> None:
        evaluator = ConversationEvaluator(rubric="equivalence rubric", model="test")
        ctx = _ctx(
            questions=["q1", "q2", "q3"],
            answers=["a1", "a2", "a3"],
            turns=[
                {
                    "reference": "r1",
                    "answerability": "ANSWERABLE",
                    "relevant_uris": ["p1", "p2"],
                },
                {"reference": "r2", "answerability": "UNANSWERABLE"},
                {
                    "reference": "r3",
                    "answerability": "PARTIAL",
                    "relevant_uris": ["p3"],
                },
            ],
            turn_cited_uris=[["p1"], [], ["p3"]],
        )

        with (
            patch(
                "evaluations.evaluators.conversation.judge_input_output_expected",
                new_callable=AsyncMock,
                side_effect=[_grading(True), _grading(False), _grading(True)],
            ) as judge_answer,
            patch(
                "evaluations.evaluators.conversation.judge_output",
                new_callable=AsyncMock,
                side_effect=[_grading(False), _grading(True)],
            ) as judge_refusal,
        ):
            result = await evaluator.evaluate(ctx)

        assert isinstance(result, dict)
        assert result == {
            "turn_pass_rate": pytest.approx(2 / 3),
            "turns_passed": 2,
            "turns_judged": 3,
            "turns_total": 3,
            "cited_map": pytest.approx((0.5 + 1.0) / 2),
            "cited_eligible": 2,
            "true_refusals": 1,
            "false_refusals": 0,
            "unanswerable_turns": 1,
            "turn_1_pass": EvaluationReason(value=True, reason=None),
            "turn_2_pass": EvaluationReason(value=False, reason=None),
            "turn_3_pass": EvaluationReason(value=True, reason=None),
            "turn_1_refused": False,
            "turn_2_refused": True,
            "turn_1_cited_ap": 0.5,
            "turn_3_cited_ap": 1.0,
        }
        # Refusal judged only on ANSWERABLE/UNANSWERABLE turns.
        assert judge_refusal.await_count == 2
        assert judge_answer.await_count == 3

    @pytest.mark.asyncio
    async def test_judge_sees_live_transcript(self) -> None:
        """Turn 2 is judged against the conversation so far with OUR answer to
        turn 1, not the reference."""
        evaluator = ConversationEvaluator(rubric="rubric", model="test")
        ctx = _ctx(
            questions=["q1", "q2"],
            answers=["my a1", "my a2"],
            turns=[
                {"reference": "r1", "answerability": "ANSWERABLE"},
                {"reference": "r2", "answerability": "ANSWERABLE"},
            ],
        )

        with (
            patch(
                "evaluations.evaluators.conversation.judge_input_output_expected",
                new_callable=AsyncMock,
                return_value=_grading(True),
            ) as judge_answer,
            patch(
                "evaluations.evaluators.conversation.judge_output",
                new_callable=AsyncMock,
                return_value=_grading(False),
            ),
        ):
            await evaluator.evaluate(ctx)

        second_call = judge_answer.await_args_list[1]
        transcript, answer, reference = second_call.args[:3]
        assert transcript == "user: q1\nagent: my a1\nuser: q2"
        assert answer == "my a2"
        assert reference == "r2"

    @pytest.mark.asyncio
    async def test_no_citation_scores_without_eligible_turns(self) -> None:
        evaluator = ConversationEvaluator(rubric="rubric", model="test")
        ctx = _ctx(
            questions=["q1"],
            answers=["a1"],
            turns=[{"reference": "r1", "answerability": "UNANSWERABLE"}],
        )

        with (
            patch(
                "evaluations.evaluators.conversation.judge_input_output_expected",
                new_callable=AsyncMock,
                return_value=_grading(False),
            ),
            patch(
                "evaluations.evaluators.conversation.judge_output",
                new_callable=AsyncMock,
                return_value=_grading(True),
            ),
        ):
            result = await evaluator.evaluate(ctx)

        assert result == {
            "turn_pass_rate": 0.0,
            "turns_passed": 0,
            "turns_judged": 1,
            "turns_total": 1,
            "cited_eligible": 0,
            "true_refusals": 1,
            "false_refusals": 0,
            "unanswerable_turns": 1,
            "turn_1_pass": EvaluationReason(value=False, reason=None),
            "turn_1_refused": True,
        }

    @pytest.mark.asyncio
    async def test_mismatched_arrays_raise(self) -> None:
        evaluator = ConversationEvaluator(rubric="rubric", model="test")
        ctx = _ctx(
            questions=["q1", "q2"],
            answers=["a1"],
            turns=[{"reference": "r1", "answerability": "ANSWERABLE"}],
        )

        with pytest.raises(ValueError, match="conversation arrays disagree"):
            await evaluator.evaluate(ctx)

    @pytest.mark.asyncio
    async def test_judge_error_voids_one_turn_not_the_conversation(self) -> None:
        evaluator = ConversationEvaluator(rubric="rubric", model="test")
        ctx = _ctx(
            questions=["q1", "q2", "q3"],
            answers=["a1", "a2", "a3"],
            turns=[
                {"reference": "r1", "answerability": "ANSWERABLE"},
                {"reference": "r2", "answerability": "ANSWERABLE"},
                {"reference": "r3", "answerability": "ANSWERABLE"},
            ],
        )

        with (
            patch(
                "evaluations.evaluators.conversation.judge_input_output_expected",
                new_callable=AsyncMock,
                side_effect=[
                    _grading(True),
                    RuntimeError("token limit exceeded"),
                    _grading(True),
                ],
            ),
            patch(
                "evaluations.evaluators.conversation.judge_output",
                new_callable=AsyncMock,
                return_value=_grading(False),
            ),
        ):
            result = await evaluator.evaluate(ctx)

        assert result == {
            "turn_pass_rate": 1.0,
            "turns_passed": 2,
            "turns_judged": 2,
            "turns_total": 3,
            "cited_eligible": 0,
            "true_refusals": 0,
            "false_refusals": 0,
            "unanswerable_turns": 0,
            "turn_1_pass": EvaluationReason(value=True, reason=None),
            "turn_3_pass": EvaluationReason(value=True, reason=None),
            "turn_2_judge_error": "token limit exceeded",
            "turn_1_refused": False,
            "turn_2_refused": False,
            "turn_3_refused": False,
        }

    @pytest.mark.asyncio
    async def test_refusal_judge_error_skips_refusal_verdict_only(self) -> None:
        evaluator = ConversationEvaluator(rubric="rubric", model="test")
        ctx = _ctx(
            questions=["q1"],
            answers=["a1"],
            turns=[{"reference": "r1", "answerability": "UNANSWERABLE"}],
        )

        with (
            patch(
                "evaluations.evaluators.conversation.judge_input_output_expected",
                new_callable=AsyncMock,
                return_value=_grading(True),
            ),
            patch(
                "evaluations.evaluators.conversation.judge_output",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
        ):
            result = await evaluator.evaluate(ctx)

        assert result == {
            "turn_pass_rate": 1.0,
            "turns_passed": 1,
            "turns_judged": 1,
            "turns_total": 1,
            "cited_eligible": 0,
            "true_refusals": 0,
            "false_refusals": 0,
            "unanswerable_turns": 0,
            "turn_1_pass": EvaluationReason(value=True, reason=None),
            "turn_1_judge_error": "boom",
        }
