from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_evals import Case

from gepa.core.adapter import EvaluationBatch

from evaluations.config import DatasetSpec
from evaluations.optimization import (
    EvalTrajectory,
    QAPromptAdapter,
    ReflectionLM,
    run_optimization,
)
from haiku.rag.config.models import AppConfig


@pytest.fixture
def sample_cases() -> list[Case[str, str, dict[str, str]]]:
    return [
        Case(
            name="q1",
            inputs="What is X?",
            expected_output="X is a thing.",
            metadata={"case_index": "1"},
        ),
        Case(
            name="q2",
            inputs="How does Y work?",
            expected_output="Y works by Z.",
            metadata={"case_index": "2"},
        ),
    ]


@pytest.fixture
def adapter(tmp_path: Path) -> QAPromptAdapter:
    return QAPromptAdapter(
        config=AppConfig(),
        db_path=tmp_path / "test.lancedb",
        judge_model=MagicMock(),
    )


class TestMakeReflectiveDataset:
    def test_builds_records_from_trajectories(self, adapter: QAPromptAdapter) -> None:
        trajectories = [
            EvalTrajectory(
                question="What is X?",
                expected_answer="X is a thing.",
                actual_answer="X is wrong.",
                score=0.2,
                judge_reason="Factually incorrect",
            ),
            EvalTrajectory(
                question="How does Y?",
                expected_answer="Y works by Z.",
                actual_answer="Y works by Z.",
                score=1.0,
                judge_reason=None,
            ),
        ]
        eval_batch: EvaluationBatch[EvalTrajectory, str | None] = EvaluationBatch(
            outputs=["X is wrong.", "Y works by Z."],
            scores=[0.2, 1.0],
            trajectories=trajectories,
        )

        result = adapter.make_reflective_dataset(
            {"instructions": "test"}, eval_batch, ["instructions"]
        )

        assert "instructions" in result
        records = result["instructions"]
        assert len(records) == 2

        assert records[0]["Inputs"]["question"] == "What is X?"
        assert records[0]["Generated Outputs"]["answer"] == "X is wrong."
        assert "Expected answer: X is a thing." in records[0]["Feedback"]
        assert "Score: 0.20" in records[0]["Feedback"]
        assert "Factually incorrect" in records[0]["Feedback"]

        assert records[1]["Inputs"]["question"] == "How does Y?"
        assert records[1]["Generated Outputs"]["answer"] == "Y works by Z."
        assert "Score: 1.00" in records[1]["Feedback"]
        assert "N/A" in records[1]["Feedback"]

    def test_returns_empty_when_no_trajectories(self, adapter: QAPromptAdapter) -> None:
        eval_batch: EvaluationBatch[EvalTrajectory, str | None] = EvaluationBatch(
            outputs=[], scores=[], trajectories=None
        )

        result = adapter.make_reflective_dataset(
            {"instructions": "test"}, eval_batch, ["instructions"]
        )

        assert result == {}

    def test_none_answer_becomes_no_answer(self, adapter: QAPromptAdapter) -> None:
        trajectories = [
            EvalTrajectory(
                question="What?",
                expected_answer="Answer.",
                actual_answer=None,
                score=0.0,
                judge_reason="Failed",
            ),
        ]
        eval_batch: EvaluationBatch[EvalTrajectory, str | None] = EvaluationBatch(
            outputs=[None],
            scores=[0.0],
            trajectories=trajectories,
        )

        result = adapter.make_reflective_dataset(
            {"instructions": "test"}, eval_batch, ["instructions"]
        )

        assert result["instructions"][0]["Generated Outputs"]["answer"] == "(no answer)"


class TestEvaluateAsync:
    @pytest.mark.asyncio
    async def test_returns_scores_and_outputs(
        self,
        adapter: QAPromptAdapter,
        sample_cases: list[Case[str, str, dict[str, str]]],
    ) -> None:
        mock_qa = AsyncMock()
        mock_qa.answer = AsyncMock(return_value=("X is a thing.", []))

        with (
            patch("evaluations.optimization.HaikuRAG") as mock_haiku_cls,
            patch("evaluations.optimization.get_qa_agent", return_value=mock_qa),
        ):
            mock_haiku = AsyncMock()
            mock_haiku_cls.return_value.__aenter__ = AsyncMock(return_value=mock_haiku)
            mock_haiku_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            adapter._judge = AsyncMock(return_value=(0.85, "Good answer"))  # type: ignore[method-assign]

            result = await adapter._evaluate_async(
                sample_cases, "test prompt", capture_traces=False
            )

        assert len(result.outputs) == 2
        assert len(result.scores) == 2
        assert all(o == "X is a thing." for o in result.outputs)
        assert all(s == 0.85 for s in result.scores)
        assert result.trajectories is None

    @pytest.mark.asyncio
    async def test_populates_trajectories_when_captured(
        self,
        adapter: QAPromptAdapter,
        sample_cases: list[Case[str, str, dict[str, str]]],
    ) -> None:
        mock_qa = AsyncMock()
        mock_qa.answer = AsyncMock(return_value=("An answer.", []))

        with (
            patch("evaluations.optimization.HaikuRAG") as mock_haiku_cls,
            patch("evaluations.optimization.get_qa_agent", return_value=mock_qa),
        ):
            mock_haiku = AsyncMock()
            mock_haiku_cls.return_value.__aenter__ = AsyncMock(return_value=mock_haiku)
            mock_haiku_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            adapter._judge = AsyncMock(return_value=(0.9, "Almost perfect"))  # type: ignore[method-assign]

            result = await adapter._evaluate_async(
                sample_cases, "test prompt", capture_traces=True
            )

        assert result.trajectories is not None
        assert len(result.trajectories) == 2

        traj = result.trajectories[0]
        assert traj.question == "What is X?"
        assert traj.expected_answer == "X is a thing."
        assert traj.actual_answer == "An answer."
        assert traj.score == 0.9
        assert traj.judge_reason == "Almost perfect"

    @pytest.mark.asyncio
    async def test_handles_qa_failure(
        self,
        adapter: QAPromptAdapter,
        sample_cases: list[Case[str, str, dict[str, str]]],
    ) -> None:
        mock_qa = AsyncMock()
        mock_qa.answer = AsyncMock(side_effect=RuntimeError("LLM down"))

        with (
            patch("evaluations.optimization.HaikuRAG") as mock_haiku_cls,
            patch("evaluations.optimization.get_qa_agent", return_value=mock_qa),
        ):
            mock_haiku = AsyncMock()
            mock_haiku_cls.return_value.__aenter__ = AsyncMock(return_value=mock_haiku)
            mock_haiku_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await adapter._evaluate_async(
                sample_cases, "test prompt", capture_traces=True
            )

        assert all(o is None for o in result.outputs)
        assert all(s == 0.0 for s in result.scores)
        assert result.trajectories is not None
        assert all(t.actual_answer is None for t in result.trajectories)
        assert all(
            t.judge_reason == "QA agent failed to produce an answer"
            for t in result.trajectories
        )


class TestReflectionLM:
    def test_handles_string_prompt(self) -> None:
        mock_result = MagicMock()
        mock_result.output = "Reflected response"

        with (
            patch("evaluations.optimization.get_model"),
            patch("pydantic_ai.Agent") as mock_agent_cls,
        ):
            mock_agent = MagicMock()
            mock_agent.run_sync.return_value = mock_result
            mock_agent_cls.return_value = mock_agent

            lm = ReflectionLM(model_config=AppConfig().qa.model, config=AppConfig())
            lm._agent = mock_agent

            result = lm("test prompt")

        assert result == "Reflected response"
        mock_agent.run_sync.assert_called_once_with("test prompt")

    def test_handles_chat_messages(self) -> None:
        mock_result = MagicMock()
        mock_result.output = "Chat response"

        with (
            patch("evaluations.optimization.get_model"),
            patch("pydantic_ai.Agent") as mock_agent_cls,
        ):
            mock_agent = MagicMock()
            mock_agent.run_sync.return_value = mock_result
            mock_agent_cls.return_value = mock_agent

            lm = ReflectionLM(model_config=AppConfig().qa.model, config=AppConfig())
            lm._agent = mock_agent

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
            result = lm(messages)

        assert result == "Chat response"
        call_arg = mock_agent.run_sync.call_args[0][0]
        assert "system: You are helpful." in call_arg
        assert "user: Hello" in call_arg


class TestEvaluateSync:
    def test_delegates_to_evaluate_async(
        self,
        adapter: QAPromptAdapter,
        sample_cases: list[Case[str, str, dict[str, str]]],
    ) -> None:
        expected_batch: EvaluationBatch[EvalTrajectory, str | None] = EvaluationBatch(
            outputs=["answer1", "answer2"],
            scores=[0.9, 0.8],
            trajectories=None,
        )

        with patch.object(
            adapter,
            "_evaluate_async",
            new_callable=AsyncMock,
            return_value=expected_batch,
        ) as mock_eval:
            result = adapter.evaluate(
                sample_cases, {"instructions": "my prompt"}, capture_traces=True
            )

        mock_eval.assert_called_once_with(sample_cases, "my prompt", True)
        assert result is expected_batch


class TestProposalAttribute:
    def test_propose_new_texts_is_none(self, adapter: QAPromptAdapter) -> None:
        assert adapter.propose_new_texts is None


class TestRunOptimization:
    def _make_spec(self, db_path: Path) -> DatasetSpec:
        return DatasetSpec(
            key="test",
            db_filename="test.lancedb",
            document_loader=lambda: None,  # type: ignore[return-value]
            document_mapper=lambda doc: None,
            qa_loader=lambda: None,  # type: ignore[return-value]
            qa_case_builder=lambda idx, doc: None,  # type: ignore[return-value]
            system_prompt="You are a test assistant.",
        )

    def test_returns_results(self, tmp_path: Path) -> None:
        spec = self._make_spec(tmp_path / "test.lancedb")
        cases: list[Case[str, str, dict[str, str]]] = [
            Case(
                name="q1",
                inputs="Q?",
                expected_output="A.",
                metadata={"case_index": "1"},
            ),
        ]

        mock_result = MagicMock()
        mock_result.best_idx = 0
        mock_result.val_aggregate_scores = [0.95]
        mock_result.best_candidate = {"instructions": "optimized prompt"}
        mock_result.total_metric_calls = 10
        mock_result.num_candidates = 3

        with (
            patch("evaluations.optimization.get_model"),
            patch("evaluations.optimization.ReflectionLM"),
            patch("gepa.optimize", return_value=mock_result),
        ):
            result = run_optimization(
                spec=spec,
                config=AppConfig(),
                cases=cases,
                max_calls=10,
                db_path=tmp_path / "test.lancedb",
            )

        assert result["best_score"] == 0.95
        assert result["best_prompt"] == "optimized prompt"
        assert result["total_calls"] == 10
        assert result["num_candidates"] == 3

    def test_saves_output_file(self, tmp_path: Path) -> None:
        spec = self._make_spec(tmp_path / "test.lancedb")
        cases: list[Case[str, str, dict[str, str]]] = [
            Case(
                name="q1",
                inputs="Q?",
                expected_output="A.",
                metadata={"case_index": "1"},
            ),
        ]
        output_path = tmp_path / "prompt.txt"

        mock_result = MagicMock()
        mock_result.best_idx = 0
        mock_result.val_aggregate_scores = [0.85]
        mock_result.best_candidate = {"instructions": "saved prompt"}
        mock_result.total_metric_calls = 5
        mock_result.num_candidates = 2

        with (
            patch("evaluations.optimization.get_model"),
            patch("evaluations.optimization.ReflectionLM"),
            patch("gepa.optimize", return_value=mock_result),
        ):
            run_optimization(
                spec=spec,
                config=AppConfig(),
                cases=cases,
                max_calls=5,
                db_path=tmp_path / "test.lancedb",
                output=output_path,
            )

        assert output_path.read_text() == "saved prompt"

    def test_uses_default_prompt_when_spec_has_none(self, tmp_path: Path) -> None:
        spec = DatasetSpec(
            key="test",
            db_filename="test.lancedb",
            document_loader=lambda: None,  # type: ignore[return-value]
            document_mapper=lambda doc: None,
            qa_loader=lambda: None,  # type: ignore[return-value]
            qa_case_builder=lambda idx, doc: None,  # type: ignore[return-value]
        )
        cases: list[Case[str, str, dict[str, str]]] = [
            Case(
                name="q1",
                inputs="Q?",
                expected_output="A.",
                metadata={"case_index": "1"},
            ),
        ]

        mock_result = MagicMock()
        mock_result.best_idx = 0
        mock_result.val_aggregate_scores = [0.5]
        mock_result.best_candidate = "fallback prompt"
        mock_result.total_metric_calls = 1
        mock_result.num_candidates = 1

        with (
            patch("evaluations.optimization.get_model"),
            patch("evaluations.optimization.ReflectionLM"),
            patch("gepa.optimize", return_value=mock_result) as mock_gepa,
        ):
            result = run_optimization(
                spec=spec,
                config=AppConfig(),
                cases=cases,
                max_calls=1,
                db_path=tmp_path / "test.lancedb",
            )

        # When best_candidate is a string (not dict), it should be used directly
        assert result["best_prompt"] == "fallback prompt"

        # Verify seed_candidate used QA_SYSTEM_PROMPT (not None)
        call_kwargs = mock_gepa.call_args[1]
        seed = call_kwargs["seed_candidate"]
        assert seed["instructions"] is not None
        assert len(seed["instructions"]) > 0
