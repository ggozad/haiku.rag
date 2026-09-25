import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_evals import Case
from pydantic_evals.evaluators import EvaluationReason
from typesafe_sdk import TypeSafeError

from evaluations.capability_runner import CapabilityRunResult
from evaluations.config import ConversationInput, DatasetSpec, Turn
from evaluations.evaluators.answer_equivalence import (
    AnswerEquivalenceJudge,
    check_system_one,
    system_one_client,
)
from evaluations.qa import run_qa_benchmark
from haiku.rag.config.models import AppConfig, SystemOneConfig
from tests.test_system_one import Endpoint, FixedJudge, _ctx


class TestAnswerEquivalenceJudge:
    async def test_request_carries_question_gold_and_answer_as_state(self) -> None:
        endpoint = Endpoint(p=0.9)
        judge = AnswerEquivalenceJudge(client=endpoint.client(), fallback=FixedJudge())
        result = await judge.evaluate(_ctx())

        request = endpoint.requests[0]
        assert request["state"] == {
            "question": "What is 2 + 2?",
            "expected_answer": "4",
            "generated_answer": "four",
        }
        question = request["questions"]["answer_equivalent"]
        assert set(question["instructions"]) == {"judge", "question", "asymmetry"}
        assert len(question["criteria"]["true"]) == 4
        assert len(question["criteria"]["false"]) == 4
        assert result["answer_equivalent_decided_by"] == "system_one"

    async def test_conversation_inputs_go_to_the_fallback(self) -> None:
        endpoint = Endpoint(p=0.9)
        fallback = FixedJudge(verdict=False)
        judge = AnswerEquivalenceJudge(client=endpoint.client(), fallback=fallback)
        conversation = ConversationInput(turns=[Turn(speaker="user", text="q")])
        result = await judge.evaluate(_ctx(inputs=conversation))

        assert result["answer_equivalent"] == EvaluationReason(
            value=False, reason="fallback reason"
        )
        assert result["answer_equivalent_decided_by"] == "fallback_conversation"
        assert endpoint.requests == []


class TestSystemOneClient:
    def test_hosted_api_needs_a_key(self, monkeypatch) -> None:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        with pytest.raises(TypeSafeError, match="API key"):
            system_one_client(SystemOneConfig())

    def test_a_local_server_needs_no_key(self, monkeypatch) -> None:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        client = system_one_client(SystemOneConfig(base_url="http://127.0.0.1:8010"))
        assert client is not None

    async def test_an_unreachable_endpoint_fails_the_check(self) -> None:
        with pytest.raises(RuntimeError, match="did not answer"):
            await check_system_one(
                Endpoint(status=503).client(),
                SystemOneConfig(base_url="http://decider.test"),
            )

    async def test_the_check_returns_the_model_that_served(self) -> None:
        endpoint = Endpoint(p=0.9)
        served = await check_system_one(
            endpoint.client(), SystemOneConfig(model="decider-4b-v1")
        )
        assert endpoint.requests[0]["model"] == "decider-4b-v1"
        assert served == "decider-4b-v1"


def _spec() -> DatasetSpec:
    return DatasetSpec(
        key="test",
        db_filename="test.lancedb",
        document_loader=lambda: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        document_mapper=lambda doc: None,
        qa_loader=lambda: [{"id": "t1"}],  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        qa_case_builder=lambda idx, doc: Case(
            name="c1", inputs="q1", expected_output="ref", metadata={"id": "t1"}
        ),
        pair_key="id",
    )


def _config() -> AppConfig:
    return AppConfig.model_validate(
        {"evaluations": {"system_one": {"base_url": "http://decider.test"}}}
    )


class TestRunWithSystemOne:
    async def test_confident_cases_are_decided_by_the_endpoint(
        self, tmp_path: Path
    ) -> None:
        endpoint = Endpoint(p=0.95)
        with (
            patch("evaluations.qa.system_one_client", return_value=endpoint.client()),
            patch("evaluations.qa.get_model", return_value="fake-model"),
            patch(
                "evaluations.qa.run_capability_question",
                new_callable=AsyncMock,
                return_value=CapabilityRunResult(answer="answer"),
            ),
        ):
            await run_qa_benchmark(
                _spec(),
                _config(),
                db_path=tmp_path / "test.lancedb",
                results_dir=tmp_path / "results",
            )

        (results,) = (tmp_path / "results").glob("*.jsonl")
        row = json.loads(results.read_text())
        assert row["passed"] is True
        assert row["judge_decided_by"] == "system_one"
        assert row["judge_probability"] == 0.95
        assert row["system_one_model"] == "jev-latest"
        # One readiness check plus one judged case.
        assert len(endpoint.requests) == 2

    async def test_an_unreachable_endpoint_stops_the_run_before_any_case(
        self, tmp_path: Path
    ) -> None:
        with (
            patch(
                "evaluations.qa.system_one_client",
                return_value=Endpoint(status=503).client(),
            ),
            patch("evaluations.qa.get_model", return_value="fake-model"),
            patch(
                "evaluations.qa.run_capability_question", new_callable=AsyncMock
            ) as run_question,
            pytest.raises(RuntimeError, match="did not answer"),
        ):
            await run_qa_benchmark(
                _spec(), _config(), db_path=tmp_path / "test.lancedb"
            )

        run_question.assert_not_awaited()

    async def test_a_failed_check_closes_the_client_and_writes_no_file(
        self, tmp_path: Path
    ) -> None:
        client = Endpoint(status=503).client()
        with (
            patch("evaluations.qa.system_one_client", return_value=client),
            patch.object(client, "aclose", wraps=client.aclose) as aclose,
            patch("evaluations.qa.get_model", return_value="fake-model"),
            pytest.raises(RuntimeError, match="did not answer"),
        ):
            await run_qa_benchmark(
                _spec(),
                _config(),
                db_path=tmp_path / "test.lancedb",
                results_dir=tmp_path / "results",
            )

        aclose.assert_awaited_once()
        assert not list(tmp_path.glob("results/*"))

    async def test_the_run_prints_who_decided(self, tmp_path: Path) -> None:
        from rich.console import Console

        console = Console(record=True, width=200)
        with (
            patch("evaluations.qa.console", console),
            patch(
                "evaluations.qa.system_one_client",
                return_value=Endpoint(p=0.95).client(),
            ),
            patch("evaluations.qa.get_model", return_value="fake-model"),
            patch(
                "evaluations.qa.run_capability_question",
                new_callable=AsyncMock,
                return_value=CapabilityRunResult(answer="answer"),
            ),
        ):
            await run_qa_benchmark(
                _spec(), _config(), db_path=tmp_path / "test.lancedb"
            )

        assert "Judge decided by: system_one 1" in console.export_text()

    async def test_metadata_records_the_endpoint_and_bands(
        self, tmp_path: Path
    ) -> None:
        from evaluations.qa import EvalDataset

        evaluate = AsyncMock(side_effect=RuntimeError("stop after metadata"))
        with (
            patch(
                "evaluations.qa.system_one_client",
                return_value=Endpoint(p=0.9).client(),
            ),
            patch("evaluations.qa.get_model", return_value="fake-model"),
            patch.object(EvalDataset, "evaluate", evaluate),
            pytest.raises(RuntimeError, match="stop after metadata"),
        ):
            await run_qa_benchmark(
                _spec(), _config(), db_path=tmp_path / "test.lancedb"
            )

        assert evaluate.await_args is not None
        metadata = evaluate.await_args.kwargs["metadata"]
        assert metadata["system_one_base_url"] == "http://decider.test"
        assert metadata["system_one_model"] is None
        assert metadata["system_one_served_model"] == "jev-latest"
        assert metadata["system_one_pass_at"] == 0.8
        assert metadata["system_one_fail_below"] == 0.2


class TestLiveRunWithSystemOne:
    async def test_a_live_run_says_the_endpoint_is_not_used(
        self, tmp_path: Path
    ) -> None:
        from rich.console import Console

        from evaluations.qa import EvalDataset, run_live_qa_benchmark

        console = Console(record=True, width=200)
        spec = _spec()
        spec.live = True
        with (
            patch("evaluations.qa.console", console),
            patch("evaluations.qa.get_model", return_value="fake-model"),
            patch.object(
                EvalDataset, "evaluate", AsyncMock(side_effect=RuntimeError("stop"))
            ),
            pytest.raises(RuntimeError, match="stop"),
        ):
            await run_live_qa_benchmark(
                spec, _config(), db_path=tmp_path / "test.lancedb"
            )

        assert "evaluations.system_one is not used by live runs" in (
            console.export_text()
        )
