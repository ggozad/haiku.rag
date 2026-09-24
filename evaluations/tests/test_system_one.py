import json
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import httpx2
import pytest
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeError

from evaluations.evaluators import SystemOneJudge


def _ctx(inputs="What is 2 + 2?", expected="4", output="four") -> EvaluatorContext:
    return EvaluatorContext(
        name="case",
        inputs=inputs,
        metadata=None,
        expected_output=expected,
        output=output,
        duration=0.0,
        _span_tree=MagicMock(),
        attributes={},
        metrics={},
    )


@dataclass
class Endpoint:
    """A /v1/systemone server answering every question with a fixed probability."""

    p: float | None = 0.9
    status: int = 200
    requests: list[dict] = field(default_factory=list)

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if self.status != 200:
            return httpx2.Response(self.status, json={"error": {"message": "boom"}})
        answers = {name: {"type": "noul", "noul": self.p} for name in body["questions"]}
        return httpx2.Response(
            200,
            json={
                "model": body.get("model") or "server-default",
                "usage": {"input_tokens": 10, "output_tokens": 0},
                "answers": answers,
            },
        )

    def client(self) -> AsyncTypeSafeClient:
        return AsyncTypeSafeClient(
            api_key="test",
            base_url="http://decider.test",
            transport=httpx2.MockTransport(self),
            retry=RetryPolicy(max_retries=0),
        )


@dataclass
class FixedJudge(Evaluator):
    """A fallback judge with a fixed verdict that counts its calls."""

    verdict: bool = True
    calls: int = 0

    async def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        self.calls += 1
        return EvaluationReason(value=self.verdict, reason="fallback reason")


def _verdict(result: dict) -> EvaluationReason:
    verdict = result["verdict"]
    assert isinstance(verdict, EvaluationReason)
    return verdict


def _judge(endpoint: Endpoint, **kwargs) -> SystemOneJudge:
    return SystemOneJudge(
        client=endpoint.client(), instructions="Is the output correct?", **kwargs
    )


class TestSystemOneJudgeBands:
    @pytest.mark.parametrize("p", [0.8, 0.95])
    async def test_confident_pass(self, p: float) -> None:
        fallback = FixedJudge(verdict=False)
        result = await _judge(Endpoint(p=p), fallback=fallback).evaluate(_ctx())

        assert _verdict(result).value is True
        assert result["verdict_probability"] == p
        assert result["verdict_decided_by"] == "system_one"
        assert fallback.calls == 0

    @pytest.mark.parametrize("p", [0.0, 0.1999])
    async def test_confident_fail(self, p: float) -> None:
        fallback = FixedJudge(verdict=True)
        result = await _judge(Endpoint(p=p), fallback=fallback).evaluate(_ctx())

        assert _verdict(result).value is False
        assert result["verdict_decided_by"] == "system_one"
        assert fallback.calls == 0

    @pytest.mark.parametrize("p", [0.2, 0.5, 0.7999])
    async def test_uncertain_band_defers_to_fallback(self, p: float) -> None:
        fallback = FixedJudge(verdict=False)
        result = await _judge(Endpoint(p=p), fallback=fallback).evaluate(_ctx())

        assert _verdict(result) == EvaluationReason(
            value=False, reason="fallback reason"
        )
        assert result["verdict_probability"] == p
        assert result["verdict_decided_by"] == "fallback"
        assert fallback.calls == 1

    async def test_bands_are_configurable(self) -> None:
        fallback = FixedJudge(verdict=False)
        judge = _judge(Endpoint(p=0.7), pass_at=0.6, fail_below=0.1, fallback=fallback)
        result = await judge.evaluate(_ctx())

        assert _verdict(result).value is True
        assert fallback.calls == 0

    @pytest.mark.parametrize("p, verdict", [(0.5, True), (0.4999, False)])
    async def test_without_fallback_the_band_splits_at_one_half(
        self, p: float, verdict: bool
    ) -> None:
        result = await _judge(Endpoint(p=p)).evaluate(_ctx())

        assert _verdict(result).value is verdict
        assert result["verdict_decided_by"] == "system_one"

    async def test_reason_names_the_probability(self) -> None:
        result = await _judge(Endpoint(p=0.93)).evaluate(_ctx())

        assert _verdict(result).reason == "system_one p=0.930"


class TestSystemOneJudgeErrors:
    async def test_endpoint_error_falls_back(self) -> None:
        fallback = FixedJudge(verdict=True)
        result = await _judge(Endpoint(status=413), fallback=fallback).evaluate(_ctx())

        assert _verdict(result).value is True
        assert result["verdict_decided_by"] == "fallback_on_error"
        assert "verdict_probability" not in result
        assert fallback.calls == 1

    async def test_endpoint_error_without_fallback_raises(self) -> None:
        with pytest.raises(TypeSafeError):
            await _judge(Endpoint(status=500)).evaluate(_ctx())

    async def test_fallback_mapping_output_is_read_by_evaluation_name(self) -> None:
        @dataclass
        class NamedJudge(Evaluator):
            async def evaluate(self, ctx: EvaluatorContext):
                return {"verdict": EvaluationReason(value=True, reason="named")}

        judge = _judge(Endpoint(p=0.5), fallback=NamedJudge())
        result = await judge.evaluate(_ctx())

        assert _verdict(result) == EvaluationReason(value=True, reason="named")

    async def test_fallback_scalar_output_becomes_the_verdict(self) -> None:
        @dataclass
        class BareJudge(Evaluator):
            async def evaluate(self, ctx: EvaluatorContext) -> bool:
                return False

        judge = _judge(Endpoint(p=0.5), fallback=BareJudge())
        result = await judge.evaluate(_ctx())

        assert _verdict(result) == EvaluationReason(value=False)


class TestSystemOneJudgeRequest:
    async def test_state_carries_only_the_output_by_default(self) -> None:
        endpoint = Endpoint()
        await _judge(endpoint).evaluate(_ctx())

        assert endpoint.requests[0]["state"] == {"output": "four"}

    async def test_state_includes_input_and_expected_output_under_their_names(
        self,
    ) -> None:
        endpoint = Endpoint()
        judge = _judge(
            endpoint,
            include_input=True,
            include_expected_output=True,
            state_names={
                "input": "question",
                "expected_output": "expected_answer",
                "output": "generated_answer",
            },
        )
        await judge.evaluate(_ctx())

        assert endpoint.requests[0]["state"] == {
            "question": "What is 2 + 2?",
            "expected_answer": "4",
            "generated_answer": "four",
        }

    async def test_rubric_goes_in_the_question_not_the_state(self) -> None:
        endpoint = Endpoint()
        judge = _judge(
            endpoint,
            criteria={"true": ["It is correct."], "false": ["It is wrong."]},
            model="decider-4b-v1",
            evaluation_name="answer_equivalent",
        )
        await judge.evaluate(_ctx())

        request = endpoint.requests[0]
        assert request["model"] == "decider-4b-v1"
        assert request["questions"] == {
            "answer_equivalent": {
                "type": "noul",
                "instructions": "Is the output correct?",
                "criteria": {"true": ["It is correct."], "false": ["It is wrong."]},
            }
        }
        assert "Is the output correct?" not in json.dumps(request["state"])
