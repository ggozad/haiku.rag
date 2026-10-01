import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from typesafe_sdk import Noul, NoulCriteria

from evaluations.capability_runner import CapabilityRunResult
from evaluations.evaluators.answer_equivalence import (
    ANSWER_EQUIVALENCE_QUESTION,
    answer_equivalence_judge,
    system_one_endpoint,
)
from evaluations.evaluators.chat_decision import (
    DECISION_SYSTEM_PROMPT,
    ChatDecisionEndpoint,
)
from evaluations.evaluators.system_one import DecisionError
from haiku.rag.config.models import AppConfig, SystemOneConfig
from tests.test_system_one import FixedJudge, _ctx


@dataclass
class DecisionModel:
    """A chat decision model: p(yes) by the position of the yes option, as logprobs."""

    p_yes_first: float = 0.9
    p_yes_second: float = 0.9
    letters: tuple[str, str] = ("A", "B")
    logprobs: bool = True
    answer: str | None = None
    tasks: list[dict] = field(default_factory=list)
    system_prompts: list[str] = field(default_factory=list)

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        system, user = (str(part.content) for part in messages[0].parts)  # ty: ignore[unresolved-attribute]
        self.system_prompts.append(system)
        task = json.loads(user)
        self.tasks.append(task)
        yes_first = task["options"][0]["key"] == "yes"
        p = self.p_yes_first if yes_first else self.p_yes_second
        yes_token, no_token = self.letters if yes_first else self.letters[::-1]
        top = [
            {"token": yes_token, "logprob": math.log(p)},
            {"token": no_token, "logprob": math.log(1 - p)},
        ]
        details = (
            {
                "logprobs": [
                    {"token": yes_token, "logprob": math.log(p), "top_logprobs": top}
                ]
            }
            if self.logprobs
            else {}
        )
        text = self.answer if self.answer is not None else yes_token
        return ModelResponse(parts=[TextPart(text)], provider_details=details)

    def endpoint(self) -> ChatDecisionEndpoint:
        return ChatDecisionEndpoint(model=FunctionModel(self, model_name="tev1-4b"))


_QUESTION = Noul(
    instructions="Is the output correct?",
    criteria=NoulCriteria(true=["It is right.", "It says so."], false="It is wrong."),
)


class TestChatDecisionEndpoint:
    async def test_the_two_option_orders_are_averaged(self) -> None:
        model = DecisionModel(p_yes_first=0.7, p_yes_second=0.9)
        p, served = await model.endpoint().noul(
            {"output": "four"}, "verdict", _QUESTION
        )

        assert p == pytest.approx(0.8)
        assert served == "tev1-4b"

    async def test_the_task_is_the_decision_format_in_both_orders(self) -> None:
        model = DecisionModel()
        await model.endpoint().noul({"output": "four"}, "verdict", _QUESTION)

        assert model.system_prompts == [DECISION_SYSTEM_PROMPT] * 2
        assert model.tasks[0] == {
            "state": {"output": "four"},
            "question": "Is the output correct?",
            "options": [
                {"label": "A", "key": "no", "description": "It is wrong."},
                {"label": "B", "key": "yes", "description": "It is right. It says so."},
            ],
        }
        assert [o["key"] for o in model.tasks[1]["options"]] == ["yes", "no"]

    async def test_without_criteria_the_options_read_yes_and_no(self) -> None:
        model = DecisionModel()
        await model.endpoint().noul(
            {"output": "four"}, "verdict", Noul(instructions="Is it right?")
        )

        descriptions = {o["key"]: o["description"] for o in model.tasks[0]["options"]}
        assert descriptions == {"no": "No.", "yes": "Yes."}

    @pytest.mark.parametrize(
        "model, message",
        [
            (DecisionModel(letters=("Yes", "No"), answer="A"), "neither option letter"),
            (DecisionModel(answer="C"), "answered 'C'"),
            (DecisionModel(logprobs=False), "no logprobs"),
        ],
    )
    async def test_an_unreadable_answer_is_a_decision_error(
        self, model: DecisionModel, message: str
    ) -> None:
        with pytest.raises(DecisionError, match=message):
            await model.endpoint().noul({"output": "four"}, "verdict", _QUESTION)

    async def test_an_endpoint_failure_is_a_decision_error(self) -> None:
        def failing(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise ModelHTTPError(status_code=400, model_name="tev1-4b", body="too long")

        endpoint = ChatDecisionEndpoint(model=FunctionModel(failing))
        with pytest.raises(DecisionError):
            await endpoint.noul({"output": "four"}, "verdict", _QUESTION)


class TestChatProvider:
    @pytest.mark.parametrize("provider", ["vllm", "ollama", "openai"])
    def test_the_model_answers_one_token_without_thinking(self, provider: str) -> None:
        config = SystemOneConfig(
            provider=provider,  # ty: ignore[invalid-argument-type]
            base_url="http://localhost:8000",
            model="tev1-4b",
        )
        endpoint = system_one_endpoint(config, AppConfig())

        assert isinstance(endpoint, ChatDecisionEndpoint)
        assert isinstance(endpoint.model, OpenAIChatModel)
        assert endpoint.model.model_name == "tev1-4b"
        settings = endpoint.model.settings or {}
        assert settings.get("temperature") == 0.0
        assert settings.get("max_tokens") == 1
        assert settings.get("openai_reasoning_effort") == "none"

    async def test_the_judge_asks_the_short_question(self) -> None:
        model = DecisionModel(p_yes_first=0.95, p_yes_second=0.95)
        config = SystemOneConfig(provider="vllm", model="tev1-4b")
        judge = answer_equivalence_judge(config, model.endpoint(), FixedJudge())

        result = await judge.evaluate(_ctx())

        assert model.tasks[0]["question"] == ANSWER_EQUIVALENCE_QUESTION
        assert model.tasks[0]["state"] == {
            "question": "What is 2 + 2?",
            "expected_answer": "4",
            "generated_answer": "four",
        }
        assert result["answer_equivalent_decided_by"] == "system_one"
        assert result["answer_equivalent_probability"] == pytest.approx(0.95)
        assert result["answer_equivalent_model"] == "tev1-4b"


class TestRunWithChatProvider:
    async def test_a_gated_run_through_a_chat_provider(self, tmp_path: Path) -> None:
        from evaluations.qa import EvalDataset, run_qa_benchmark
        from tests.test_answer_equivalence import _spec

        model = DecisionModel(p_yes_first=0.95, p_yes_second=0.95)
        endpoint = model.endpoint()
        config = AppConfig.model_validate(
            {"evaluations": {"system_one": {"provider": "vllm", "model": "tev1-4b"}}}
        )
        evaluate = EvalDataset.evaluate
        with (
            patch("evaluations.qa.system_one_endpoint", return_value=endpoint),
            patch.object(endpoint, "aclose", wraps=endpoint.aclose) as aclose,
            patch("evaluations.qa.get_model", return_value="fake-model"),
            patch(
                "evaluations.qa.run_capability_question",
                new_callable=AsyncMock,
                return_value=CapabilityRunResult(answer="answer"),
            ),
            patch.object(
                EvalDataset, "evaluate", autospec=True, side_effect=evaluate
            ) as run,
        ):
            await run_qa_benchmark(
                _spec(), config, db_path=tmp_path / "test.lancedb", results_dir=tmp_path
            )

        (results,) = tmp_path.glob("*.jsonl")
        row = json.loads(results.read_text())
        assert row["judge_decided_by"] == "system_one"
        assert row["system_one_model"] == "tev1-4b"
        assert run.call_args.kwargs["metadata"]["system_one_provider"] == "vllm"
        # The readiness check asks the judge's own question, in both option orders.
        assert [task["question"] for task in model.tasks[:2]] == [
            ANSWER_EQUIVALENCE_QUESTION
        ] * 2
        aclose.assert_awaited_once()
