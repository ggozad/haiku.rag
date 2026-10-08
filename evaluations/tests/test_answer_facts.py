import re
from unittest.mock import MagicMock

import pytest
from pydantic_ai import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals.evaluators import EvaluationReason, EvaluatorContext

from evaluations.evaluators import AnswerFactsJudge


def _section(prompt: str, heading: str) -> str:
    match = re.search(rf"## {heading}\n```\n(.*?)\n```", prompt, re.S)
    assert match is not None
    return match.group(1)


def _judge(judged: list[str], reply=None) -> FunctionModel:
    """Says yes to a fact when its upper-case keyword appears in the answer."""

    def grade(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        part = messages[-1].parts[-1]
        assert isinstance(part, UserPromptPart)
        prompt = str(part.content)
        statement, answer = _section(prompt, "Statement"), _section(prompt, "Answer")
        judged.append(statement)
        keywords = re.findall(r"\b[A-Z]{3,}\b", statement)
        supported = any(keyword in answer for keyword in keywords)
        text = reply if reply is not None else ("yes" if supported else "no")
        return ModelResponse(parts=[TextPart(text)])

    return FunctionModel(grade)


def _ctx(output: str, metadata: dict | None) -> EvaluatorContext:
    return EvaluatorContext(
        name="case",
        inputs="question",
        metadata=metadata,
        expected_output="gold",
        output=output,
        duration=0.0,
        _span_tree=MagicMock(),
        attributes={},
        metrics={},
    )


class TestAnswerFactsJudge:
    async def test_completeness_is_the_fraction_of_facts_supported(self) -> None:
        judged: list[str] = []
        judge = AnswerFactsJudge(model=_judge(judged))
        facts = ["Rate is ALPHA.", "Owner is BETA.", "Region is GAMMA.", "DELTA."]

        result = await judge.evaluate(_ctx("ALPHA and GAMMA", {"answer_facts": facts}))

        assert isinstance(result, dict)
        completeness = result["answer_completeness"]
        assert isinstance(completeness, EvaluationReason)
        assert completeness.value == 0.5
        assert completeness.reason is not None
        assert "Owner is BETA." in completeness.reason
        assert "Rate is ALPHA." not in completeness.reason
        assert len(judged) == 4

    async def test_each_fact_is_judged_alone(self) -> None:
        judged: list[str] = []
        facts = ["ALPHA.", "BETA."]

        await AnswerFactsJudge(model=_judge(judged)).evaluate(
            _ctx("ALPHA", {"answer_facts": facts})
        )

        assert sum("ALPHA" in r for r in judged) == 1
        assert sum("BETA" in r for r in judged) == 1
        assert not any("ALPHA" in r and "BETA" in r for r in judged)

    async def test_every_fact_supported_is_complete(self) -> None:
        result = await AnswerFactsJudge(model=_judge([])).evaluate(
            _ctx("ALPHA BETA", {"answer_facts": ["ALPHA.", "BETA."]})
        )

        assert isinstance(result, dict)
        completeness = result["answer_completeness"]
        assert isinstance(completeness, EvaluationReason)
        assert completeness.value == 1.0

    @pytest.mark.parametrize(
        ("reply", "supported"),
        [
            ("Yes, it does.", 1.0),
            ("yes", 1.0),
            ("No\nyes on a later line", 0.0),
            (" ", 0.0),
        ],
    )
    async def test_only_the_first_line_of_the_reply_decides(
        self, reply: str, supported: float
    ) -> None:
        result = await AnswerFactsJudge(model=_judge([], reply=reply)).evaluate(
            _ctx("ALPHA", {"answer_facts": ["ALPHA."]})
        )

        assert isinstance(result, dict)
        completeness = result["answer_completeness"]
        assert isinstance(completeness, EvaluationReason)
        assert completeness.value == supported

    @pytest.mark.parametrize("metadata", [{"answer_facts": []}, {}, None])
    async def test_cases_without_facts_cost_no_judge_call(self, metadata) -> None:
        judged: list[str] = []

        result = await AnswerFactsJudge(model=_judge(judged)).evaluate(
            _ctx("ALPHA", metadata)
        )

        assert result == {}
        assert judged == []

    def test_serializes_the_model_by_id(self) -> None:
        model = _judge([])

        arguments = AnswerFactsJudge(model=model).build_serialization_arguments()

        assert arguments["model"] == model.model_id

    def test_serializes_a_model_name_as_given(self) -> None:
        judge = AnswerFactsJudge(model="ollama:qwen3")

        assert judge.build_serialization_arguments()["model"] == "ollama:qwen3"
