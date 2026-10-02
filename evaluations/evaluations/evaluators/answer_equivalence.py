import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from typesafe_sdk import AsyncTypeSafeClient, NoulCriteria

from evaluations.config import ConversationInput
from evaluations.evaluators.chat_decision import ChatDecisionEndpoint
from evaluations.evaluators.system_one import (
    DecisionEndpoint,
    DecisionError,
    Judgement,
    StateField,
    SystemOneEndpoint,
    SystemOneJudge,
)
from haiku.rag.config.models import AppConfig, ModelConfig, SystemOneConfig
from haiku.rag.utils.models import get_model

ANSWER_EQUIVALENCE_INSTRUCTIONS = {
    "judge": "Answer equivalence for a retrieval question-answering benchmark.",
    "question": "`generated_answer` is equivalent to `expected_answer` for `question`.",
    "asymmetry": "Judge the generated answer against the expected answer, not the reverse.",
}

ANSWER_EQUIVALENCE_CRITERIA = NoulCriteria(
    true=[
        "It carries the facts the question asked for, in any wording, notation, "
        "rounding or level of precision.",
        "It is broader, longer or more detailed than the expected answer and "
        "consistent with it.",
        "It omits parts of the expected answer that the question did not ask for.",
        "It reaches the same conclusion from different but compatible specifics.",
    ],
    false=[
        "It contradicts a fact in the expected answer, including a different "
        "figure for the same quantity.",
        "It omits something the expected answer treats as central to the "
        "question asked.",
        "It is about the right subject without answering what was asked.",
        "It asserts where the expected answer declines, or declines where the "
        "expected answer asserts.",
    ],
)

# The chat format takes one short question; the criteria describe its two options.
ANSWER_EQUIVALENCE_QUESTION = (
    "Is `generated_answer` equivalent to `expected_answer` for `question`?"
)

_STATE_NAMES: Mapping[StateField, str] = {
    "input": "question",
    "expected_output": "expected_answer",
    "output": "generated_answer",
}


@dataclass(repr=False)
class AnswerEquivalenceJudge(SystemOneJudge):
    """SystemOneJudge set up for answer equivalence; conversations go to the fallback."""

    instructions: Any = field(default_factory=lambda: ANSWER_EQUIVALENCE_INSTRUCTIONS)
    criteria: NoulCriteria | None = field(
        default_factory=lambda: ANSWER_EQUIVALENCE_CRITERIA
    )
    include_input: bool = True
    include_expected_output: bool = True
    state_names: Mapping[StateField, str] = field(default_factory=lambda: _STATE_NAMES)
    evaluation_name: str = "answer_equivalent"

    async def evaluate(self, ctx: EvaluatorContext) -> Judgement:
        if isinstance(ctx.inputs, ConversationInput):
            return {
                self.evaluation_name: await self._fallback_verdict(ctx),
                f"{self.evaluation_name}_decided_by": "fallback_conversation",
            }
        return await super().evaluate(ctx)


def system_one_endpoint(
    config: SystemOneConfig, app_config: AppConfig
) -> DecisionEndpoint:
    """The configured decision endpoint; `/v1/systemone` reads its key from TYPESAFE_API_KEY."""
    if config.provider is not None:
        assert config.model is not None
        model_config = ModelConfig(
            provider=config.provider,
            name=config.model,
            base_url=config.base_url,
            thinking=False,
            temperature=0.0,
            max_tokens=1,
        )
        return ChatDecisionEndpoint(model=get_model(model_config, app_config))
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if api_key is None and config.base_url is not None:
        # A local server takes no key, but the client refuses to start without one.
        api_key = "none"
    client = AsyncTypeSafeClient(api_key=api_key, base_url=config.base_url)
    return SystemOneEndpoint(client=client, model=config.model)


def answer_equivalence_judge(
    config: SystemOneConfig, endpoint: DecisionEndpoint, fallback: Evaluator
) -> AnswerEquivalenceJudge:
    """The judge for the configured endpoint, asking in the format it reads."""
    if config.provider is not None:
        return AnswerEquivalenceJudge(
            endpoint=endpoint,
            instructions=ANSWER_EQUIVALENCE_QUESTION,
            fallback=fallback,
        )
    return AnswerEquivalenceJudge(endpoint=endpoint, fallback=fallback)


async def check_system_one(judge: SystemOneJudge, config: SystemOneConfig) -> str:
    """The model the endpoint serves; raises unless it answers the judge's question."""
    try:
        _, served_model = await judge.endpoint.noul(
            {"output": "yes"}, judge.evaluation_name, judge.question()
        )
    except DecisionError as exc:
        endpoint = config.base_url or (
            f"the default {config.provider} server"
            if config.provider
            else "TypeSafe's hosted API"
        )
        raise RuntimeError(
            f"evaluations.system_one: {endpoint} did not answer: {exc}"
        ) from exc
    return served_model
