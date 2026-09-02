from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.run import AgentRunResult

from haiku.rag.capabilities.rag import create_capability as create_rag
from haiku.rag.config.models import AppConfig


@dataclass
class Deps:
    state: dict[str, Any] = field(default_factory=dict)


def burst_model(bursts: list[list[str]]) -> FunctionModel:
    """Emit one `rag_search` call per query in each burst, then answer."""
    responses = 0

    def model_function(_messages, _info) -> ModelResponse:
        nonlocal responses
        responses += 1
        if responses <= len(bursts):
            return ModelResponse(
                parts=[
                    ToolCallPart("rag_search", {"query": query})
                    for query in bursts[responses - 1]
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    return FunctionModel(model_function)


def burst_agent(
    bursts: list[list[str]], db_path, max_searches: int
) -> Agent[Deps, str]:
    config = AppConfig()
    config.qa.max_searches = max_searches
    return Agent(
        burst_model(bursts),
        deps_type=Deps,
        capabilities=[create_rag(db_path=db_path, config=config, defer_loading=False)],
    )


def search_returns(result: AgentRunResult[Any]) -> list[ToolReturnPart]:
    return [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == "rag_search"
    ]


def outcomes(result: AgentRunResult[Any]) -> list[str]:
    return [
        "failed" if part.outcome == "failed" else "ok"
        for part in search_returns(result)
    ]


@pytest.mark.asyncio
async def test_a_burst_in_one_response_consumes_one_unit(rag_db):
    """Three searches emitted together cost one unit and run in emission order."""
    agent = burst_agent([["ai", "machine learning", "deep learning"]], rag_db, 1)

    result = await agent.run("question", deps=Deps())

    assert outcomes(result) == ["ok", "ok", "ok"]
    calls = [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_name == "rag_search"
    ]
    assert [part.tool_call_id for part in search_returns(result)] == [
        part.tool_call_id for part in calls
    ]


@pytest.mark.asyncio
async def test_sequential_searches_pay_one_unit_each(rag_db):
    agent = burst_agent([["ai"], ["machine learning"]], rag_db, 1)

    result = await agent.run("question", deps=Deps())

    assert outcomes(result) == ["ok", "failed"]
    assert "Search limit reached" in str(search_returns(result)[1].content)


@pytest.mark.asyncio
async def test_max_searches_zero_fails_every_sibling(rag_db):
    agent = burst_agent([["ai", "machine learning", "deep learning"]], rag_db, 0)

    result = await agent.run("question", deps=Deps())

    assert outcomes(result) == ["failed", "failed", "failed"]


@pytest.mark.asyncio
async def test_a_rejected_round_fails_all_its_siblings(rag_db):
    agent = burst_agent([["ai"], ["ml", "deep learning", "supervised"]], rag_db, 1)

    result = await agent.run("question", deps=Deps())

    assert outcomes(result) == ["ok", "failed", "failed", "failed"]


@pytest.mark.asyncio
async def test_a_sibling_past_the_allowance_pays_its_own_unit(rag_db):
    burst = [["ai", "machine learning", "deep learning", "supervised learning"]]

    within = await burst_agent(burst, rag_db, 2).run("question", deps=Deps())
    over = await burst_agent(burst, rag_db, 1).run("question", deps=Deps())

    assert outcomes(within) == ["ok", "ok", "ok", "ok"]
    assert outcomes(over) == ["ok", "ok", "ok", "failed"]


@pytest.mark.asyncio
async def test_unit_tracking_resets_between_runs(rag_db):
    """A second run's opening burst prices like a first run's."""

    def model_function(messages, _info) -> ModelResponse:
        if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(
            parts=[
                ToolCallPart("rag_search", {"query": query})
                for query in ["ai", "machine learning", "deep learning"]
            ]
        )

    config = AppConfig()
    config.qa.max_searches = 1
    agent = Agent(
        FunctionModel(model_function),
        deps_type=Deps,
        capabilities=[create_rag(db_path=rag_db, config=config, defer_loading=False)],
    )
    deps = Deps()

    first = await agent.run("question", deps=deps)
    second = await agent.run("another", deps=deps, message_history=first.all_messages())

    assert outcomes(first) == ["ok", "ok", "ok"]
    assert outcomes(second)[-3:] == ["ok", "ok", "ok"]
