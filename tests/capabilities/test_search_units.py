import base64
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock

import pytest
from PIL import Image as PILImage
from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.run import AgentRunResult

from haiku.rag.capabilities.ledger import CapabilityEvidenceRecord
from haiku.rag.capabilities.rag import RAGState
from haiku.rag.capabilities.rag import create_capability as create_rag
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models.chunk import SearchResult


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


def _png() -> str:
    buffer = BytesIO()
    PILImage.new("RGB", (4, 4), "red").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def make_result(**overrides: Any) -> SearchResult:
    fields: dict[str, Any] = {
        "content": "body",
        "score": 0.9,
        "source": "main",
        "chunk_id": "c1",
        "document_id": "d1",
        "image_data": {"#/pictures/0": _png()},
    }
    fields.update(overrides)
    return SearchResult(**fields)


def stub_client(
    *batches: list[SearchResult], sources: list[str] | None = None
) -> AsyncMock:
    client = AsyncMock()
    client.search.side_effect = list(batches)
    client.expand_context.side_effect = lambda results: results
    client.source_names = sources or ["main"]
    return client


def dedup_capability(client: AsyncMock, temp_db_path, *, vision: bool = True):
    capability = create_rag(db_path=temp_db_path, config=AppConfig(), vision=vision)
    capability.state = RAGState(evidence=CapabilityEvidenceRecord(question=0))
    capability.borrowed_rag = client
    return capability


def images_of(returned: Any) -> list[BinaryContent]:
    if isinstance(returned, str):
        return []
    return [item for item in returned.content if isinstance(item, BinaryContent)]


def text_of(returned: Any) -> str:
    return returned if isinstance(returned, str) else returned.return_value


@pytest.mark.asyncio
async def test_a_duplicate_sibling_is_elided_and_stays_citable(temp_db_path):
    duplicate, novel = make_result(), make_result(chunk_id="c2", content="novel")
    client = stub_client([make_result()], [duplicate, novel])
    capability = dedup_capability(client, temp_db_path)

    first = await capability._search("q", None, 1)
    second = await capability._search("q rephrased", None, 1)

    assert len(images_of(first)) == 1
    assert images_of(second) == []
    text = text_of(second)
    assert "Also matched, shown above: [c1] [rank 1 of 2]" in text
    assert "body" not in text
    assert "[rank 2 of 2]" in text and "novel" in text
    assert [r.chunk_id for r in capability.state.searches["q rephrased"]] == [
        "c1",
        "c2",
    ]
    assert await capability._cite(["c1"]) == "Registered 1 citation(s)."


@pytest.mark.asyncio
async def test_a_new_run_step_formats_shown_results_in_full(temp_db_path):
    client = stub_client([make_result()], [make_result()])
    capability = dedup_capability(client, temp_db_path)

    await capability._search("q", None, 1)
    second = await capability._search("q again", None, 2)

    assert "body" in text_of(second)
    assert len(images_of(second)) == 1


@pytest.mark.asyncio
async def test_same_chunk_id_from_another_collection_is_not_elided(temp_db_path):
    client = stub_client(
        [make_result(source="alpha")],
        [make_result(source="beta")],
        sources=["alpha", "beta"],
    )
    capability = dedup_capability(client, temp_db_path)

    await capability._search("q", None, 1)
    second = await capability._search("q rephrased", None, 1)

    assert "body" in text_of(second)


@pytest.mark.asyncio
async def test_same_anchor_with_new_evidence_formats_in_full(temp_db_path):
    shared, extra = _png(), _png()
    client = stub_client(
        [make_result(content="c1 with c2", image_data={"#/pictures/1": shared})],
        [
            make_result(
                content="c1 with c3",
                image_data={"#/pictures/1": shared, "#/pictures/3": extra},
            )
        ],
    )
    capability = dedup_capability(client, temp_db_path)

    await capability._search("q", None, 1)
    second = await capability._search("q rephrased", None, 1)

    assert "c1 with c3" in text_of(second)
    assert len(images_of(second)) == 1
    labels = [item for item in second.content if isinstance(item, str)]
    assert any("#/pictures/3" in label for label in labels)


@pytest.mark.parametrize(
    ("overrides", "elided"),
    [
        ({"score": 0.1}, True),
        ({"content": "different"}, False),
        ({"document_title": "Other"}, False),
        ({"headings": ["Heading"]}, False),
        ({"labels": ["table"]}, False),
        ({"picture_captions": {"#/pictures/0": "A caption"}}, False),
        ({"image_data": {"#/pictures/9": _png()}}, False),
    ],
)
@pytest.mark.asyncio
async def test_equivalence_follows_the_rendered_evidence(
    temp_db_path, overrides: dict[str, Any], elided: bool
):
    """Any rendered field or picture identity defeats elision; score alone does not."""
    client = stub_client([make_result()], [make_result(**overrides)])
    capability = dedup_capability(client, temp_db_path)

    await capability._search("q", None, 1)
    second = await capability._search("q rephrased", None, 1)

    assert ("Also matched, shown above" in text_of(second)) is elided


@pytest.mark.asyncio
async def test_a_failed_sibling_commits_nothing(temp_db_path):
    client = stub_client(
        [make_result(image_data={"#/pictures/0": "AAA"})],
        [make_result()],
    )
    capability = dedup_capability(client, temp_db_path)
    evidence_before = capability.state.evidence.model_dump()

    with pytest.raises(Exception):
        await capability._search("q", None, 1)

    assert capability.state.searches == {}
    assert capability.state.evidence.model_dump() == evidence_before

    second = await capability._search("q rephrased", None, 1)

    assert "body" in text_of(second)
    assert len(images_of(second)) == 1
