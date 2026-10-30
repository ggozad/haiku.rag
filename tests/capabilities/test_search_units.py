from typing import Any
from unittest.mock import AsyncMock

import pytest
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
from tests.capabilities.helpers import Deps, make_result, png, stub_client


def burst_model(bursts: list[list[str]]) -> FunctionModel:
    """Emit one `search` call per query in each burst, then answer."""
    responses = 0

    def model_function(_messages, _info) -> ModelResponse:
        nonlocal responses
        responses += 1
        if responses <= len(bursts):
            return ModelResponse(
                parts=[
                    ToolCallPart("search", {"query": query})
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
        capabilities=[create_rag(db_path=db_path, config=config)],
    )


def search_returns(result: AgentRunResult[Any]) -> list[ToolReturnPart]:
    return [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == "search"
    ]


def outcomes(result: AgentRunResult[Any]) -> list[str]:
    return [
        "failed" if part.outcome == "failed" else "ok"
        for part in search_returns(result)
    ]


async def test_a_burst_in_one_response_consumes_one_unit(rag_db):
    """Three searches emitted together cost one unit and run in emission order."""
    agent = burst_agent([["ai", "machine learning", "deep learning"]], rag_db, 1)

    result = await agent.run("question", deps=Deps())

    assert outcomes(result) == ["ok", "ok", "ok"]
    calls = [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_name == "search"
    ]
    assert [part.tool_call_id for part in search_returns(result)] == [
        part.tool_call_id for part in calls
    ]


@pytest.mark.parametrize(
    ("bursts", "max_searches", "expected"),
    [
        pytest.param(
            [["ai"], ["machine learning"]],
            1,
            ["ok", "failed"],
            id="sequential searches pay one unit each",
        ),
        pytest.param(
            [["ai", "machine learning", "deep learning"]],
            0,
            ["failed", "failed", "failed"],
            id="no budget fails every sibling",
        ),
        pytest.param(
            [["ai"], ["ml", "deep learning", "supervised"]],
            1,
            ["ok", "failed", "failed", "failed"],
            id="a rejected round fails all its siblings",
        ),
        pytest.param(
            [["ai", "machine learning", "deep learning", "supervised learning"]],
            2,
            ["ok", "ok", "ok", "ok"],
            id="a fourth sibling fits a second unit",
        ),
        pytest.param(
            [["ai", "machine learning", "deep learning", "supervised learning"]],
            1,
            ["ok", "ok", "ok", "failed"],
            id="a fourth sibling fails without a second unit",
        ),
    ],
)
async def test_search_units_price_each_burst(rag_db, bursts, max_searches, expected):
    result = await burst_agent(bursts, rag_db, max_searches).run(
        "question", deps=Deps()
    )

    assert outcomes(result) == expected
    for part in search_returns(result):
        if part.outcome == "failed":
            assert "Search limit reached" in str(part.content)


async def test_unit_tracking_resets_between_runs(rag_db):
    """A second run's opening burst prices like a first run's."""

    def model_function(messages, _info) -> ModelResponse:
        if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(
            parts=[
                ToolCallPart("search", {"query": query})
                for query in ["ai", "machine learning", "deep learning"]
            ]
        )

    config = AppConfig()
    config.qa.max_searches = 1
    agent = Agent(
        FunctionModel(model_function),
        deps_type=Deps,
        capabilities=[create_rag(db_path=rag_db, config=config)],
    )
    deps = Deps()

    first = await agent.run("question", deps=deps)
    second = await agent.run("another", deps=deps, message_history=first.all_messages())

    assert outcomes(first) == ["ok", "ok", "ok"]
    assert outcomes(second)[-3:] == ["ok", "ok", "ok"]


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


async def test_a_new_run_step_formats_shown_results_in_full(temp_db_path):
    client = stub_client([make_result()], [make_result()])
    capability = dedup_capability(client, temp_db_path)

    await capability._search("q", None, 1)
    second = await capability._search("q again", None, 2)

    assert "body" in text_of(second)
    assert len(images_of(second)) == 1


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


async def test_same_anchor_with_new_evidence_formats_in_full(temp_db_path):
    shared, extra = png(), png()
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
        ({"image_data": {"#/pictures/9": png()}}, False),
    ],
)
async def test_equivalence_follows_the_rendered_evidence(
    temp_db_path, overrides: dict[str, Any], elided: bool
):
    """Any rendered field or picture identity defeats elision; score alone does not."""
    client = stub_client([make_result()], [make_result(**overrides)])
    capability = dedup_capability(client, temp_db_path)

    await capability._search("q", None, 1)
    second = await capability._search("q rephrased", None, 1)

    assert ("Also matched, shown above" in text_of(second)) is elided


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


async def test_the_same_chunk_from_differently_narrowed_siblings_is_elided(
    temp_db_path,
):
    """The collection label follows the run, so a broad sibling recognises what
    a narrowed one showed."""
    client = stub_client(
        [make_result(source="alpha")],
        [make_result(source="alpha")],
        sources=["alpha", "beta"],
    )
    capability = dedup_capability(client, temp_db_path)

    first = await capability._search("q", None, 1, sources=["alpha"])
    second = await capability._search("q rephrased", None, 1)

    assert "Collection: alpha" in text_of(first)
    assert "Also matched, shown above: [c1] [rank 1 of 1]" in text_of(second)


async def test_a_narrowed_search_labels_its_pictures_with_the_collection(
    temp_db_path,
):
    client = stub_client([make_result(source="alpha")], sources=["alpha", "beta"])
    capability = dedup_capability(client, temp_db_path)

    first = await capability._search("q", None, 1, sources=["alpha"])

    labels = [item for item in first.content if isinstance(item, str)]
    assert any("Collection: alpha" in label for label in labels)
