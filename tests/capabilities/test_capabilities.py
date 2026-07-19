from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from haiku.rag.capabilities._base import _compact_old_tool_returns
from haiku.rag.capabilities.analysis import AnalysisCapability, AnalysisState
from haiku.rag.capabilities.analysis import create_capability as create_analysis
from haiku.rag.capabilities.rag import AGENT_PREAMBLE, RAGCapability, RAGState
from haiku.rag.capabilities.rag import create_capability as create_rag
from haiku.rag.config.models import AppConfig, PromptsConfig
from haiku.rag.sandbox import Sandbox, SandboxResult
from haiku.rag.store.models.chunk import Chunk, SearchResult


@dataclass
class Deps:
    state: dict[str, Any] = field(default_factory=dict)


def make_context(deps: Deps) -> RunContext[Deps]:
    return RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        run_id="test-run",
    )


def test_rag_capability_api(temp_db_path):
    capability = create_rag(db_path=temp_db_path, config=AppConfig())

    assert isinstance(capability, RAGCapability)
    assert capability.id == "haiku-rag"
    assert capability.defer_loading is True
    assert set(capability.get_toolset().tools) == {"rag_search", "rag_cite"}
    toolset = capability.get_toolset()
    assert toolset.max_retries == 3
    assert toolset.sequential is True
    assert capability.state_type is RAGState
    assert capability.state_namespace == "rag"
    assert capability.request_limit == 20


def test_analysis_capability_api(temp_db_path):
    capability = create_analysis(db_path=temp_db_path, config=AppConfig())

    assert isinstance(capability, AnalysisCapability)
    assert capability.id == "haiku-rag-analysis"
    assert capability.defer_loading is True
    assert set(capability.get_toolset().tools) == {
        "analysis_search",
        "analysis_execute_code",
        "analysis_cite",
    }
    toolset = capability.get_toolset()
    assert toolset.max_retries == 3
    assert toolset.sequential is True
    assert capability.state_type is AnalysisState
    assert capability.request_limit == 30


def test_capability_factories_resolve_environment_and_defaults(
    temp_db_path, monkeypatch
):
    config = AppConfig()
    monkeypatch.setenv("HAIKU_RAG_DB", str(temp_db_path))
    assert create_rag(config=config).db_path == temp_db_path

    monkeypatch.delenv("HAIKU_RAG_DB")
    assert create_rag(config=config).db_path == (
        config.storage.data_dir / "haiku.rag.lancedb"
    )

    with patch("haiku.rag.config.get_config", return_value=config):
        assert create_rag().config is config
        assert create_analysis().config is config


def test_domain_preamble_is_added_to_capability_instructions(temp_db_path):
    config = AppConfig(
        prompts=PromptsConfig(domain_preamble="The corpus contains solar manuals.")
    )
    capability = create_rag(db_path=temp_db_path, config=config)

    assert capability.get_instructions().startswith(
        "The corpus contains solar manuals.\n\n# RAG"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("factory", "agent_instructions", "heading"),
    [
        (create_rag, AGENT_PREAMBLE, "# RAG"),
        (create_analysis, None, "# Analysis"),
    ],
)
async def test_capability_instructions_are_injected_once(
    temp_db_path, factory, agent_instructions, heading
):
    domain = "The corpus contains solar manuals."
    seen_instructions = []

    def model_function(_messages, info):
        seen_instructions.append(info.instructions or "")
        return ModelResponse(parts=[TextPart("done")])

    config = AppConfig(prompts=PromptsConfig(domain_preamble=domain))
    agent = Agent(
        FunctionModel(model_function),
        deps_type=Deps,
        instructions=agent_instructions,
        capabilities=[
            factory(
                db_path=temp_db_path,
                config=config,
                defer_loading=False,
            )
        ],
    )

    await agent.run("Answer", deps=Deps())

    assert seen_instructions[0].count(domain) == 1
    assert seen_instructions[0].count(heading) == 1


@pytest.mark.asyncio
async def test_request_limit_removes_only_exhausted_capability_tools_per_run(
    temp_db_path,
):
    calls = 0
    seen_tools = []
    seen_instructions = []

    def model_function(_messages, info):
        nonlocal calls
        calls += 1
        seen_tools.append({tool.name for tool in info.function_tools})
        seen_instructions.append(info.instructions or "")
        if calls % 2 == 1:
            return ModelResponse(parts=[ToolCallPart("host_tool", {})])
        return ModelResponse(parts=[TextPart("best available answer")])

    def host_tool(_ctx: RunContext[Deps]) -> str:
        """Return host-owned context."""
        return "host context"

    rag = create_rag(
        db_path=temp_db_path,
        config=AppConfig(),
        defer_loading=False,
    )
    analysis = create_analysis(
        db_path=temp_db_path,
        config=AppConfig(),
        defer_loading=False,
        request_limit=1,
    )
    agent = Agent(
        FunctionModel(model_function),
        deps_type=Deps,
        tools=[host_tool],
        capabilities=[rag, analysis],
    )

    first = await agent.run("Analyze this", deps=Deps())
    second = await agent.run("Analyze another question", deps=Deps())

    assert first.output == "best available answer"
    assert second.output == "best available answer"
    analysis_tools = {
        "analysis_search",
        "analysis_execute_code",
        "analysis_cite",
    }
    for initial, exhausted in ((0, 1), (2, 3)):
        assert analysis_tools <= seen_tools[initial]
        assert analysis_tools.isdisjoint(seen_tools[exhausted])
        assert {"host_tool", "rag_search", "rag_cite"} <= seen_tools[exhausted]
        assert (
            "analysis capability has reached its request limit"
            in (seen_instructions[exhausted])
        )


@pytest.mark.asyncio
async def test_deferred_request_limit_starts_after_capability_load(temp_db_path):
    seen_tools = []
    seen_instructions = []

    def model_function(_messages, info):
        seen_tools.append({tool.name for tool in info.function_tools})
        seen_instructions.append(info.instructions or "")
        if len(seen_tools) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "load_capability",
                        {"id": "haiku-rag-analysis"},
                    )
                ]
            )
        if len(seen_tools) == 2:
            return ModelResponse(parts=[ToolCallPart("host_tool", {})])
        return ModelResponse(parts=[TextPart("best available answer")])

    def host_tool(_ctx: RunContext[Deps]) -> str:
        """Return host-owned context."""
        return "host context"

    agent = Agent(
        FunctionModel(model_function),
        deps_type=Deps,
        tools=[host_tool],
        capabilities=[
            create_analysis(
                db_path=temp_db_path,
                config=AppConfig(),
                request_limit=1,
            )
        ],
    )

    result = await agent.run("Analyze this", deps=Deps())

    assert result.output == "best available answer"
    assert "load_capability" in seen_tools[0]
    assert "analysis_search" in seen_tools[1]
    assert "analysis_search" not in seen_tools[2]
    assert "host_tool" in seen_tools[2]
    assert "analysis capability has reached its request limit" in seen_instructions[2]


@pytest.mark.asyncio
async def test_capability_isolated_per_run_and_round_trips_state(temp_db_path):
    capability = create_rag(db_path=temp_db_path, config=AppConfig())
    deps = Deps(
        state={
            "rag": RAGState(
                document_filter="uri = 'manual.pdf'",
                citations=["old"],
                searches={"old": []},
            ).model_dump(mode="json")
        }
    )

    run_capability = await capability.for_run(make_context(deps))

    assert run_capability is not capability
    assert run_capability.state is not None
    assert run_capability.state.document_filter == "uri = 'manual.pdf'"
    assert run_capability.state.citations == []
    assert run_capability.state.searches == {}
    assert deps.state["rag"]["document_filter"] == "uri = 'manual.pdf'"


@pytest.mark.asyncio
async def test_run_error_closes_resources_and_propagates(temp_db_path):
    capability = create_rag(db_path=temp_db_path, config=AppConfig())
    client = AsyncMock()
    capability.rag = client
    error = RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        await capability.on_run_error(make_context(Deps()), error=error)

    client.__aexit__.assert_awaited_once_with(None, None, None)
    assert capability.rag is None


@pytest.mark.asyncio
async def test_search_and_empty_citation_limits(temp_db_path):
    config = AppConfig()
    config.qa.max_searches = 0
    capability = create_rag(db_path=temp_db_path, config=config)
    capability.state = RAGState()

    result = await capability._search("anything", None)

    assert (
        result
        == "Search limit reached. Answer the question using the results you already have."
    )
    assert await capability._cite([]) == "Registered 0 citations (empty chunk_ids)."


@pytest.mark.asyncio
async def test_cite_resolves_direct_chunk_ids_and_reuses_document_lookup(temp_db_path):
    capability = create_rag(db_path=temp_db_path, config=AppConfig())
    capability.state = RAGState()
    client = AsyncMock()
    client.get_chunk_by_id.side_effect = [
        Chunk(id="chunk-1", document_id="doc-1", content="first"),
        Chunk(id="chunk-2", document_id="doc-1", content="second"),
    ]
    client.get_document_by_id.return_value = SimpleNamespace(
        uri="test://document",
        title="Document",
        metadata={"topic": "ai"},
    )
    capability.rag = client

    result = await capability._cite(["chunk-1", "chunk-2"])

    assert result == "Registered 2 citation(s)."
    assert capability.state.citations == ["chunk-1", "chunk-2"]
    assert capability.state.citation_index["chunk-1"].index == 1
    assert capability.state.citation_index["chunk-2"].index == 2
    assert capability.state.citation_index["chunk-1"].document_meta == {"topic": "ai"}
    client.get_document_by_id.assert_awaited_once_with("doc-1")


@pytest.mark.asyncio
async def test_analysis_records_new_sandbox_search_results(temp_db_path):
    capability = create_analysis(db_path=temp_db_path, config=AppConfig())
    existing = SearchResult(content="existing", score=1, chunk_id="chunk-1")
    new = SearchResult(content="new", score=1, chunk_id="chunk-2")
    capability.state = AnalysisState(searches={"_sandbox": [existing]})
    sandbox = AsyncMock()
    sandbox.execute.return_value = SandboxResult(stdout="done", stderr="", success=True)
    sandbox._search_results = [existing, new]
    capability.sandbox = cast(Sandbox, sandbox)

    result = await capability._execute_code("print('done')")

    assert result == "done"
    assert [item.chunk_id for item in capability.state.searches["_sandbox"]] == [
        "chunk-1",
        "chunk-2",
    ]


@pytest.mark.asyncio
async def test_native_agent_composition_initializes_host_state(temp_db_path):
    capability = create_rag(
        db_path=temp_db_path,
        config=AppConfig(),
        defer_loading=False,
    )
    deps = Deps()
    agent = Agent(
        TestModel(call_tools=[]),
        deps_type=Deps,
        capabilities=[capability],
    )

    result = await agent.run("Hello", deps=deps)

    assert result.output == "success (no tool calls)"
    assert deps.state["rag"] == RAGState().model_dump(mode="json")


@pytest.mark.asyncio
async def test_deferred_capability_loads_native_tools(temp_db_path):
    seen_instructions = []
    loaded_payloads = []

    def model_function(messages, info):
        seen_instructions.append(info.instructions or "")
        loaded_payloads.extend(
            str(part.content)
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "load_capability"
        )
        loaded = any(
            isinstance(part, ToolReturnPart) and part.tool_name == "load_capability"
            for message in messages
            for part in message.parts
        )
        if not loaded:
            return ModelResponse(
                parts=[ToolCallPart("load_capability", {"id": "haiku-rag"})]
            )
        return ModelResponse(parts=[TextPart("loaded")])

    agent = Agent(
        FunctionModel(model_function),
        deps_type=Deps,
        capabilities=[create_rag(db_path=temp_db_path, config=AppConfig())],
    )

    result = await agent.run("Use RAG", deps=Deps())

    assert result.output == "loaded"
    assert "# RAG" not in seen_instructions[0]
    assert "# RAG" in loaded_payloads[0]
    assert "rag_search" in loaded_payloads[0]


def test_prior_turn_tool_results_are_compacted_but_current_evidence_is_kept():
    messages = [
        ModelRequest(parts=[UserPromptPart("old question")]),
        ModelResponse(parts=[ToolCallPart("rag_search", {}, "old-call")]),
        ModelRequest(
            parts=[ToolReturnPart("rag_search", "large old evidence", "old-call")]
        ),
        ModelRequest(parts=[UserPromptPart("current question")]),
        ModelResponse(parts=[ToolCallPart("rag_search", {}, "current-call")]),
        ModelRequest(
            parts=[ToolReturnPart("rag_search", "current evidence", "current-call")]
        ),
    ]

    compacted = _compact_old_tool_returns(messages, frozenset({"rag_search"}))

    old_return = compacted[2].parts[0]
    current_return = compacted[5].parts[0]
    assert isinstance(old_return, ToolReturnPart)
    assert "removed" in str(old_return.content)
    assert isinstance(current_return, ToolReturnPart)
    assert current_return.content == "current evidence"


def test_tool_results_are_unchanged_when_history_has_no_user_prompt():
    messages = [
        ModelResponse(parts=[ToolCallPart("rag_search", {}, "current-call")]),
        ModelRequest(
            parts=[ToolReturnPart("rag_search", "current evidence", "current-call")]
        ),
    ]

    compacted = _compact_old_tool_returns(messages, frozenset({"rag_search"}))

    assert compacted is messages
    current_return = compacted[1].parts[0]
    assert isinstance(current_return, ToolReturnPart)
    assert current_return.content == "current evidence"
