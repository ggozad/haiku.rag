from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai import Agent, ModelRetry, RunContext, ToolFailed
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

from haiku.rag.capabilities._base import (
    CITATION_GRACE_REQUESTS,
    _called_own_tool,
    _compact_old_tool_returns,
)
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
        assert {"analysis_search", "analysis_execute_code"}.isdisjoint(
            seen_tools[exhausted]
        )
        assert "analysis_cite" in seen_tools[exhausted]
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

    with pytest.raises(ToolFailed, match="Search limit reached"):
        await capability._search("anything", None)

    with pytest.raises(ModelRetry, match="chunk_ids was empty"):
        await capability._cite([])


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
async def test_cite_reports_unresolved_ids_on_partial_success(temp_db_path):
    capability = create_rag(db_path=temp_db_path, config=AppConfig())
    capability.state = RAGState()
    client = AsyncMock()
    client.get_chunk_by_id.side_effect = [
        Chunk(id="chunk-1", document_id="doc-1", content="first"),
        None,
        None,
    ]
    client.get_document_by_id.return_value = SimpleNamespace(
        uri="test://document",
        title="Document",
        metadata={},
    )
    capability.rag = client

    result = await capability._cite(["chunk-1", "6.43", "6.51.2"])

    assert "Registered 1 citation(s)" in result
    assert "6.43" in result
    assert "6.51.2" in result
    assert "verbatim" in result
    assert capability.state.citations == ["chunk-1"]


@pytest.mark.asyncio
async def test_cite_repairs_chunk_ids_damaged_in_transcription(temp_db_path):
    """Models mistype opaque UUIDs; near misses resolve to the retrieved id."""
    true_id = "b8e25ea1-0bb3-48b1-8fea-2ac1f148bf7c"
    unrelated = "9c2cd07e-5a3f-45a6-968d-cbd6f06ab57b"
    capability = create_rag(db_path=temp_db_path, config=AppConfig())
    capability.state = RAGState(
        searches={
            "q": [
                SearchResult(
                    content="evidence",
                    score=1.0,
                    chunk_id=true_id,
                    document_id="doc-1",
                    document_uri="test://document",
                )
            ]
        }
    )
    client = AsyncMock()
    client.get_chunk_by_id.return_value = None
    capability.rag = client

    dropped_char = "b8e25ea1-0bb3-48b1-8fea-2ac1f148bf7"
    dropped_group = "0bb3-48b1-8fea-2ac1f148bf7c"

    assert await capability._cite([dropped_char]) == "Registered 1 citation(s)."
    assert await capability._cite([dropped_group]) == "Registered 1 citation(s)."
    assert capability.state.citations == [true_id]

    # An unrelated UUID is never attributed to a retrieved neighbour.
    with pytest.raises(ModelRetry, match=unrelated):
        await capability._cite([unrelated])

    assert capability.state.citations == [true_id]
    client.get_chunk_by_id.assert_awaited_once_with(unrelated)


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
async def test_failed_tool_reaches_the_model_and_the_run_continues(temp_db_path):
    """A `ToolFailed` tool leaves a failed result in history and answers anyway."""
    config = AppConfig()
    config.qa.max_searches = 0
    calls = 0

    def model_function(_messages, _info):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart("rag_search", {"query": "x"})])
        return ModelResponse(parts=[TextPart("answered from what I had")])

    agent = Agent(
        FunctionModel(model_function),
        deps_type=Deps,
        capabilities=[
            create_rag(db_path=temp_db_path, config=config, defer_loading=False)
        ],
    )

    result = await agent.run("question", deps=Deps())

    assert result.output == "answered from what I had"
    failed = [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.outcome == "failed"
    ]
    assert [part.tool_name for part in failed] == ["rag_search"]
    assert "Search limit reached" in str(failed[0].content)


@pytest.mark.asyncio
async def test_analysis_execution_limit_fails_the_tool(temp_db_path):
    config = AppConfig()
    config.analysis.max_executions = 0
    capability = create_analysis(db_path=temp_db_path, config=config)
    capability.state = AnalysisState()

    with pytest.raises(ToolFailed, match="Code-execution limit reached"):
        await capability._execute_code("print('done')")


@pytest.mark.asyncio
async def test_spent_search_budget_is_announced_but_keeps_the_tool(rag_db):
    """A spent budget is announced; the tool stays declared to avoid a dead run.

    Withdrawing it would make a model that calls it anyway hit `Unknown tool
    name`, which exhausts the agent's unknown-tool retries and aborts the run.
    """
    config = AppConfig()
    config.qa.max_searches = 1
    seen_tools = []
    seen_instructions = []
    calls = 0

    def model_function(_messages, info):
        nonlocal calls
        calls += 1
        seen_tools.append({tool.name for tool in info.function_tools})
        seen_instructions.append(info.instructions or "")
        if calls == 1:
            return ModelResponse(
                parts=[ToolCallPart("rag_search", {"query": "machine learning"})]
            )
        return ModelResponse(parts=[TextPart("answered")])

    agent = Agent(
        FunctionModel(model_function),
        deps_type=Deps,
        capabilities=[create_rag(db_path=rag_db, config=config, defer_loading=False)],
    )

    result = await agent.run("question", deps=Deps())

    assert result.output == "answered"
    assert {"rag_search", "rag_cite"} <= seen_tools[1]
    assert "spent its budget for rag_search" in seen_instructions[1]


def test_grace_window_ignores_other_capabilities_turns():
    """Only this capability's own tool calls may spend its cite window.

    A multi-capability agent spends turns elsewhere; those must not expire the
    window that exists to give this capability a chance to cite.
    """
    rag_tools = frozenset({"rag_search", "rag_cite"})

    # Nothing to attribute before the model has responded at all.
    assert not _called_own_tool(
        [ModelRequest(parts=[UserPromptPart(content="q")])], rag_tools
    )
    assert not _called_own_tool(
        [ModelResponse(parts=[ToolCallPart("analysis_search", {"query": "x"})])],
        rag_tools,
    )
    assert not _called_own_tool(
        [ModelResponse(parts=[TextPart("just talking")])], rag_tools
    )
    assert _called_own_tool(
        [ModelResponse(parts=[ToolCallPart("rag_cite", {"chunk_ids": ["a"]})])],
        rag_tools,
    )
    # Only the most recent response counts, not any earlier one.
    assert not _called_own_tool(
        [
            ModelResponse(parts=[ToolCallPart("rag_cite", {"chunk_ids": ["a"]})]),
            ModelRequest(parts=[UserPromptPart(content="next")]),
            ModelResponse(parts=[ToolCallPart("analysis_search", {"query": "x"})]),
        ],
        rag_tools,
    )


@pytest.mark.asyncio
async def test_spent_search_notice_points_at_code_while_it_has_budget(temp_db_path):
    """Analysis must be sent to the sandbox, not told to answer, while it can.

    In-code `search()` bypasses `qa.max_searches`, and the instructions tell the
    model to escalate to code when search results are insufficient.
    """
    config = AppConfig()
    config.qa.max_searches = 2
    capability = create_analysis(db_path=temp_db_path, config=config)
    capability.search_count = 2

    notice = capability._budget_notice()

    assert notice is not None
    assert "analysis_search" in notice
    assert "analysis_execute_code" in notice

    # Once the code budget is gone too there is nowhere left to send it.
    capability.execute_count = config.analysis.max_executions
    notice = capability._budget_notice()
    assert notice is not None
    assert "analysis_execute_code" in notice
    assert capability._evidence_tool_names() <= capability._spent_tool_names()


@pytest.mark.asyncio
async def test_spent_search_notice_tells_rag_to_answer(temp_db_path):
    """Search is the RAG capability's only evidence tool, so stopping is right."""
    config = AppConfig()
    config.qa.max_searches = 2
    capability = create_rag(db_path=temp_db_path, config=config)
    capability.search_count = 2

    notice = capability._budget_notice()

    assert notice is not None
    assert "rag_search" in notice
    assert capability._evidence_tool_names() == {"rag_search"}


@pytest.mark.asyncio
async def test_spent_execution_budget_joins_the_notice(temp_db_path):
    config = AppConfig()
    config.analysis.max_executions = 3
    capability = create_analysis(db_path=temp_db_path, config=config)

    assert capability._spent_tool_names() == set()

    capability.execute_count = 3

    assert capability._spent_tool_names() == {"analysis_execute_code"}
    notice = capability._budget_notice()
    assert notice is not None
    assert "analysis_execute_code" in notice


@pytest.mark.asyncio
async def test_exhausted_run_can_still_register_citations(rag_db):
    """The cite tool outlives the request limit so evidence is not lost.

    Reproduces the measured pathology: the model burns its request budget and
    reaches the limit, at which point it must still be able to cite what it
    already found.
    """
    config = AppConfig()
    seen_tools = []
    calls = 0
    chunk_id: str | None = None

    def model_function(_messages, info):
        nonlocal calls
        calls += 1
        seen_tools.append({tool.name for tool in info.function_tools})
        if calls == 1:
            return ModelResponse(
                parts=[ToolCallPart("rag_search", {"query": "machine learning"})]
            )
        if calls == 2:
            return ModelResponse(
                parts=[ToolCallPart("rag_cite", {"chunk_ids": [chunk_id]})]
            )
        return ModelResponse(parts=[TextPart("answered from gathered evidence")])

    capability = create_rag(
        db_path=rag_db,
        config=config,
        defer_loading=False,
        request_limit=1,
    )
    agent = Agent(
        FunctionModel(model_function),
        deps_type=Deps,
        capabilities=[capability],
    )
    deps = Deps()

    async with agent.iter("question", deps=deps) as run:
        async for _node in run:
            if chunk_id is None:
                searches = deps.state.get("rag", {}).get("searches") or {}
                for results in searches.values():
                    if results:
                        chunk_id = results[0]["chunk_id"]
                        break

    assert chunk_id is not None
    # The limit lands on request 2, where cite must still be offered.
    assert "rag_search" not in seen_tools[1]
    assert "rag_cite" in seen_tools[1]
    assert deps.state["rag"]["citations"] == [chunk_id]


@pytest.mark.asyncio
async def test_cite_tool_is_withdrawn_after_the_grace_window(temp_db_path):
    capability = create_rag(
        db_path=temp_db_path,
        config=AppConfig(),
        defer_loading=False,
        request_limit=2,
    )
    tool_defs = [
        SimpleNamespace(name=name, capability_id=capability.id)
        for name in ("rag_search", "rag_cite")
    ]
    ctx = make_context(Deps())

    capability.request_count = 2
    kept = await capability.prepare_tools(ctx, cast(Any, tool_defs))
    assert {tool.name for tool in kept} == {"rag_cite"}
    notice = capability._budget_notice()
    assert notice is not None and "rag_cite" in notice

    capability.grace_requests_used = CITATION_GRACE_REQUESTS
    kept = await capability.prepare_tools(ctx, cast(Any, tool_defs))
    assert kept == []
    # The notice must never point at a tool prepare_tools has withdrawn:
    # calling a missing tool burns the agent's unknown-tool retries and can
    # abort the run.
    notice = capability._budget_notice()
    assert notice is not None
    assert "rag_cite" not in notice
    assert "no longer available" in notice


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stderr", "expect_hint"),
    [
        ("TypeError: '_io.TextIOWrapper' object is not iterable", True),
        ("TypeError: 'list' object is not an iterator", False),
    ],
)
async def test_sandbox_iteration_failure_carries_the_workaround(
    temp_db_path, stderr, expect_hint
):
    """A model that iterates a file object gets told what to do instead."""
    capability = create_analysis(db_path=temp_db_path, config=AppConfig())
    capability.state = AnalysisState()
    sandbox = AsyncMock()
    sandbox.execute.return_value = SandboxResult(
        stdout="", stderr=stderr, success=False
    )
    sandbox._search_results = []
    capability.sandbox = cast(Sandbox, sandbox)

    with pytest.raises(ToolFailed) as failure:
        await capability._execute_code(
            "for line in open('/documents/x/items.jsonl'): pass"
        )

    assert (".readlines()" in str(failure.value)) is expect_hint


@pytest.mark.asyncio
async def test_analysis_sandbox_failure_records_execution_and_fails_the_tool(
    temp_db_path,
):
    capability = create_analysis(db_path=temp_db_path, config=AppConfig())
    capability.state = AnalysisState()
    capability.outer_state = {}
    sandbox = AsyncMock()
    sandbox.execute.return_value = SandboxResult(
        stdout="partial", stderr="NameError: undefined", success=False
    )
    sandbox._search_results = []
    capability.sandbox = cast(Sandbox, sandbox)

    with pytest.raises(ToolFailed, match="NameError: undefined"):
        await capability._with_state(capability._execute_code("boom"))

    entry = capability.state.executions[-1]
    assert entry.success is False
    assert entry.stderr == "NameError: undefined"
    assert capability.outer_state["analysis"]["executions"][-1]["code"] == "boom"


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
