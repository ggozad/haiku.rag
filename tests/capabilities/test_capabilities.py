from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    CallDeferred,
    DeferredToolRequests,
    DeferredToolResults,
    ModelRetry,
    RunContext,
    ToolFailed,
)
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
)
from haiku.rag.capabilities.analysis import AnalysisCapability, AnalysisState
from haiku.rag.capabilities.analysis import create_capability as create_analysis
from haiku.rag.capabilities.ledger import (
    CapabilityEvidenceRecord,
    citation_status,
)
from haiku.rag.capabilities.rag import AGENT_PREAMBLE, RAGCapability, RAGState
from haiku.rag.capabilities.rag import create_capability as create_rag
from haiku.rag.client.scope import DatabaseRef
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


def _placed(capability) -> "Path | None":
    """Where a capability covering one local database will open it."""
    [ref] = capability.scope.databases
    return ref.db_path


def test_capability_factories_resolve_environment_and_defaults(
    temp_db_path, monkeypatch
):
    config = AppConfig()
    monkeypatch.setenv("HAIKU_RAG_DB", str(temp_db_path))
    assert _placed(create_rag(config=config)) == temp_db_path

    monkeypatch.delenv("HAIKU_RAG_DB")
    assert _placed(create_rag(config=config)) == (
        config.storage.data_dir / "haiku.rag.lancedb"
    )

    for factory in (create_rag, create_analysis):
        db_path = _placed(factory(db_path=str(temp_db_path), config=config))
        assert db_path == temp_db_path
        assert isinstance(db_path, Path)

    with patch("haiku.rag.config.get_config", return_value=config):
        assert create_rag().config is config
        assert create_analysis().config is config


class TestACapabilityFollowsTheConfiguredLocation:
    """A capability nobody handed a client opens one for itself, and has to open
    the database the configuration places rather than the default directory."""

    def _config(self, tmp_path, uri: str) -> AppConfig:
        from haiku.rag.config.models import LanceDBConfig, StorageConfig

        return AppConfig(
            lancedb=LanceDBConfig(uri=uri),
            storage=StorageConfig(data_dir=tmp_path / "elsewhere"),
        )

    def test_a_configured_uri_is_left_to_the_client(self, tmp_path):
        """A path overrides a configured location, so manufacturing one would
        send the capability to the default directory instead of the bucket."""
        located = tmp_path / "notes.lancedb"
        for factory in (create_rag, create_analysis):
            [local] = factory(
                config=self._config(tmp_path, str(located))
            ).scope.databases
            assert local == DatabaseRef.configured(None, str(located))

            remote = self._config(tmp_path, "s3://bucket/one.lancedb")
            [ref] = factory(config=remote).scope.databases
            assert ref == DatabaseRef(None, "s3://bucket/one.lancedb", None)

    @pytest.mark.asyncio
    async def test_it_opens_the_database_the_uri_places(self, tmp_path):
        from haiku.rag.client import HaikuRAG

        located = tmp_path / "notes.lancedb"
        config = self._config(tmp_path, str(located))
        async with HaikuRAG(config=config, create=True):
            pass

        capability = create_rag(config=config)
        rag = await capability._ensure_rag()
        try:
            assert rag.store.db_path == located
        finally:
            await capability._close()

    def test_an_explicit_path_still_overrides_the_configured_uri(self, tmp_path):
        config = self._config(tmp_path, str(tmp_path / "notes.lancedb"))
        chosen = tmp_path / "chosen.lancedb"

        assert _placed(create_rag(db_path=chosen, config=config)) == chosen

    def test_the_environment_still_overrides_the_configured_uri(
        self, tmp_path, monkeypatch
    ):
        config = self._config(tmp_path, "s3://bucket/one.lancedb")
        monkeypatch.setenv("HAIKU_RAG_DB", str(tmp_path / "from-env.lancedb"))

        assert _placed(create_rag(config=config)) == tmp_path / "from-env.lancedb"


@pytest.mark.asyncio
async def test_a_string_db_path_opens_a_store(temp_db_path):
    """Store calls `absolute()` and `exists()` on db_path, which a str lacks."""
    from haiku.rag.client import HaikuRAG

    config = AppConfig()
    async with HaikuRAG(temp_db_path, config, create=True):
        pass

    capability = create_rag(db_path=str(temp_db_path), config=config)
    try:
        rag = await capability._ensure_rag()
        assert rag.store.db_path == temp_db_path
    finally:
        await capability._close()


def test_domain_preamble_is_added_to_capability_instructions(temp_db_path):
    config = AppConfig(
        prompts=PromptsConfig(domain_preamble="The corpus contains solar manuals.")
    )
    capability = create_rag(db_path=temp_db_path, config=config)

    assert capability.get_instructions().startswith(
        "The corpus contains solar manuals.\n\n# RAG"
    )


def _single_database_client() -> AsyncMock:
    """A stand-in for a client covering one unnamed database.

    A bare AsyncMock answers every attribute with a truthy Mock, so
    `covers_multiple` would read as a set of databases, `source` would reach a
    validated field, and `clients_covering` would return a Mock where the code
    iterates clients.
    """
    client = AsyncMock()
    client.covers_multiple = False
    client.source_names = ()
    client.source = None
    client.clients_covering.return_value = [client]
    return client


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
    client = _single_database_client()
    capability.rag = client
    error = RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        await capability.on_run_error(make_context(Deps()), error=error)

    client.__aexit__.assert_awaited_once_with(None, None, None)
    assert capability.rag is None


@pytest.mark.asyncio
async def test_a_spent_search_budget_fails_the_tool(temp_db_path):
    config = AppConfig()
    config.qa.max_searches = 0
    capability = create_rag(db_path=temp_db_path, config=config)
    capability.state = RAGState()

    with pytest.raises(ToolFailed, match="Search limit reached"):
        await capability._search("anything", None)


def _stub_client(*batches: list[SearchResult]) -> AsyncMock:
    """A client whose successive searches return the given result batches."""
    client = AsyncMock()
    client.search.side_effect = list(batches)
    client.expand_context.side_effect = lambda results: results
    return client


@pytest.mark.asyncio
async def test_a_fruitless_search_says_so(temp_db_path):
    """A blank tool return reads as a broken tool, not as an empty corpus."""
    capability = create_rag(db_path=temp_db_path, config=AppConfig())
    capability.state = RAGState()
    capability.borrowed_rag = _stub_client([])

    assert await capability._search("nothing about this", None) == "No results found."


@pytest.mark.asyncio
async def test_a_narrower_repeat_keeps_what_the_wider_search_returned(temp_db_path):
    """One query, two limits: the model can still cite the results it was shown."""
    capability = create_rag(db_path=temp_db_path, config=AppConfig())
    capability.state = RAGState()
    capability.borrowed_rag = _stub_client(
        [
            SearchResult(content="first", score=1.0, chunk_id="chunk-1"),
            SearchResult(content="second", score=0.9, chunk_id="chunk-2"),
            SearchResult(content="third", score=0.8, chunk_id="chunk-3"),
        ],
        [SearchResult(content="first", score=1.0, chunk_id="chunk-1")],
    )

    await capability._search("Figure 3-1", 20)
    await capability._search("Figure 3-1", None)

    stored = capability.state.searches["Figure 3-1"]
    assert [result.chunk_id for result in stored] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]


@pytest.mark.asyncio
async def test_two_databases_holding_one_chunk_id_both_survive(temp_db_path):
    """A database copied from another holds the same chunk ids, so what tells
    two results apart is the database and the id together."""
    capability = create_rag(db_path=temp_db_path, config=AppConfig())
    capability.state = RAGState()
    capability.borrowed_rag = _stub_client(
        [SearchResult(content="alpha", score=1.0, chunk_id="c1", source="alpha")],
        [
            SearchResult(content="alpha", score=1.0, chunk_id="c1", source="alpha"),
            SearchResult(content="beta", score=0.9, chunk_id="c1", source="beta"),
        ],
    )

    await capability._search("cats", 20)
    await capability._search("cats", None)

    stored = capability.state.searches["cats"]
    assert [(r.source, r.chunk_id) for r in stored] == [
        ("alpha", "c1"),
        ("beta", "c1"),
    ]


@pytest.mark.asyncio
async def test_cite_resolves_direct_chunk_ids_and_reuses_document_lookup(temp_db_path):
    capability = create_rag(db_path=temp_db_path, config=AppConfig())
    capability.state = RAGState(evidence=CapabilityEvidenceRecord(question=0))
    client = _single_database_client()
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
    capability.state = RAGState(evidence=CapabilityEvidenceRecord(question=0))
    client = _single_database_client()
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
    # Never ask for another call here: something did register, and a model that
    # keeps mangling ids obeys the ask until the run dies on output retries.
    assert "again" not in result
    assert capability.state.citations == ["chunk-1"]


@pytest.mark.asyncio
async def test_cite_repairs_chunk_ids_damaged_in_transcription(temp_db_path):
    """Models mistype opaque UUIDs; near misses resolve to the retrieved id."""
    true_id = "b8e25ea1-0bb3-48b1-8fea-2ac1f148bf7c"
    unrelated = "9c2cd07e-5a3f-45a6-968d-cbd6f06ab57b"
    capability = create_rag(db_path=temp_db_path, config=AppConfig())
    capability.state = RAGState(
        evidence=CapabilityEvidenceRecord(question=0),
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
        },
    )
    client = _single_database_client()
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
async def test_a_spent_execution_budget_is_not_evidence(temp_db_path):
    """Nothing was produced to ground an answer on, so nothing is recorded."""
    config = AppConfig()
    config.analysis.max_executions = 0
    capability = create_analysis(db_path=temp_db_path, config=config)
    capability.state = AnalysisState()
    capability.epoch = 5

    with pytest.raises(ToolFailed):
        await capability._execute_code("print('done')")

    assert capability.state.evidence.latest_evidence_epoch == 0


@pytest.mark.parametrize(
    ("success", "stdout", "expected_epoch"),
    [
        pytest.param(True, "42", 5, id="succeeded"),
        pytest.param(True, "", 5, id="succeeded without output"),
        pytest.param(False, "42", 5, id="failed after printing"),
        pytest.param(False, "", 0, id="failed without printing"),
    ],
)
@pytest.mark.asyncio
async def test_only_a_code_execution_the_model_can_read_is_evidence(
    temp_db_path, success, stdout, expected_epoch
):
    """A raised error with nothing printed grounds nothing, so it is not evidence.

    A failure that printed first does ground an answer, and so does a successful
    run whose outcome is that it printed nothing.
    """
    capability = create_analysis(db_path=temp_db_path, config=AppConfig())
    capability.state = AnalysisState(evidence=CapabilityEvidenceRecord(question=0))
    capability.epoch = 5
    sandbox = AsyncMock(spec=Sandbox)
    sandbox._search_results = []
    sandbox.execute.return_value = SandboxResult(
        stdout=stdout, stderr="" if success else "boom", success=success
    )
    capability.sandbox = sandbox

    if success:
        await capability._execute_code("print(42)")
    else:
        with pytest.raises(ToolFailed):
            await capability._execute_code("print(42)")

    assert capability.state.evidence.latest_evidence_epoch == expected_epoch


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
    assert capability.evidence_tool_names() <= capability._spent_tool_names()


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
    assert capability.evidence_tool_names() == {"rag_search"}


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
    assert deps.state["rag"] == RAGState(
        evidence=CapabilityEvidenceRecord(question=0)
    ).model_dump(mode="json")


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


def _resuming_deps() -> Deps:
    """State as a resumption always finds it: the question already identified.

    A run that resumes has been through ``for_run`` before, so the identity of the
    question in progress is stored. Fabricating the history without it is a state
    the design does not produce, and is rejected rather than guessed at.
    """
    return Deps(
        state={
            "rag": RAGState(evidence=CapabilityEvidenceRecord(question=0)).model_dump(
                mode="json"
            )
        }
    )


def _in_flight_history() -> list[Any]:
    """A question already asked and searched, still awaiting its answer."""
    return [
        ModelRequest(parts=[UserPromptPart("what does the supervisor do?")]),
        ModelResponse(parts=[ToolCallPart("rag_search", {"query": "s"}, "call-1")]),
        ModelRequest(
            parts=[ToolReturnPart("rag_search", "EVIDENCE FOR THE LIVE TURN", "call-1")]
        ),
    ]


def _record(deps: Deps, namespace: str) -> CapabilityEvidenceRecord:
    return CapabilityEvidenceRecord.model_validate(deps.state[namespace]["evidence"])


async def _stub_search(self, query: str, _limit: int | None) -> str:
    """Record a result the way the real search does, so citing resolves."""
    cast(Any, self.state).searches[query] = [
        SearchResult(content="evidence", score=1.0, chunk_id="chunk-1")
    ]
    self._note_evidence()
    return "EVIDENCE"


@pytest.mark.asyncio
async def test_a_question_takes_its_own_identity_and_both_capabilities_agree(
    temp_db_path,
):
    """Identity is derived from the conversation, so no counter is shared."""
    rag = create_rag(db_path=temp_db_path, config=AppConfig(), defer_loading=False)
    analysis = create_analysis(
        db_path=temp_db_path, config=AppConfig(), defer_loading=False
    )

    async def model(_messages, _info):
        return ModelResponse(parts=[TextPart("answer")])

    agent = Agent(FunctionModel(model), deps_type=Deps, capabilities=[rag, analysis])
    deps = Deps()

    first = await agent.run("first question", deps=deps)
    first_identity = _record(deps, "rag").question
    await agent.run("second question", deps=deps, message_history=first.all_messages())
    second_identity = _record(deps, "rag").question

    assert first_identity == 0
    assert second_identity is not None and second_identity > 0
    assert _record(deps, "analysis").question == second_identity


@pytest.mark.asyncio
async def test_a_resumption_keeps_the_identity_of_the_question_in_progress(
    temp_db_path,
):
    rag = create_rag(db_path=temp_db_path, config=AppConfig(), defer_loading=False)

    async def model(_messages, _info):
        return ModelResponse(parts=[TextPart("answer")])

    agent = Agent(FunctionModel(model), deps_type=Deps, capabilities=[rag])
    deps = Deps(
        state={
            "rag": RAGState(
                evidence=CapabilityEvidenceRecord(question=7, in_progress=True)
            ).model_dump(mode="json")
        }
    )
    history = [
        *_in_flight_history(),
        ModelResponse(parts=[ToolCallPart("external_tool", {}, "call-2")]),
    ]

    await agent.run(
        "carry on",
        deferred_tool_results=DeferredToolResults(calls={"call-2": "external result"}),
        message_history=history,
        deps=deps,
    )

    assert _record(deps, "rag").question == 7


@pytest.mark.asyncio
async def test_resuming_without_a_stored_identity_fails_instead_of_guessing(
    temp_db_path,
):
    """Adopting the message count would relabel a question already in progress.

    Every declaration and epoch comparison in it would then be judged against the
    wrong question, silently. This state is not one the design produces, so it is
    reported rather than repaired.
    """
    rag = create_rag(db_path=temp_db_path, config=AppConfig(), defer_loading=False)

    async def model(_messages, _info):  # pragma: no cover - never reached
        return ModelResponse(parts=[TextPart("answer")])

    agent = Agent(FunctionModel(model), deps_type=Deps, capabilities=[rag])

    with pytest.raises(RuntimeError, match="no stored question identity"):
        await agent.run(
            "carry on",
            deferred_tool_results=DeferredToolResults(
                calls={"call-2": "external result"}
            ),
            message_history=[
                *_in_flight_history(),
                ModelResponse(parts=[ToolCallPart("external_tool", {}, "call-2")]),
            ],
            deps=Deps(),
        )


@pytest.mark.asyncio
async def test_citing_after_searching_grounds_the_question(temp_db_path):
    """The whole rule, end to end, with no compactor and no policy capability."""
    rag = create_rag(db_path=temp_db_path, config=AppConfig(), defer_loading=False)
    calls = iter(
        [
            [ToolCallPart("rag_search", {"query": "supervisor"}, "call-1")],
            [ToolCallPart("rag_cite", {"chunk_ids": ["chunk-1"]}, "call-2")],
            [TextPart("answer")],
        ]
    )

    async def model(_messages, _info):
        return ModelResponse(parts=next(calls))

    agent = Agent(FunctionModel(model), deps_type=Deps, capabilities=[rag])
    deps = Deps()

    with patch.object(RAGCapability, "_search", _stub_search):
        await agent.run("what does the supervisor do?", deps=deps)

    record = _record(deps, "rag")
    question = record.question
    assert question is not None
    assert record.declaration is not None
    assert [ref.chunk_id for ref in record.declaration.refs] == ["chunk-1"]
    assert record.occurrences["chunk-1"].retrieved_in_questions == [question]
    assert citation_status([record], question=question) == "grounded"


@pytest.mark.asyncio
async def test_searching_after_citing_leaves_the_question_uncited(temp_db_path):
    rag = create_rag(db_path=temp_db_path, config=AppConfig(), defer_loading=False)
    calls = iter(
        [
            [ToolCallPart("rag_search", {"query": "supervisor"}, "call-1")],
            [ToolCallPart("rag_cite", {"chunk_ids": ["chunk-1"]}, "call-2")],
            [ToolCallPart("rag_search", {"query": "again"}, "call-3")],
            [TextPart("answer")],
        ]
    )

    async def model(_messages, _info):
        return ModelResponse(parts=next(calls))

    agent = Agent(FunctionModel(model), deps_type=Deps, capabilities=[rag])
    deps = Deps()

    with patch.object(RAGCapability, "_search", _stub_search):
        await agent.run("what does the supervisor do?", deps=deps)

    record = _record(deps, "rag")
    question = record.question
    assert question is not None
    assert record.declaration is not None
    assert citation_status([record], question=question) == "missing"


@pytest.mark.asyncio
async def test_a_citation_in_the_same_request_as_its_search_is_not_current(
    temp_db_path,
):
    """Two calls in one response share an epoch, and citing must follow seeing."""
    rag = create_rag(db_path=temp_db_path, config=AppConfig(), defer_loading=False)
    calls = iter(
        [
            [
                ToolCallPart("rag_search", {"query": "supervisor"}, "call-1"),
                ToolCallPart("rag_cite", {"chunk_ids": ["chunk-1"]}, "call-2"),
            ],
            [TextPart("answer")],
        ]
    )

    async def model(_messages, _info):
        return ModelResponse(parts=next(calls))

    agent = Agent(FunctionModel(model), deps_type=Deps, capabilities=[rag])
    deps = Deps()

    with patch.object(RAGCapability, "_search", _stub_search):
        await agent.run("what does the supervisor do?", deps=deps)

    record = _record(deps, "rag")
    question = record.question
    assert question is not None
    assert record.declaration is not None
    assert record.declaration.epoch == record.latest_evidence_epoch
    assert citation_status([record], question=question) == "missing"


@pytest.mark.asyncio
async def test_evidence_cited_in_two_questions_keeps_both_in_the_record(temp_db_path):
    """Occurrences outlive the question that wrote them, through the state dict."""
    rag = create_rag(db_path=temp_db_path, config=AppConfig(), defer_loading=False)
    calls = iter(
        [
            [ToolCallPart("rag_search", {"query": "supervisor"}, "call-1")],
            [ToolCallPart("rag_cite", {"chunk_ids": ["chunk-1"]}, "call-2")],
            [TextPart("first answer")],
            [ToolCallPart("rag_search", {"query": "supervisor again"}, "call-3")],
            [ToolCallPart("rag_cite", {"chunk_ids": ["chunk-1"]}, "call-4")],
            [TextPart("second answer")],
        ]
    )

    async def model(_messages, _info):
        return ModelResponse(parts=next(calls))

    agent = Agent(FunctionModel(model), deps_type=Deps, capabilities=[rag])
    deps = Deps()

    with patch.object(RAGCapability, "_search", _stub_search):
        first = await agent.run("who supervises?", deps=deps)
        first_question = _record(deps, "rag").question
        await agent.run(
            "and who supervises them?", deps=deps, message_history=first.all_messages()
        )

    record = _record(deps, "rag")
    assert record.occurrences["chunk-1"].cited_in_questions == [
        first_question,
        record.question,
    ]
    assert record.question != first_question


@pytest.mark.asyncio
async def test_a_run_with_no_prompt_and_no_history_starts_a_question(temp_db_path):
    """An instructions-only run is a first question, not a resumption.

    There is no question in progress to keep an identity for, so nothing is
    missing and the run proceeds with a fresh one.
    """
    rag = create_rag(db_path=temp_db_path, config=AppConfig(), defer_loading=False)

    async def model(_messages, _info):
        return ModelResponse(parts=[TextPart("answer")])

    agent = Agent(FunctionModel(model), deps_type=Deps, capabilities=[rag])
    deps = Deps()

    await agent.run(deps=deps)

    assert _record(deps, "rag").question == 0


@pytest.mark.asyncio
async def test_citing_without_searching_grounds_the_question(temp_db_path):
    """A direct chunk-id citation stands on its own, with no evidence outcome.

    Epochs count messages and so start above zero, which is what lets a
    declaration made in the first request still beat an empty evidence horizon.
    """
    rag = create_rag(db_path=temp_db_path, config=AppConfig(), defer_loading=False)
    calls = iter(
        [
            [ToolCallPart("rag_cite", {"chunk_ids": ["chunk-1"]}, "call-1")],
            [TextPart("answer")],
        ]
    )

    async def model(_messages, _info):
        return ModelResponse(parts=next(calls))

    client = _single_database_client()
    client.get_chunk_by_id.return_value = Chunk(
        id="chunk-1", document_id="doc-1", content="evidence"
    )
    client.get_document_by_id.return_value = None
    agent = Agent(FunctionModel(model), deps_type=Deps, capabilities=[rag])
    deps = Deps()

    with patch.object(RAGCapability, "_ensure_rag", AsyncMock(return_value=client)):
        await agent.run("cite chunk-1", deps=deps)

    record = _record(deps, "rag")
    question = record.question
    assert question is not None
    assert record.latest_evidence_epoch == 0
    assert record.declaration is not None
    assert record.declaration.epoch > 0
    assert record.occurrences["chunk-1"].retrieved_in_questions == []
    assert citation_status([record], question=question) == "grounded"


@pytest.mark.asyncio
async def test_a_host_seeded_record_does_not_pass_for_a_resumption(temp_db_path):
    """A seeded record says nothing about a question, so the history has to.

    Seeding one is what a host does when it has no state to send. Its flag is
    unset, so a history that unmistakably awaits the model means the host dropped
    the state of a question in progress, and answering as question zero would
    silently relabel it.
    """
    rag = create_rag(db_path=temp_db_path, config=AppConfig(), defer_loading=False)

    async def model(_messages, _info):  # pragma: no cover - never reached
        return ModelResponse(parts=[TextPart("answer")])

    agent = Agent(FunctionModel(model), deps_type=Deps, capabilities=[rag])

    with pytest.raises(RuntimeError, match="no stored question identity"):
        await agent.run(
            "carry on",
            message_history=[
                *_in_flight_history(),
                ModelResponse(parts=[ToolCallPart("external_tool", {}, "call-2")]),
            ],
            deps=Deps(state={"rag": RAGState().model_dump(mode="json")}),
        )


@pytest.mark.asyncio
async def test_a_resumption_keeps_the_evidence_the_question_already_gathered(
    temp_db_path,
):
    """Clearing it would lose the results the model is still answering from.

    A citation after the resumption then records no provenance, and cannot resolve
    against the expanded search result the model actually saw.
    """
    rag = create_rag(db_path=temp_db_path, config=AppConfig(), defer_loading=False)
    calls = iter(
        [
            [ToolCallPart("rag_search", {"query": "supervisor"}, "call-1")],
            [TextPart("partial answer")],
            [ToolCallPart("rag_cite", {"chunk_ids": ["chunk-1"]}, "call-3")],
            [TextPart("answer")],
        ]
    )

    async def model(_messages, _info):
        return ModelResponse(parts=next(calls))

    agent = Agent(FunctionModel(model), deps_type=Deps, capabilities=[rag])
    deps = Deps()

    with patch.object(RAGCapability, "_search", _stub_search):
        interrupted = await agent.run("what does the supervisor do?", deps=deps)
        identity = _record(deps, "rag").question
        assert identity is not None
        # A run that ends awaiting external work leaves the question in progress,
        # which is what the resumption claims. See the deferred-request test.
        deps.state["rag"]["evidence"]["in_progress"] = True
        await agent.run(
            deferred_tool_results=DeferredToolResults(
                calls={"call-2": "external result"}
            ),
            message_history=[
                *interrupted.all_messages(),
                ModelResponse(parts=[ToolCallPart("external_tool", {}, "call-2")]),
            ],
            deps=deps,
        )

    record = _record(deps, "rag")
    assert record.question == identity
    assert record.occurrences["chunk-1"].retrieved_in_questions == [identity]
    assert citation_status([record], question=identity) == "grounded"


@pytest.mark.asyncio
async def test_a_capability_fetches_its_own_evidences_pictures(temp_db_path):
    """Compaction rehydrates through the owner, which already holds the connection."""
    capability = create_rag(db_path=temp_db_path, config=AppConfig())
    client = AsyncMock()
    client.get_picture_bytes.return_value = b"picture-bytes"
    capability.rag = client

    data = await capability.get_picture_bytes("doc-1", "#/pictures/0", "beta")

    assert data == b"picture-bytes"
    client.get_picture_bytes.assert_awaited_once_with("doc-1", "#/pictures/0", "beta")


@pytest.mark.asyncio
async def test_citing_nothing_is_a_valid_declaration(temp_db_path):
    """A model with nothing to cite must be able to say so.

    Refusing the call left silence as the only way to express it, which is
    indistinguishable from forgetting to cite at all.
    """
    capability = create_rag(db_path=temp_db_path, config=AppConfig())
    capability.state = RAGState(evidence=CapabilityEvidenceRecord(question=0))
    capability.epoch = 5

    result = await capability._cite([])

    record = capability.state.evidence
    assert "no" in result.lower()
    assert record.declaration is not None
    assert record.declaration.refs == []
    assert citation_status([record], question=0) == "ungrounded"
    assert capability.state.citations == []


@pytest.mark.asyncio
async def test_citing_nothing_after_citing_something_keeps_it_grounded(temp_db_path):
    """Declaring again cannot narrow what a question already declared."""
    capability = create_rag(db_path=temp_db_path, config=AppConfig())
    capability.state = RAGState(evidence=CapabilityEvidenceRecord(question=0))
    capability.epoch = 5
    client = _single_database_client()
    client.get_chunk_by_id.return_value = Chunk(
        id="chunk-1", document_id="doc-1", content="evidence"
    )
    client.get_document_by_id.return_value = None
    capability.rag = client

    await capability._cite(["chunk-1"])
    await capability._cite([])

    record = capability.state.evidence
    assert citation_status([record], question=0) == "grounded"


@pytest.mark.asyncio
async def test_a_promptless_run_on_a_settled_history_is_a_new_question(temp_db_path):
    """AG-UI hosts never pass a prompt: the client's message is the history.

    Pydantic AI's UI adapter builds `message_history` from the frontend messages
    and calls the agent without a prompt, so the run has no prompt *and* the
    history ends with the user's own request. Reading either as a continuation
    fails every AG-UI host on its first message.
    """
    rag = create_rag(db_path=temp_db_path, config=AppConfig(), defer_loading=False)

    async def model(_messages, _info):
        return ModelResponse(parts=[TextPart("answer")])

    agent = Agent(FunctionModel(model), deps_type=Deps, capabilities=[rag])
    deps = Deps(state={"rag": RAGState().model_dump(mode="json")})
    history: list[Any] = [
        ModelRequest(parts=[UserPromptPart("what does the manual say about masks?")])
    ]

    await agent.run(message_history=history, deps=deps)

    assert _record(deps, "rag").question == len(history)


@pytest.mark.asyncio
async def test_a_promptless_run_on_an_unfinished_tail_is_still_a_continuation(
    temp_db_path,
):
    """A suspended run resumes without a prompt, and must keep its question."""
    rag = create_rag(db_path=temp_db_path, config=AppConfig(), defer_loading=False)

    async def model(_messages, _info):
        return ModelResponse(parts=[TextPart("answer")])

    agent = Agent(FunctionModel(model), deps_type=Deps, capabilities=[rag])
    deps = Deps(
        state={
            "rag": RAGState(
                evidence=CapabilityEvidenceRecord(question=3, in_progress=True)
            ).model_dump(mode="json")
        }
    )

    await agent.run(message_history=_in_flight_history(), deps=deps)

    assert _record(deps, "rag").question == 3


@pytest.mark.asyncio
async def test_a_structured_answer_does_not_leave_the_question_in_progress(
    temp_db_path,
):
    """A settled run ends with a tool return, which says nothing about progress.

    Pydantic AI answers a structured `output_type` by calling an output tool, so the
    history ends with a request carrying that tool's return. Reading the transcript
    shape alone, that is indistinguishable from tool results delivered to a question
    still being answered, and every following question inherited the first one's
    identity.
    """

    class Answer(BaseModel):
        text: str

    rag = create_rag(db_path=temp_db_path, config=AppConfig(), defer_loading=False)

    async def model(_messages, info):
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"text": "answer"})]
        )

    agent = Agent(
        FunctionModel(model),
        deps_type=Deps,
        capabilities=[rag],
        output_type=Answer,
    )
    deps = Deps()

    first = await agent.run("first question", deps=deps)
    first_identity = _record(deps, "rag").question
    assert _record(deps, "rag").in_progress is False

    await agent.run(
        "second question", deps=deps, message_history=list(first.all_messages())
    )

    second_identity = _record(deps, "rag").question
    assert first_identity == 0
    assert second_identity is not None and second_identity > 0


@pytest.mark.asyncio
async def test_a_run_pausing_for_deferred_work_leaves_the_question_in_progress(
    temp_db_path,
):
    """The question is unfinished, so its resumption must find it claimable.

    A deferred tool call ends the run with `DeferredToolRequests` rather than an
    answer. Closing the question here would let the resumption relabel it.
    """
    rag = create_rag(db_path=temp_db_path, config=AppConfig(), defer_loading=False)

    async def model(_messages, _info):
        return ModelResponse(parts=[ToolCallPart("external_tool", {})])

    agent = Agent(
        FunctionModel(model),
        deps_type=Deps,
        capabilities=[rag],
        output_type=[str, DeferredToolRequests],
    )

    @agent.tool_plain
    def external_tool() -> str:
        raise CallDeferred

    deps = Deps()

    result = await agent.run("a question needing external work", deps=deps)

    assert isinstance(result.output, DeferredToolRequests)
    assert _record(deps, "rag").in_progress is True


@pytest.mark.asyncio
async def test_an_answered_question_is_no_longer_in_progress(temp_db_path):
    """The flag is what tells the next run it is asking something new."""
    rag = create_rag(db_path=temp_db_path, config=AppConfig(), defer_loading=False)

    async def model(_messages, _info):
        return ModelResponse(parts=[TextPart("answer")])

    agent = Agent(FunctionModel(model), deps_type=Deps, capabilities=[rag])
    deps = Deps()

    await agent.run("a question", deps=deps)

    record = _record(deps, "rag")
    assert record.question == 0
    assert record.in_progress is False


class TestMultipleCollectionsInstructions:
    """The note follows what a run reads, not what the configuration names, so a
    run over one collection is instructed exactly as it was before collections
    could be named."""

    @staticmethod
    def _config(**databases):
        from haiku.rag.config.models import LanceDBConfig

        return AppConfig(lancedb=LanceDBConfig(databases=databases))

    @staticmethod
    def _client(federated):
        client = AsyncMock()
        client.covers_multiple = len(federated) > 1
        client.source_names = tuple(federated)
        return client

    def test_one_database_is_instructed_as_before(self):
        from haiku.rag.capabilities.analysis import instructions as analysis_text
        from haiku.rag.capabilities.rag import instructions as rag_text

        for factory, baseline in (
            (create_rag, rag_text),
            (create_analysis, analysis_text),
        ):
            for config in (AppConfig(), self._config(alpha="/a.lancedb")):
                capability = factory(db_path=Path("/tmp/x.lancedb"), config=config)
                assert capability.instruction_text == baseline()

    def test_an_explicit_path_opens_one_database(self):
        """A path names one database, whatever the configuration names."""
        from haiku.rag.capabilities.analysis import instructions as analysis_text
        from haiku.rag.capabilities.rag import instructions as rag_text

        config = self._config(alpha="/a.lancedb", beta="/b.lancedb")
        for factory, baseline in (
            (create_rag, rag_text),
            (create_analysis, analysis_text),
        ):
            capability = factory(db_path=Path("/tmp/one.lancedb"), config=config)
            assert capability.instruction_text == baseline()

    def test_a_lent_client_covering_one_database_is_instructed_as_before(self):
        from haiku.rag.capabilities.analysis import instructions as analysis_text
        from haiku.rag.capabilities.rag import instructions as rag_text

        config = self._config(alpha="/a.lancedb", beta="/b.lancedb")
        for factory, baseline in (
            (create_rag, rag_text),
            (create_analysis, analysis_text),
        ):
            capability = factory(config=config, rag=self._client({}))
            assert capability.instruction_text == baseline()

    def test_a_lent_client_covering_a_set_is_told_about_it(self):
        config = self._config(alpha="/a.lancedb", beta="/b.lancedb")
        covering = self._client({"alpha": "/a.lancedb", "beta": "/b.lancedb"})

        for factory in (create_rag, create_analysis):
            capability = factory(config=config, rag=covering)
            assert "Collection:" in capability.get_instructions()

    def test_the_rag_note_names_the_line_a_result_carries(self):
        config = self._config(alpha="/a.lancedb", beta="/b.lancedb")

        text = create_rag(config=config).get_instructions()

        assert "Collection:" in text

    def test_the_analysis_note_separates_the_interfaces(self):
        """The three interfaces name a collection differently, and the mounted
        files do not name it at all."""
        config = self._config(alpha="/a.lancedb", beta="/b.lancedb")

        text = create_analysis(config=config).get_instructions()

        assert "Collection:" in text  # analysis_search results
        assert "source" in text  # in-code search / list_documents
        assert "metadata.json" in text  # the mounted files, which lack it
        assert "list_documents" in text  # how to map ids to collections

    def test_a_run_narrowed_to_one_collection_drops_the_note(self):
        """A question narrows the conversation, so a capability over a set can
        still read one collection."""
        config = self._config(alpha="/a.lancedb", beta="/b.lancedb")

        rag = create_rag(config=config)
        rag.state = RAGState(sources=["alpha"])
        analysis = create_analysis(config=config)
        analysis.state = AnalysisState(sources=["alpha"])

        assert "Collection:" not in rag.get_instructions()
        assert "Collection:" not in analysis.get_instructions()

    def test_a_run_narrowed_to_two_collections_keeps_the_note(self):
        config = self._config(alpha="/a.lancedb", beta="/b.lancedb", gamma="/c.lancedb")

        capability = create_rag(config=config)
        capability.state = RAGState(sources=["alpha", "beta"])

        assert "Collection:" in capability.get_instructions()

    def test_an_unnarrowed_run_follows_the_lent_client(self):
        config = self._config(alpha="/a.lancedb", beta="/b.lancedb")
        one = create_rag(config=config, rag=self._client({}))
        one.state = RAGState()
        covering = create_rag(
            config=config, rag=self._client({"alpha": "/a", "beta": "/b"})
        )
        covering.state = RAGState()

        assert "Collection:" not in one.get_instructions()
        assert "Collection:" in covering.get_instructions()

    def test_the_note_follows_the_preamble_and_the_base(self):
        config = self._config(alpha="/a.lancedb", beta="/b.lancedb")
        config.prompts.domain_preamble = "PREAMBLE"

        text = create_rag(config=config).get_instructions()

        assert text.index("PREAMBLE") < text.index("# RAG") < text.index("Collection:")
