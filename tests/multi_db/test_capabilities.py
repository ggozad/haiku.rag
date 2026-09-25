"""Asking across the databases a question covers."""

from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import Agent, ToolFailed
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from haiku.rag.capabilities._tools import search_corpus
from haiku.rag.capabilities.rag import RAGState, create_capability
from haiku.rag.client import HaikuRAG
from haiku.rag.client.session import FederatedSession
from haiku.rag.config.models import AppConfig
from haiku.rag.sandbox import AnalysisContext, Sandbox
from haiku.rag.store.exceptions import AmbiguousDatabaseError, UnknownDatabaseError
from haiku.rag.store.models import SearchResult
from tests.multi_db.helpers import (
    _config,
    _seed,
)


class TestAskAcrossDatabases:
    @pytest.mark.vcr()
    async def test_the_capability_searches_the_selected_databases(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            capability = create_capability(config=config, rag=rag)
            capability.state = RAGState(sources=["alpha"])

            formatted = await capability._search("cats", 10, 1)

        assert isinstance(formatted, str)
        assert "alpha" in formatted
        assert "beta document" not in formatted

    @pytest.mark.vcr()
    async def test_searching_all_databases_reaches_both(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            capability = create_capability(config=config, rag=rag)
            capability.state = RAGState()

            formatted = await capability._search("cats", 10, 1)

        assert isinstance(formatted, str)
        assert "alpha document" in formatted
        assert "beta document" in formatted


class TestStandaloneCapabilities:
    """A capability nobody hands a client opens its own. It has to reach the
    configured set, or a host that only registers capabilities gets one
    database while the configuration names several."""

    @pytest.mark.vcr()
    async def test_a_rag_capability_opens_the_configured_set(self, tmp_path):
        from tests.capabilities.test_capabilities import Deps, make_context

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        capability = create_capability(config=config)
        assert capability.scope.names == ("alpha", "beta")
        run = await capability.for_run(make_context(Deps()))
        try:
            formatted = await run._search("cats", 10, 1)
        finally:
            await run._close()

        assert isinstance(formatted, str)
        assert "alpha document" in formatted
        assert "beta document" in formatted

    async def test_the_capability_mounts_the_configured_set(self, tmp_path):
        from tests.capabilities.test_capabilities import Deps, make_context

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        capability = create_capability(config=config)
        run = await capability.for_run(make_context(Deps()))
        try:
            sandbox = await run._ensure_sandbox()
            docs, owners = await sandbox._documents()
        finally:
            await run._close()

        assert len(docs) == 2
        assert {owner.source for owner in owners.values()} == {"alpha", "beta"}

    async def test_a_single_configured_database_is_still_opened(self, tmp_path):
        """One named database is a set of one, not a path to guess."""
        config = _config(tmp_path, ["alpha"])
        await _seed(config, "alpha", ["alpha document about cats"])

        capability = create_capability(config=config)
        rag = await capability._ensure_rag()
        try:
            assert rag.source == "alpha"
        finally:
            await capability._close()


class TestTheSandboxAcrossDatabases:
    @pytest.mark.vcr()
    async def test_the_sandbox_is_scoped_by_the_search_selection(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            capability = create_capability(config=config, rag=rag)
            capability.state = RAGState(sources=["alpha"])

            formatted = await capability._search("cats", 10, 1)
            sandbox = await capability._ensure_sandbox()
            await capability._close()

        assert isinstance(formatted, str)
        assert "alpha document" in formatted
        assert "beta document" not in formatted
        assert sandbox._context.sources == ["alpha"]


class TestScopingACapabilityToASubset:
    """`sources` at construction narrows what the capability can reach."""

    async def test_a_scoped_capability_never_reaches_the_other_database(
        self, tmp_path, query_embedding
    ):
        from tests.capabilities.test_capabilities import Deps, make_context

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        capability = create_capability(config=config, sources=["alpha"])
        assert capability.scope.names == ("alpha",)
        run = await capability.for_run(make_context(Deps()))
        try:
            formatted = await run._search("cats", 10, 1)
        finally:
            await run._close()

        assert isinstance(formatted, str)
        assert "alpha document" in formatted
        assert "beta document" not in formatted

    async def test_a_scoped_capability_mounts_only_its_databases(self, tmp_path):
        from tests.capabilities.test_capabilities import Deps, make_context

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        capability = create_capability(config=config, sources=["alpha"])
        run = await capability.for_run(make_context(Deps()))
        try:
            sandbox = await run._ensure_sandbox()
            # One database is one connection, which leaves `owners` empty.
            docs, _ = await sandbox._documents()
        finally:
            await run._close()

        assert [doc.uri for doc in docs] == ["test://alpha/alpha document about cats"]

    async def test_a_question_naming_a_database_outside_the_scope_fails(
        self, tmp_path, query_embedding
    ):
        """State `sources` selects within the scope."""
        from tests.capabilities.test_capabilities import Deps, make_context

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        capability = create_capability(config=config, sources=["alpha"])
        deps = Deps(state={"rag": RAGState(sources=["beta"]).model_dump(mode="json")})
        run = await capability.for_run(make_context(deps))
        try:
            with pytest.raises(UnknownDatabaseError, match="beta"):
                await run._search("cats", 10, 1)
        finally:
            await run._close()

    def test_sources_beside_a_path_is_refused(self, tmp_path):
        with pytest.raises(AmbiguousDatabaseError, match="alpha"):
            create_capability(
                db_path=tmp_path / "kb.lancedb",
                config=AppConfig(),
                sources=["alpha"],
            )

    def test_selecting_no_database_is_refused(self, tmp_path):
        """Unlike state `sources=[]`, which selects nothing to search."""
        with pytest.raises(ValueError, match="pass None for all of them"):
            create_capability(config=_config(tmp_path, ["alpha", "beta"]), sources=[])

    async def test_sources_beside_a_lent_client_is_refused(self, tmp_path):
        """A lent client's coverage is what the capability reads."""
        config = _config(tmp_path, ["alpha", "beta"])

        async with HaikuRAG(config=config) as rag:
            with pytest.raises(AmbiguousDatabaseError, match="client"):
                create_capability(config=config, rag=rag, sources=["alpha"])


class TestCollectionIdentityForTheModel:
    """A collection is named to the model only when the search spans more than
    one, and the caller decides that: a result cannot tell from its own fields
    whether anything else was searched."""

    def test_a_result_names_its_collection_when_asked(self):
        """The model has to attribute and compare evidence by collection while
        it composes the answer, not only afterwards through the citations."""
        result = SearchResult(content="body", score=0.9, source="alpha", chunk_id="c1")

        assert "Collection: alpha" in result.format_for_agent(include_collection=True)

    def test_a_named_collection_is_silent_unless_asked(self):
        """One collection has nothing to distinguish, named or not."""
        result = SearchResult(content="body", score=0.9, source="alpha", chunk_id="c1")

        assert "Collection" not in result.format_for_agent()

    def test_a_hand_built_result_without_a_source_is_never_labelled(self):
        """A result built by hand carries no source to name, whatever the
        caller asked for."""
        result = SearchResult(content="body", score=0.9, chunk_id="c1")

        assert "Collection" not in result.format_for_agent(include_collection=True)

    @pytest.mark.vcr()
    async def test_in_code_search_names_the_collection(self, tmp_path):
        """The dictionaries code reads carry `source` whatever the formatted
        output renders, since grouping by it is computation."""

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            sandbox = Sandbox(
                db_path=None,
                config=config,
                context=AnalysisContext(),
                rag=rag,
            )
            try:
                result = await sandbox.execute(
                    "rows = await search('cats', limit=10)\n"
                    "print(sorted(r['source'] for r in rows))\n"
                    "docs = await list_documents()\n"
                    "print(sorted(d['source'] for d in docs))"
                )
            finally:
                await sandbox.close()

        assert result.success, result.stderr
        assert "['alpha', 'beta']" in result.stdout
        assert result.stdout.count("['alpha', 'beta']") == 2


class TestNamingDatabasesBeforeTheModelRuns:
    """A name is checked at the boundary. Discovering it from a failed search
    spends model requests, and a run can answer without reaching one."""

    async def test_ask_refuses_an_unknown_source_before_the_model(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            with pytest.raises(UnknownDatabaseError, match="typo"):
                await rag.ask("what about cats?", sources=["typo"])

    async def test_checking_a_name_opens_nothing(self, tmp_path):
        """Validating a name reads the configured set; no database opens for
        it."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            assert isinstance(rag._session, FederatedSession)

            rag._require_known_sources(None)
            rag._require_known_sources(["alpha"])
            rag._require_known_sources([])
            with pytest.raises(UnknownDatabaseError, match="typo"):
                rag._require_known_sources(["alpha", "typo"])

            assert rag._session._sessions == {}


class TestLendingANamedClient:
    async def test_a_lent_named_client_names_the_citation(self, tmp_path):
        """What a citation records is the lent client's database, not the scope
        the capability was constructed with. That chat lends its client is
        `TestLendingTheClient` in `tests/chat/test_chat_app.py`."""
        from tests.capabilities.test_capabilities import Deps, make_context

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        # What `run_chat` builds: the capability's own scope is the set, and
        # the lent client is what narrows it.
        capability = create_capability(config=config)

        async with HaikuRAG(config=config, sources=["alpha"]) as client:
            # What `ChatApp.on_mount` does.
            capability.borrowed_rag = client
            deps = Deps(state={"rag": RAGState().model_dump(mode="json")})
            run = await capability.for_run(make_context(deps))
            assert run.state is not None

            # A borrowed client overrides the capability's configured placement.
            assert await run._ensure_rag() is client

            run.state.searches["cats"] = await client.search("cats", search_type="fts")
            [result] = run.state.searches["cats"]
            assert result.chunk_id is not None
            await run._cite([result.chunk_id])

            [citation] = run.state.citation_index.values()

        assert result.source == "alpha"
        assert citation.source == "alpha"


class TestWhenTheModelIsToldTheCollection:
    """The line follows the run, which the caller states: a result cannot tell
    from its own fields what else the run could reach."""

    async def test_the_label_is_the_callers_decision(self, tmp_path, monkeypatch):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        only_alpha = [
            SearchResult(content="body", score=0.9, source="alpha", chunk_id="c1")
        ]

        async with HaikuRAG(config=config) as rag:
            monkeypatch.setattr(rag, "search", AsyncMock(return_value=only_alpha))

            labelled, _, _ = await search_corpus(
                rag, "cats", sources=["alpha"], include_collection=True
            )
            plain, _, _ = await search_corpus(rag, "cats", include_collection=False)

        assert "Collection: alpha" in labelled
        assert "Collection" not in plain


class TestNarrowingASearch:
    """`sources` on one search selects within the run, and the run's own
    selection is the ceiling: a lent client may cover more."""

    async def test_sources_narrows_one_search(self, tmp_path, query_embedding):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            capability = create_capability(config=config, rag=rag)
            capability.state = RAGState()

            narrowed = await capability._search("cats", 10, 1, sources=["beta"])
            broad = await capability._search("cats", 10, 2)

        assert isinstance(narrowed, str) and isinstance(broad, str)
        assert "beta document" in narrowed
        assert "alpha document" not in narrowed
        assert "alpha document" in broad and "beta document" in broad

    async def test_an_empty_selection_searches_nothing(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            capability = create_capability(config=config, rag=rag)
            capability.state = RAGState()

            assert await capability._search("cats", 10, 1, sources=[]) == (
                "No results found."
            )

    async def test_a_name_outside_the_run_fails_naming_the_run(self, tmp_path):
        """Checked before anything opens: a wrong name costs no database."""
        from tests.capabilities.test_capabilities import Deps, make_context

        capability = create_capability(config=_config(tmp_path, ["alpha", "beta"]))
        run = await capability.for_run(make_context(Deps()))

        with pytest.raises(ToolFailed) as raised:
            await run._search("cats", 10, 1, sources=["alpha", "typo"])

        assert raised.value.message == (
            "Unknown collection(s): typo. This run covers: alpha, beta."
        )
        assert run.rag is None

    async def test_the_run_selection_is_the_ceiling_not_the_client(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            capability = create_capability(config=config, rag=rag)
            capability.state = RAGState(sources=["alpha"])

            with pytest.raises(ToolFailed) as raised:
                await capability._search("cats", 10, 1, sources=["beta"])

        assert raised.value.message == (
            "Unknown collection(s): beta. This run covers: alpha."
        )

    async def test_a_narrowed_search_in_a_spanning_run_names_the_collection(
        self, tmp_path, query_embedding
    ):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            capability = create_capability(config=config, rag=rag)
            capability.state = RAGState()

            narrowed = await capability._search("cats", 10, 1, sources=["alpha"])

        assert isinstance(narrowed, str)
        assert "Collection: alpha" in narrowed

    async def test_a_run_of_one_collection_names_nothing(
        self, tmp_path, query_embedding
    ):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            capability = create_capability(config=config, rag=rag)
            capability.state = RAGState(sources=["alpha"])

            formatted = await capability._search("cats", 10, 1, sources=["alpha"])

        assert isinstance(formatted, str)
        assert "alpha document" in formatted
        assert "Collection" not in formatted


class TestTheRunsCollections:
    """What the model may select among: the question's `sources`, else the
    lent client's coverage, else the scope."""

    def test_state_sources_are_the_run(self, tmp_path):
        capability = create_capability(config=_config(tmp_path, ["alpha", "beta"]))
        capability.state = RAGState(sources=["beta", "beta"])

        assert capability.collections == ("beta",)
        assert not capability.spans_collections

    def test_a_lent_client_is_the_run(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        capability = create_capability(config=config)
        capability.borrowed_rag = HaikuRAG(config=config, sources=["alpha"])
        capability.state = RAGState()

        assert capability.collections == ("alpha",)
        assert not capability.spans_collections

    def test_the_scope_is_the_run_by_default(self, tmp_path):
        capability = create_capability(config=_config(tmp_path, ["alpha", "beta"]))

        assert capability.collections == ("alpha", "beta")
        assert capability.spans_collections


class TestTheSearchToolSchema:
    """`sources` is offered only where there is something to select among, so
    a run over one collection keeps the `(query, limit)` contract."""

    async def _search_tool(self, capability, deps):
        from tests.capabilities.test_capabilities import make_context

        ctx = make_context(deps)
        run = await capability.for_run(ctx)
        try:
            tools = await run.get_toolset().get_tools(ctx)
            # The agent stamps the capability on its tools before `prepare_tools`.
            defs = [replace(t.tool_def, capability_id=run.id) for t in tools.values()]
            prepared = await run.prepare_tools(ctx, defs)
        finally:
            await run._close()
        return next(tool for tool in prepared if tool.name == "search")

    async def test_a_spanning_run_offers_sources(self, tmp_path):
        from tests.capabilities.test_capabilities import Deps

        capability = create_capability(config=_config(tmp_path, ["alpha", "beta"]))
        tool = await self._search_tool(capability, Deps())

        properties = tool.parameters_json_schema["properties"]
        assert set(properties) == {"query", "limit", "sources"}
        assert "collection" in properties["sources"]["description"].lower()
        assert "sources" not in tool.parameters_json_schema.get("required", [])

    async def test_a_run_over_one_collection_keeps_two_parameters(self, tmp_path):
        from tests.capabilities.test_capabilities import Deps

        capability = create_capability(config=_config(tmp_path, ["alpha"]))
        tool = await self._search_tool(capability, Deps())

        assert set(tool.parameters_json_schema["properties"]) == {"query", "limit"}

    async def test_a_question_narrowed_to_one_collection_hides_sources(self, tmp_path):
        from tests.capabilities.test_capabilities import Deps

        capability = create_capability(config=_config(tmp_path, ["alpha", "beta"]))
        deps = Deps(state={"rag": RAGState(sources=["alpha"]).model_dump(mode="json")})
        tool = await self._search_tool(capability, deps)

        assert "sources" not in tool.parameters_json_schema["properties"]


class TestTheModelNarrowingASearch:
    """Through the agent: what the model is offered, and what it gets back."""

    async def test_the_model_narrows_a_search_and_reads_the_collection(
        self, tmp_path, query_embedding
    ):
        from tests.capabilities.test_capabilities import Deps

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])
        seen: dict = {}

        def model(messages, info: AgentInfo) -> ModelResponse:
            search = next(t for t in info.function_tools if t.name == "search")
            seen["properties"] = set(search.parameters_json_schema["properties"])
            if len(messages) == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart("search", {"query": "cats", "sources": ["beta"]})
                    ]
                )
            seen["returned"] = messages[-1].parts[0].content
            return ModelResponse(parts=[TextPart("done")])

        agent = Agent(
            FunctionModel(model),
            deps_type=Deps,
            capabilities=[create_capability(config=config)],
        )
        result = await agent.run("cats", deps=Deps())

        assert result.output == "done"
        assert seen["properties"] == {"query", "limit", "sources"}
        assert "beta document" in seen["returned"]
        assert "alpha document" not in seen["returned"]
        assert "Collection: beta" in seen["returned"]

    async def test_over_one_collection_the_model_is_offered_no_sources(self, tmp_path):
        from tests.capabilities.test_capabilities import Deps

        config = _config(tmp_path, ["alpha"])
        await _seed(config, "alpha", ["alpha document about cats"])
        seen: dict = {}

        def model(messages, info: AgentInfo) -> ModelResponse:
            search = next(t for t in info.function_tools if t.name == "search")
            seen["properties"] = set(search.parameters_json_schema["properties"])
            return ModelResponse(parts=[TextPart("done")])

        agent = Agent(
            FunctionModel(model),
            deps_type=Deps,
            capabilities=[create_capability(config=config)],
        )
        await agent.run("cats", deps=Deps())

        assert seen["properties"] == {"query", "limit"}


class TestActionableFailures:
    async def test_a_migration_error_survives_being_named(self, tmp_path, temp_db_path):
        """The remedy is the whole value of the message, and it names no location,
        so it is not replaced by the database's name."""
        from haiku.rag.store.exceptions import MigrationRequiredError

        config = _config(tmp_path, ["alpha"])
        await _seed(config, "alpha", ["alpha document about cats"])

        async with HaikuRAG(config=config, sources=["alpha"]) as rag:
            await rag.store.set_haiku_version("0.20.0")

        with pytest.raises(MigrationRequiredError) as raised:
            async with HaikuRAG(config=config, sources=["alpha"]):
                pass

        # Both halves: which database failed, and what to run about it.
        assert "haiku-rag migrate" in str(raised.value)
        assert "alpha" in str(raised.value)
        assert str(tmp_path) not in str(raised.value)
