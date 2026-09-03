"""Asking and analyzing across the databases a question covers."""

from unittest.mock import AsyncMock

import pytest

from haiku.rag.capabilities._tools import search_corpus
from haiku.rag.capabilities.rag import RAGState, create_capability
from haiku.rag.client import HaikuRAG
from haiku.rag.client.scope import DatabaseScope
from haiku.rag.client.session import FederatedSession
from haiku.rag.sandbox import AnalysisContext, Sandbox
from haiku.rag.store.exceptions import UnknownDatabaseError
from haiku.rag.store.models import SearchResult
from tests.multi_db.helpers import (
    _config,
    _seed,
)


class TestAskAcrossDatabases:
    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_the_capability_searches_the_selected_databases(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            capability = create_capability(config=config, rag=rag, defer_loading=False)
            capability.state = RAGState(sources=["alpha"])

            formatted = await capability._search("cats", 10, 1)

        assert isinstance(formatted, str)
        assert "alpha" in formatted
        assert "beta document" not in formatted

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_searching_all_databases_reaches_both(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            capability = create_capability(config=config, rag=rag, defer_loading=False)
            capability.state = RAGState()

            formatted = await capability._search("cats", 10, 1)

        assert isinstance(formatted, str)
        assert "alpha document" in formatted
        assert "beta document" in formatted


class TestStandaloneCapabilities:
    """A capability nobody hands a client opens its own. It has to reach the
    configured set, or a host that only registers capabilities gets one
    database while the configuration names several."""

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_a_rag_capability_opens_the_configured_set(self, tmp_path):
        from tests.capabilities.test_capabilities import Deps, make_context

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        capability = create_capability(config=config, defer_loading=False)
        assert capability.scope.names == ("alpha", "beta")
        run = await capability.for_run(make_context(Deps()))
        try:
            formatted = await run._search("cats", 10, 1)
        finally:
            await run._close()

        assert isinstance(formatted, str)
        assert "alpha document" in formatted
        assert "beta document" in formatted

    @pytest.mark.asyncio
    async def test_an_analysis_capability_mounts_the_configured_set(self, tmp_path):
        from haiku.rag.capabilities.analysis import (
            create_capability as create_analysis,
        )
        from tests.capabilities.test_capabilities import Deps, make_context

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        capability = create_analysis(config=config, defer_loading=False)
        run = await capability.for_run(make_context(Deps()))
        try:
            sandbox = await run._ensure_sandbox()
            docs, owners = await sandbox._documents()
        finally:
            await run._close()

        assert len(docs) == 2
        assert {owner.source for owner in owners.values()} == {"alpha", "beta"}

    @pytest.mark.asyncio
    async def test_a_single_configured_database_is_still_opened(self, tmp_path):
        """One named database is a set of one, not a path to guess."""
        config = _config(tmp_path, ["alpha"])
        await _seed(config, "alpha", ["alpha document about cats"])

        capability = create_capability(config=config, defer_loading=False)
        rag = await capability._ensure_rag()
        try:
            assert rag.source == "alpha"
        finally:
            await capability._close()


class TestAnalyzeAcrossDatabases:
    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_the_capability_searches_the_selected_databases(self, tmp_path):
        """`analysis_search` is the same tool as the RAG one, and the sandbox is
        scoped by the same selection."""
        from haiku.rag.capabilities.analysis import AnalysisState
        from haiku.rag.capabilities.analysis import (
            create_capability as create_analysis,
        )

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            capability = create_analysis(config=config, rag=rag, defer_loading=False)
            capability.state = AnalysisState(sources=["alpha"])

            formatted = await capability._search("cats", 10, 1)
            sandbox = await capability._ensure_sandbox()
            await capability._close()

        assert isinstance(formatted, str)
        assert "alpha document" in formatted
        assert "beta document" not in formatted
        assert sandbox._context.sources == ["alpha"]


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

    def test_an_unnamed_collection_is_never_mentioned(self):
        """Nothing to name, whatever the caller asked for."""
        result = SearchResult(content="body", score=0.9, chunk_id="c1")

        assert "Collection" not in result.format_for_agent(include_collection=True)

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_in_code_search_names_the_collection(self, tmp_path):
        """The dictionaries analysis code reads carry `source` whatever the
        formatted output renders, since grouping by it is computation."""

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

    @pytest.mark.asyncio
    async def test_ask_refuses_an_unknown_source_before_the_model(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            with pytest.raises(UnknownDatabaseError, match="typo"):
                await rag.ask("what about cats?", sources=["typo"])

    @pytest.mark.asyncio
    async def test_analyze_refuses_an_unknown_source_before_the_model(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])

        async with HaikuRAG(config=config) as rag:
            with pytest.raises(UnknownDatabaseError, match="typo"):
                await rag.analyze("how many?", sources=["typo"])

    @pytest.mark.asyncio
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
    @pytest.mark.asyncio
    async def test_a_lent_named_client_names_the_citation(self, tmp_path):
        """What a citation records is the lent client's database, not the scope
        the capability was constructed with. That chat lends its client is
        `TestLendingTheClient` in `tests/chat/test_chat_app.py`."""
        from tests.capabilities.test_capabilities import Deps, make_context

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        # `run_chat` derives these for a single-database scope.
        scope = DatabaseScope.resolve(config, database_name="alpha")
        one_config, one_path = scope.databases[0].connection(config)
        capability = create_capability(
            db_path=one_path, config=one_config, defer_loading=False
        )

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
    """The line is decided by what the search spans, not by whether a name
    exists: one collection has nothing to distinguish."""

    @pytest.mark.asyncio
    async def test_a_search_spanning_a_set_names_every_result(
        self, tmp_path, monkeypatch
    ):
        """Named from the selection, so a result is named even when every hit
        came back from one collection: the search could have drawn on both."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        only_alpha = [
            SearchResult(content="body", score=0.9, source="alpha", chunk_id="c1")
        ]

        async with HaikuRAG(config=config) as rag:
            monkeypatch.setattr(rag, "search", AsyncMock(return_value=only_alpha))

            spanning, _, _, spans = await search_corpus(rag, "cats")
            narrowed, _, _, narrows = await search_corpus(
                rag, "cats", sources=["alpha"]
            )

        assert "Collection: alpha" in spanning
        assert "Collection" not in narrowed
        # Images travel beside the results and are labelled the same way.
        assert (spans, narrows) == (True, False)


class TestActionableFailures:
    @pytest.mark.asyncio
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
