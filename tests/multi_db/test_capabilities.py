"""Asking and analyzing across the databases a question covers."""

import pytest

from haiku.rag.capabilities.rag import RAGState, create_capability
from haiku.rag.client import HaikuRAG
from haiku.rag.sandbox import AnalysisContext, Sandbox
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

            formatted = await capability._search("cats", limit=10)

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

            formatted = await capability._search("cats", limit=10)

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
            formatted = await run._search("cats", limit=10)
        finally:
            await run._close()

        assert isinstance(formatted, str)
        assert "alpha document" in formatted
        assert "beta document" in formatted

    @pytest.mark.asyncio
    @pytest.mark.vcr()
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

            formatted = await capability._search("cats", limit=10)
            sandbox = await capability._ensure_sandbox()
            await capability._close()

        assert isinstance(formatted, str)
        assert "alpha document" in formatted
        assert "beta document" not in formatted
        assert sandbox._context.sources == ["alpha"]


class TestDatabaseIdentityForTheModel:
    def test_a_result_names_its_database(self):
        """The model has to attribute and compare evidence by database while it
        composes the answer, not only afterwards through the citations."""
        result = SearchResult(content="body", score=0.9, source="alpha", chunk_id="c1")

        assert "Database: alpha" in result.format_for_agent()

    def test_an_unnamed_database_is_not_mentioned(self):
        """A single unnamed database renders as it always has."""
        result = SearchResult(content="body", score=0.9, chunk_id="c1")

        assert "Database" not in result.format_for_agent()

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_in_code_search_names_the_database(self, tmp_path):

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
