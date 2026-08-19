import pytest
from pydantic_ai import ModelRetry

from haiku.rag.capabilities.rag import RAGState, create_capability
from haiku.rag.client import HaikuRAG
from haiku.rag.store.models import SearchResult
from haiku.rag.store.models.citation import resolve_citations
from tests.test_multi_db import _config, _seed


class TestExpansionRouting:
    @pytest.mark.asyncio
    async def test_expansion_routes_each_result_to_its_database(self, tmp_path):
        """A federating client has no repositories of its own, so expansion has
        to go through the database each result came from."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            results = await rag.search("cats", search_type="fts", limit=10)
            expanded = await rag.expand_context(results)

        assert {r.source for r in expanded} == {"alpha", "beta"}
        for r in expanded:
            assert r.source is not None
            assert r.source in r.content


class TestCitationSource:
    def test_a_citation_carries_the_result_source(self):
        result = SearchResult(
            content="body",
            score=0.9,
            source="alpha",
            chunk_id="c1",
            document_id="d1",
            document_uri="test://alpha/one",
        )

        [citation] = resolve_citations(["c1"], [result])

        assert citation.source == "alpha"

    def test_a_single_database_citation_has_no_source(self):
        result = SearchResult(
            content="body",
            score=0.9,
            chunk_id="c1",
            document_id="d1",
            document_uri="test://one",
        )

        [citation] = resolve_citations(["c1"], [result])

        assert citation.source is None


class TestAskAcrossDatabases:
    @pytest.mark.asyncio
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


class TestCiteFallback:
    @pytest.mark.asyncio
    async def test_an_id_from_a_selected_database_resolves_with_its_source(
        self, tmp_path
    ):
        """The fallback exists for a real id this run's searches did not return.
        Across databases it looks through the selected ones and records which
        held it."""
        from tests.capabilities.test_capabilities import Deps, make_context

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(
            config, "alpha", ["alpha document about cats", "alpha on aardvarks"]
        )
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            alpha = (await rag.clients_for(["alpha"]))[0]
            chunks = await alpha.chunk_repository.list_all()
            [aardvark] = [c for c in chunks if "aardvark" in c.content]
            assert aardvark.id is not None

            capability = create_capability(config=config, rag=rag, defer_loading=False)
            deps = Deps(
                state={"rag": RAGState(sources=["alpha"]).model_dump(mode="json")}
            )
            run = await capability.for_run(make_context(deps))
            # The search returns the cats chunk, never the aardvark one.
            await run._search("cats", limit=10)

            await run._cite([aardvark.id])

        assert run.state is not None
        [citation] = list(run.state.citation_index.values())
        assert citation.chunk_id == aardvark.id
        assert citation.source == "alpha"

    @pytest.mark.asyncio
    async def test_an_id_outside_the_selected_databases_does_not_resolve(
        self, tmp_path
    ):
        """A question scoped to one database must not produce a citation from
        another: the fallback looks only where the question looked."""
        from tests.capabilities.test_capabilities import Deps, make_context

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about dogs"])

        async with HaikuRAG(config=config) as rag:
            beta = (await rag.clients_for(["beta"]))[0]
            [outside] = await beta.chunk_repository.list_all(limit=1)
            assert outside.id is not None

            capability = create_capability(config=config, rag=rag, defer_loading=False)
            deps = Deps(
                state={"rag": RAGState(sources=["alpha"]).model_dump(mode="json")}
            )
            run = await capability.for_run(make_context(deps))
            await run._search("cats", limit=10)

            with pytest.raises(ModelRetry):
                await run._cite([outside.id])


class TestFederatedEdges:
    @pytest.mark.asyncio
    async def test_expansion_passes_through_results_without_a_source(self, tmp_path):
        """A caller can hand `expand_context` results it built itself. Those name
        no database, so there is nowhere to expand them from."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        handmade = SearchResult(content="handmade", score=0.4, doc_item_refs=[])

        async with HaikuRAG(config=config) as rag:
            found = await rag.search("cats", search_type="fts", limit=10)
            expanded = await rag.expand_context([*found, handmade])

        assert "handmade" in [r.content for r in expanded]
        scores = [r.score for r in expanded]
        assert scores == sorted(scores, reverse=True), "merged in score order"

    @pytest.mark.asyncio
    async def test_a_chunk_without_a_document_is_not_cited(self, tmp_path):
        """`Chunk.document_id` is optional, and a citation without a document has
        nothing to point at."""
        from unittest.mock import AsyncMock, patch

        from haiku.rag.capabilities.rag import RAGCapability
        from haiku.rag.store.models import Chunk
        from tests.capabilities.test_capabilities import Deps, make_context

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])

        orphan = AsyncMock()
        orphan._federated = {}
        orphan._source = None
        orphan.get_chunk_by_id.return_value = Chunk(
            id="orphan", document_id=None, content="no document"
        )

        capability = create_capability(config=config, defer_loading=False)
        run = await capability.for_run(make_context(Deps()))
        with patch.object(RAGCapability, "_ensure_rag", AsyncMock(return_value=orphan)):
            with pytest.raises(ModelRetry):
                await run._cite(["orphan"])
