import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from pydantic_ai import ModelRetry

from haiku.rag.capabilities.rag import RAGState, create_capability
from haiku.rag.client import HaikuRAG
from haiku.rag.config import get_config
from haiku.rag.store.models import Chunk, Document, DocumentItem, SearchResult
from haiku.rag.store.models.citation import resolve_citations
from tests.test_multi_db import _config, _seed


async def _seed_expandable(config, name, sentences):
    """One document whose chunk covers a single item, so expansion has
    neighbours to pull in and rebuilds the result rather than passing it
    through."""
    dim = get_config().embeddings.model.vector_dim
    doc = DoclingDocument(name=name)
    for sentence in sentences:
        doc.add_text(label=DocItemLabel.TEXT, text=sentence)
    async with HaikuRAG(config=config, create=True, sources=[name]) as rag:
        await rag.import_document(
            doc,
            [
                Chunk(
                    content=sentences[0],
                    embedding=[0.1] * dim,
                    order=0,
                    metadata={"doc_item_refs": ["#/texts/0"]},
                )
            ],
            uri=f"test://{name}/expandable",
        )


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

    @pytest.mark.asyncio
    async def test_an_expanded_result_keeps_its_source(self, tmp_path):
        """Expansion rebuilds the result, and the rebuilt one has to name the
        database it was expanded through."""
        config = _config(tmp_path, ["alpha"])
        await _seed_expandable(
            config, "alpha", ["cats sleep often", "cats also hunt", "cats purr"]
        )

        async with HaikuRAG(config=config) as rag:
            results = await rag.search("cats", search_type="fts", limit=10)
            expanded = await rag.expand_context(results)

        assert len(expanded) == 1
        assert "cats also hunt" in expanded[0].content, "expansion did not run"
        assert expanded[0].source == "alpha"

    @pytest.mark.asyncio
    async def test_a_federated_result_is_expanded_by_its_own_database(self, tmp_path):
        """Routing is not enough: each result has to come back carrying the
        neighbours of the database it was expanded through, and only those."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed_expandable(
            config, "alpha", ["cats sleep often", "alpha follows on"]
        )
        await _seed_expandable(config, "beta", ["cats also hunt", "beta follows on"])

        async with HaikuRAG(config=config) as rag:
            results = await rag.search("cats", search_type="fts", limit=10)
            expanded = await rag.expand_context(results)

        content = {r.source: r.content for r in expanded}
        assert "alpha follows on" in content["alpha"]
        assert "beta follows on" not in content["alpha"]
        assert "beta follows on" in content["beta"]

    @pytest.mark.asyncio
    async def test_expansion_keeps_tied_results_in_fused_order(self, tmp_path):
        """Fused scores tie often, so grouping by database must not reorder
        them: the tiebreak is the order they arrived in."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one about cats", "alpha two about cats"])
        await _seed(config, "beta", ["beta one about cats"])

        async with HaikuRAG(config=config) as rag:
            found = await rag.search("cats", search_type="fts", limit=10)
            by_source: dict[str, list[SearchResult]] = {}
            for result in found:
                by_source.setdefault(result.source or "", []).append(result)
            # Interleaved, so grouping by database is visible as a reordering.
            fused = [by_source["alpha"][0], by_source["beta"][0], by_source["alpha"][1]]
            for result in fused:
                result.score = 0.5

            expanded = await rag.expand_context(fused)

        assert [r.chunk_id for r in expanded] == [r.chunk_id for r in fused]


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


class TestCiteFallback:
    @pytest.mark.asyncio
    @pytest.mark.vcr()
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
    @pytest.mark.vcr()
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

    @pytest.mark.asyncio
    async def test_selecting_no_databases_cites_nothing(self, tmp_path):
        """`sources=[]` selected nothing, which is not the same as everything:
        the fallback must not go looking where the question never looked."""
        from tests.capabilities.test_capabilities import Deps, make_context

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            alpha = (await rag.clients_for(["alpha"]))[0]
            [chunk] = await alpha.chunk_repository.list_all(limit=1)
            assert chunk.id is not None

            capability = create_capability(config=config, rag=rag, defer_loading=False)
            deps = Deps(state={"rag": RAGState(sources=[]).model_dump(mode="json")})
            run = await capability.for_run(make_context(deps))

            with pytest.raises(ModelRetry):
                await run._cite([chunk.id])


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
        assert capability.db_path is None
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
        from haiku.rag.sandbox import AnalysisContext, Sandbox

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


class TestPictureDeduplication:
    """One picture yields two chunks — a text-embedded one and an image-embedded
    one — that collapse to the best. Two databases holding the same picture are
    two results, not a duplicate."""

    @staticmethod
    def _picture(source, score):
        return SearchResult(
            content="a figure",
            score=score,
            source=source,
            chunk_id=f"{source}-c",
            document_id="doc-1",
            doc_item_refs=["#/pictures/0"],
        )

    def test_the_same_picture_in_two_databases_survives(self):
        from haiku.rag.client.search import _dedup_picture_chunks

        kept = _dedup_picture_chunks(
            [self._picture("alpha", 0.9), self._picture("clone", 0.5)]
        )

        assert [r.source for r in kept] == ["alpha", "clone"]

    def test_duplicates_within_one_database_still_collapse(self):
        from haiku.rag.client.search import _dedup_picture_chunks

        lower = self._picture("alpha", 0.5)
        higher = self._picture("alpha", 0.9)

        kept = _dedup_picture_chunks([lower, higher])

        assert kept == [higher]


class TestPictureRouting:
    @pytest.mark.asyncio
    async def test_a_picture_is_fetched_from_the_database_that_holds_it(self, tmp_path):
        """A `self_ref` repeats across databases, so the citation's source is
        what decides where the bytes come from."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            beta = (await rag.clients_for(["beta"]))[0]
            [document] = await beta.document_repository.list_all(limit=1)
            assert document.id is not None
            await beta.document_item_repository.create_all(
                [
                    DocumentItem(
                        document_id=document.id,
                        self_ref="#/pictures/0",
                        position=99,
                        label="picture",
                        text="",
                        picture_data=b"beta-picture",
                    )
                ]
            )

            assert (
                await rag.get_picture_bytes(document.id, "#/pictures/0", "beta")
                == b"beta-picture"
            )
            assert (
                await rag.get_picture_bytes(document.id, "#/pictures/0", "alpha")
                is None
            )

    @pytest.mark.asyncio
    async def test_a_single_database_needs_no_source(self, temp_db_path):
        """One database is where the picture is, named or not."""
        async with HaikuRAG(temp_db_path, create=True) as rag:
            document = await rag.document_repository.create(
                Document(content="body", uri="test://one")
            )
            assert document.id is not None
            await rag.document_item_repository.create_all(
                [
                    DocumentItem(
                        document_id=document.id,
                        self_ref="#/pictures/0",
                        position=0,
                        label="picture",
                        text="",
                        picture_data=b"the-picture",
                    )
                ]
            )

            assert (
                await rag.get_picture_bytes(document.id, "#/pictures/0")
                == b"the-picture"
            )

    @pytest.mark.asyncio
    async def test_a_picture_lookup_without_a_source_is_refused(self, tmp_path):
        """Federating, nothing can say which database holds an unqualified
        reference."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])

        async with HaikuRAG(config=config) as rag:
            with pytest.raises(ValueError, match="source"):
                await rag.get_picture_bytes("doc-1", "#/pictures/0")


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
        from tests.capabilities.test_capabilities import (
            Deps,
            _single_database_client,
            make_context,
        )

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])

        orphan = _single_database_client()
        orphan.get_chunk_by_id.return_value = Chunk(
            id="orphan", document_id=None, content="no document"
        )

        capability = create_capability(config=config, defer_loading=False)
        run = await capability.for_run(make_context(Deps()))
        with patch.object(RAGCapability, "_ensure_rag", AsyncMock(return_value=orphan)):
            with pytest.raises(ModelRetry):
                await run._cite(["orphan"])
