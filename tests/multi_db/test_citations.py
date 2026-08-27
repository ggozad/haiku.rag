"""Citing evidence drawn from several databases."""

import pytest
from pydantic_ai import ModelRetry

from haiku.rag.capabilities.rag import RAGState, create_capability
from haiku.rag.client import HaikuRAG
from haiku.rag.store.exceptions import (
    AmbiguousCitationError,
)
from haiku.rag.store.models import SearchResult
from haiku.rag.store.models.citation import Citation, resolve_citations
from tests.multi_db.helpers import (
    _config,
    _seed,
)


class TestSharedChunkIds:
    """A database copied from another holds the same chunk ids."""

    @pytest.mark.asyncio
    async def test_a_shared_id_does_not_confuse_the_fused_order(self, tmp_path):
        """Arrival order breaks score ties, so it has to tell two databases'
        identically-numbered chunks apart."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one about cats"])
        await _seed(config, "beta", ["beta one about cats"])
        fused = [
            SearchResult(content="a0", score=0.5, chunk_id="a0", source="alpha"),
            SearchResult(content="beta", score=0.5, chunk_id="shared", source="beta"),
            SearchResult(content="alpha", score=0.5, chunk_id="shared", source="alpha"),
        ]

        async with HaikuRAG(config=config) as rag:
            expanded = await rag.expand_context(fused)

        assert [(r.source, r.chunk_id) for r in expanded] == [
            (r.source, r.chunk_id) for r in fused
        ]

    def test_a_shared_id_cannot_be_cited(self):
        """A citation records the id alone, so resolving one held by two
        databases would attribute the answer to whichever came last."""
        results = [
            SearchResult(
                content="alpha body",
                score=0.9,
                source="alpha",
                chunk_id="c1",
                document_id="d1",
                document_uri="test://alpha/one",
            ),
            SearchResult(
                content="beta body",
                score=0.8,
                source="beta",
                chunk_id="c1",
                document_id="d1",
                document_uri="test://beta/one",
            ),
        ]

        with pytest.raises(AmbiguousCitationError, match="c1"):
            resolve_citations(["c1"], results)

    def test_a_repeated_id_from_one_database_still_collapses(self):
        """One database cannot hold two chunks under one id, so seeing it twice
        is the same chunk seen twice, and it resolves rather than raising. Which
        copy supplies the content is `resolve_citations`' own rule, pinned in
        `tests/store/test_citation.py`."""
        results = [
            SearchResult(
                content="first",
                score=0.9,
                source="alpha",
                chunk_id="c1",
                document_id="d1",
                document_uri="test://alpha/one",
            ),
            SearchResult(
                content="second",
                score=0.8,
                source="alpha",
                chunk_id="c1",
                document_id="d1",
                document_uri="test://alpha/one",
            ),
        ]

        [citation] = resolve_citations(["c1"], results)

        assert citation.chunk_id == "c1"
        assert citation.source == "alpha"

    def test_only_a_cited_id_has_to_be_unambiguous(self):
        """An id the answer never cites attributes nothing."""
        shared = [
            SearchResult(
                content=f"{name} body",
                score=0.9,
                source=name,
                chunk_id="c1",
                document_id="d1",
                document_uri=f"test://{name}/one",
            )
            for name in ("alpha", "beta")
        ]
        own = SearchResult(
            content="alpha only",
            score=0.7,
            source="alpha",
            chunk_id="c2",
            document_id="d2",
            document_uri="test://alpha/two",
        )

        [citation] = resolve_citations(["c2"], [*shared, own])

        assert citation.source == "alpha"

    @pytest.mark.asyncio
    async def test_an_unsearched_shared_id_is_refused_by_the_fallback(self, tmp_path):
        """The direct lookup is the only place a collision shows for an id no
        search returned, so it has to ask every database rather than take the
        first that answers."""
        import shutil

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(
            config, "alpha", ["alpha document about cats", "alpha on aardvarks"]
        )
        shutil.copytree(tmp_path / "alpha.lancedb", tmp_path / "beta.lancedb")

        async with HaikuRAG(config=config) as rag:
            alpha = (await rag.clients_for(["alpha"]))[0]
            chunks = await alpha.chunk_repository.list_all()
            [aardvark] = [c for c in chunks if "aardvark" in c.content]
            assert aardvark.id is not None

            capability = create_capability(config=config, rag=rag, defer_loading=False)
            capability.state = RAGState()

            # No search ran, so the id can only resolve through the fallback.
            with pytest.raises(ModelRetry, match="more than one database"):
                await capability._cite([aardvark.id])

    @pytest.mark.asyncio
    async def test_an_unsearched_id_in_one_database_still_resolves(self, tmp_path):
        """The refusal is for a collision, not for looking through several
        databases: an id only one of them holds still resolves."""
        from tests.capabilities.test_capabilities import Deps, make_context

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(
            config, "alpha", ["alpha document about cats", "alpha on aardvarks"]
        )
        await _seed(config, "beta", ["beta document about dogs"])

        async with HaikuRAG(config=config) as rag:
            alpha = (await rag.clients_for(["alpha"]))[0]
            chunks = await alpha.chunk_repository.list_all()
            [aardvark] = [c for c in chunks if "aardvark" in c.content]
            assert aardvark.id is not None

            capability = create_capability(config=config, rag=rag, defer_loading=False)
            run = await capability.for_run(make_context(Deps()))

            await run._cite([aardvark.id])

        assert run.state is not None
        [citation] = list(run.state.citation_index.values())
        assert citation.source == "alpha"

    @pytest.mark.asyncio
    async def test_cite_asks_for_other_evidence(self, tmp_path):
        capability = create_capability(
            config=_config(tmp_path, ["alpha", "beta"]), defer_loading=False
        )
        capability.state = RAGState(
            searches={
                "cats": [
                    SearchResult(
                        content=f"{name} body",
                        score=0.9,
                        source=name,
                        chunk_id="c1",
                        document_id="d1",
                        document_uri=f"test://{name}/one",
                    )
                    for name in ("alpha", "beta")
                ]
            }
        )

        with pytest.raises(ModelRetry, match="appears once"):
            await capability._cite(["c1"])

    @pytest.mark.asyncio
    async def test_cite_refuses_an_id_already_cited_from_another_database(
        self, tmp_path
    ):
        """The citation index outlives the question, so the collision can arrive
        a turn later than the search that would have shown it."""
        capability = create_capability(
            config=_config(tmp_path, ["alpha", "beta"]), defer_loading=False
        )
        capability.state = RAGState(
            citation_index={
                "c1": Citation(
                    document_id="d1",
                    source="alpha",
                    chunk_id="c1",
                    document_uri="test://alpha/one",
                    content="alpha body",
                )
            },
            searches={
                "cats": [
                    SearchResult(
                        content="beta body",
                        score=0.9,
                        source="beta",
                        chunk_id="c1",
                        document_id="d1",
                        document_uri="test://beta/one",
                    )
                ]
            },
        )

        with pytest.raises(ModelRetry, match="another database"):
            await capability._cite(["c1"])

    @pytest.mark.asyncio
    async def test_one_retrieved_copy_of_a_shared_id_cites_where_it_came_from(
        self, tmp_path
    ):
        """Rejection is about evidence the run holds, not about what the other
        databases contain. A copy the search never returned did not ground the
        answer, so the retrieved one is the citation and its database is not a
        guess. A later question that does retrieve the twin is refused by the
        citation index."""
        from tests.capabilities.test_capabilities import Deps, make_context

        config = _config(tmp_path, ["alpha", "beta"])
        capability = create_capability(config=config, defer_loading=False)
        deps = Deps(state={"rag": RAGState().model_dump(mode="json")})
        run = await capability.for_run(make_context(deps))
        assert run.state is not None
        # What a run holds when fusion returned one copy of a shared id and
        # truncated the other: the run never saw the twin.
        run.state.searches["cats"] = [
            SearchResult(
                content="alpha body",
                score=0.9,
                source="alpha",
                chunk_id="c1",
                document_id="d1",
                document_uri="test://alpha/one",
            )
        ]

        await run._cite(["c1"])

        [cited] = run.state.citation_index.values()
        assert cited.source == "alpha"
        assert cited.content == "alpha body"


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

    def test_a_result_without_an_id_is_skipped(self):
        """A result built by hand carries no chunk id, so nothing can cite it
        and it takes part in no collision."""
        handmade = SearchResult(content="loose text", score=0.5)
        real = SearchResult(
            content="body",
            score=0.9,
            source="alpha",
            chunk_id="c1",
            document_id="d1",
            document_uri="test://alpha/one",
        )

        [citation] = resolve_citations(["c1"], [handmade, real])

        assert citation.chunk_id == "c1"

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

            with pytest.raises(ModelRetry, match="None of the supplied chunk_ids"):
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

            with pytest.raises(ModelRetry, match="None of the supplied chunk_ids"):
                await run._cite([chunk.id])
