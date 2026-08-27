"""Searching several databases and fusing what they return."""

import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from haiku.rag.client import HaikuRAG
from haiku.rag.client.session import FederatedSession
from haiku.rag.config import get_config
from haiku.rag.store.exceptions import (
    ConfigMismatchError,
    SourceUnavailableError,
    UnknownDatabaseError,
)
from haiku.rag.store.models import Chunk, DocumentItem
from tests.multi_db.helpers import (
    StubReranker,
    _config,
    _restore_embedder,
    _seed,
)


class TestFederatedSearch:
    @pytest.mark.asyncio
    async def test_results_carry_their_source(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            results = await rag.search("cats", limit=10, search_type="fts")

        assert {r.source for r in results} == {"alpha", "beta"}
        for r in results:
            assert r.source is not None
            assert r.source in r.content

    @pytest.mark.asyncio
    async def test_sources_selects_a_subset(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            results = await rag.search(
                "cats", limit=10, search_type="fts", sources=["alpha"]
            )

        assert {r.source for r in results} == {"alpha"}

    @pytest.mark.asyncio
    async def test_unknown_source_is_rejected(self, tmp_path):
        config = _config(tmp_path, ["alpha"])
        await _seed(config, "alpha", ["alpha document about cats"])

        async with HaikuRAG(config=config) as rag:
            with pytest.raises(UnknownDatabaseError, match="nope"):
                await rag.search("cats", search_type="fts", sources=["nope"])

    @pytest.mark.asyncio
    async def test_an_unopenable_database_fails_the_query(self, tmp_path):
        config = _config(tmp_path, ["alpha", "missing"])
        await _seed(config, "alpha", ["alpha document about cats"])

        with pytest.raises(SourceUnavailableError, match="missing"):
            async with HaikuRAG(config=config) as rag:
                await rag.search("cats", search_type="fts")


class TestSingleDatabaseUnchanged:
    @pytest.mark.asyncio
    async def test_source_is_unset_without_configured_databases(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True) as rag:
            doc = DoclingDocument(name="one")
            doc.add_text(label=DocItemLabel.TEXT, text="a document about cats")
            await rag.import_document(
                doc,
                [
                    Chunk(
                        content="a document about cats",
                        embedding=[0.1] * get_config().embeddings.model.vector_dim,
                        order=0,
                    )
                ],
                uri="test://one",
            )
            results = await rag.search("cats", search_type="fts")

        assert results
        assert all(r.source is None for r in results)


class TestOneQueryVector:
    @pytest.mark.asyncio
    async def test_a_search_embeds_the_query_once_for_the_whole_set(
        self, tmp_path, query_embedding
    ):
        """Each database owns an embedder, so embedding per database costs a
        round trip each on a remote endpoint."""
        config = _config(tmp_path, ["alpha", "beta", "gamma"])
        for name in ("alpha", "beta", "gamma"):
            await _seed(config, name, [f"{name} one"])

        async with HaikuRAG(config=config, read_only=True) as rag:
            await rag.search("one")

        assert query_embedding == ["one"]


class TestOneEmbedderAcrossTheSet:
    """A set is searched with one query vector, so a database written with
    another model would answer from a different space."""

    @pytest.mark.asyncio
    async def test_disagreeing_databases_cannot_be_searched_together(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])
        await _restore_embedder(config, "beta", model_name="some-other-model")

        async with HaikuRAG(config=config, read_only=True) as rag:
            with pytest.raises(ConfigMismatchError, match="different embedders"):
                await rag.search("one")

    @pytest.mark.asyncio
    async def test_a_database_asked_for_alone_is_never_compared(
        self, tmp_path, query_embedding
    ):
        """Only databases searched together have to agree."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])
        await _restore_embedder(config, "beta", model_name="some-other-model")

        async with HaikuRAG(config=config, read_only=True) as rag:
            assert await rag.search("one", sources=["alpha"]) is not None
            assert await rag.count_documents(filter=None) is not None

    @pytest.mark.asyncio
    async def test_full_text_search_needs_no_agreement(self, tmp_path):
        """Full-text search embeds nothing, so which model wrote each database
        does not come into it."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])
        await _restore_embedder(config, "beta", model_name="some-other-model")

        async with HaikuRAG(config=config, read_only=True) as rag:
            results = await rag.search("one", search_type="fts")

        assert {r.source for r in results} == {"alpha", "beta"}

    @pytest.mark.asyncio
    async def test_agreeing_databases_search_together(self, tmp_path, query_embedding):
        """The databases agree with each other; that they were written by a
        differently-spelled provider than the config is the soft case."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])
        await _restore_embedder(config, "alpha", provider="openai")
        await _restore_embedder(config, "beta", provider="openai")

        async with HaikuRAG(config=config, read_only=True) as rag:
            assert len(await rag.search("one")) > 0


class TestRerankerFusion:
    @pytest.mark.asyncio
    async def test_the_reranker_scores_the_union_and_owners_survive(
        self, tmp_path, monkeypatch
    ):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        stub = StubReranker()
        monkeypatch.setattr(HaikuRAG, "reranker", property(lambda self: stub))

        async with HaikuRAG(config=config) as rag:
            results = await rag.search("cats", limit=2, search_type="fts")

        # It saw both databases' candidates, not one database at a time.
        assert len(stub.seen) == 2
        assert {c.split()[0] for c in stub.seen} == {"alpha", "beta"}
        # Each result still knows which database it came from.
        for r in results:
            assert r.source is not None
            assert r.content.startswith(r.source)

    @pytest.mark.asyncio
    async def test_a_closing_failure_does_not_mask_the_exit(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        rag = HaikuRAG(config=config)
        await rag.__aenter__()
        await rag.clients_for(["alpha", "beta"])
        assert isinstance(rag._session, FederatedSession)
        sessions = rag._session._sessions

        async def boom():
            raise RuntimeError("close failed")

        sessions["alpha"].aclose = boom  # ty: ignore[invalid-assignment]
        beta = sessions["beta"].store

        await rag.__aexit__(None, None, None)

        # The failure is swallowed, and the sibling is still closed after it.
        assert rag._clients == {}
        assert rag._session._sessions == {}
        assert not beta.db.is_open()

    @pytest.mark.asyncio
    async def test_multimodal_reranking_attaches_each_database_own_pictures(
        self, tmp_path, monkeypatch
    ):
        """Picture self_refs repeat across databases exactly as they do across
        documents, so the pre-rerank attach must stay per database."""
        config = _config(tmp_path, ["alpha", "beta"])
        config.reranking.multimodal = True
        dim = get_config().embeddings.model.vector_dim

        for name in ("alpha", "beta"):
            async with HaikuRAG(config=config, create=True, sources=[name]) as rag:
                doc = DoclingDocument(name=name)
                doc.add_text(label=DocItemLabel.TEXT, text=f"{name} figure of cats")
                await rag.import_document(
                    doc,
                    [
                        Chunk(
                            content=f"{name} figure of cats",
                            embedding=[0.1] * dim,
                            order=0,
                            metadata={
                                "doc_item_refs": ["#/pictures/0"],
                                "labels": ["picture"],
                            },
                        )
                    ],
                    uri=f"test://{name}/figure",
                )
                [document] = await rag.list_documents()
                assert document.id is not None
                await rag.document_item_repository.create_items(
                    document.id,
                    [
                        DocumentItem(
                            document_id=document.id,
                            position=0,
                            self_ref="#/pictures/0",
                            label="picture",
                            text=f"caption {name}",
                            picture_data=f"bytes-{name}".encode(),
                        )
                    ],
                )

        stub = StubReranker()
        monkeypatch.setattr(HaikuRAG, "reranker", property(lambda self: stub))

        async with HaikuRAG(config=config) as rag:
            await rag.search("cats", limit=2, search_type="fts")

        assert stub.attached == {"alpha": b"bytes-alpha", "beta": b"bytes-beta"}


class TestOneReranker:
    @pytest.mark.asyncio
    async def test_the_set_builds_one_reranker_for_a_text_query(
        self, tmp_path, monkeypatch
    ):
        """Local rerankers load model weights per instance, so a set of
        databases must build one, not one each."""
        built = []
        monkeypatch.setattr(
            "haiku.rag.client.get_reranker",
            lambda config: built.append(config) or StubReranker(),
        )

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])
        built.clear()

        async with HaikuRAG(config=config) as rag:
            await rag.search("cats", limit=2, search_type="fts")

        assert len(built) == 1, f"built {len(built)} rerankers"

    @pytest.mark.asyncio
    async def test_an_image_query_builds_no_reranker(self, tmp_path, monkeypatch):
        """Opening a database must not build one either: an image query has no
        text to score against and never uses it."""
        built = []
        monkeypatch.setattr(
            "haiku.rag.client.get_reranker",
            lambda config: built.append(config) or StubReranker(),
        )

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])
        built.clear()

        async with HaikuRAG(config=config) as rag:
            await rag.clients_for(["alpha", "beta"])

        assert built == []

    @pytest.mark.asyncio
    async def test_the_reranker_is_closed_once(self, tmp_path, monkeypatch):
        """Handing the same object to every database and letting each close it
        would close it N times, and the federator not at all."""
        closes = []

        class CountingReranker(StubReranker):
            async def aclose(self):
                closes.append(1)

        monkeypatch.setattr(
            "haiku.rag.client.get_reranker", lambda config: CountingReranker()
        )

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            await rag.search("cats", limit=2, search_type="fts")

        assert closes == [1], f"closed {len(closes)} times"


class TestNarrowingToOneDatabase:
    """A selection of one is an ordinary search. Fusion exists to reconcile
    rankings from separate indexes, and there is nothing to reconcile."""

    @pytest.mark.asyncio
    async def test_narrowing_keeps_the_database_s_own_scores(self, tmp_path):
        """RRF scores position, so fusing one ranking would report 1/(60+rank)
        where the database reported a hybrid score."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats", "alpha on dogs"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as covering:
            narrowed = await covering.search(
                "cats", search_type="fts", sources=["alpha"]
            )
        async with HaikuRAG(config=config, sources=["alpha"]) as one:
            native = await one.search("cats", search_type="fts")

        assert [r.chunk_id for r in narrowed] == [r.chunk_id for r in native]
        assert [r.score for r in narrowed] == [r.score for r in native]
        assert all(r.source == "alpha" for r in narrowed)

    @pytest.mark.asyncio
    async def test_narrowing_does_not_embed_for_a_filter_matching_nothing(
        self, tmp_path, monkeypatch
    ):
        """One database embeds inside the repository, which returns early when
        the filter matches no document. Fusing would have embedded first."""
        from haiku.rag.embeddings import EmbedderWrapper

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        def explode(self, query):
            raise AssertionError("embedded a query no document could match")

        monkeypatch.setattr(EmbedderWrapper, "embed_query", explode)

        async with HaikuRAG(config=config) as covering:
            results = await covering.search(
                "cats", filter="uri = 'test://nothing'", sources=["alpha"]
            )

        assert results == []


class TestReciprocalRankFusion:
    """Without a reranker, scores from separate indexes are not comparable, so
    fusion ranks by position. These pin what that produces."""

    @staticmethod
    def _ranked(source: str, count: int, top: float) -> list[tuple[Chunk, float]]:
        return [
            (Chunk(id=f"{source}{i}", content=f"{source} {i}"), top - i / 100)
            for i in range(count)
        ]

    def _lopsided(self, count: int) -> list[list[tuple[Chunk, float]]]:
        """Every native score in the first database beats every one in the
        second, so anything ranking by score rather than position puts all of
        one before any of the other."""
        return [self._ranked("a", count, 0.9), self._ranked("b", count, 0.2)]

    async def _fuse_over(self, tmp_path, per_source, limit):
        from haiku.rag.client.search import _fuse

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])
        async with HaikuRAG(config=config) as rag:
            assert rag.reranker is None
            clients = await rag.clients_for(["alpha", "beta"])
            fused = await _fuse(rag, clients, "cats", per_source, limit)
        return [(owner.source, chunk.id, score) for owner, chunk, score in fused]

    @pytest.mark.asyncio
    async def test_databases_interleave_by_rank(self, tmp_path):
        """Each contributes its rank-1 before either contributes its rank-2."""
        fused = await self._fuse_over(tmp_path, self._lopsided(3), 10)

        assert [(source, cid) for source, cid, _ in fused] == [
            ("alpha", "a0"),
            ("beta", "b0"),
            ("alpha", "a1"),
            ("beta", "b1"),
            ("alpha", "a2"),
            ("beta", "b2"),
        ]

    @pytest.mark.asyncio
    async def test_the_score_is_the_reciprocal_of_the_rank(self, tmp_path):
        fused = await self._fuse_over(tmp_path, self._lopsided(2), 10)

        assert [score for _, _, score in fused] == [
            1 / 61,
            1 / 61,
            1 / 62,
            1 / 62,
        ]

    @pytest.mark.asyncio
    async def test_equal_scores_keep_the_configured_order(self, tmp_path):
        """Every rank ties across databases, so the tiebreak decides all of it."""
        fused = await self._fuse_over(tmp_path, self._lopsided(2), 10)

        assert [source for source, _, _ in fused] == ["alpha", "beta", "alpha", "beta"]

    @pytest.mark.asyncio
    async def test_the_limit_cuts_the_fused_list(self, tmp_path):
        """Each database was asked for enough to fill the window on its own."""
        fused = await self._fuse_over(tmp_path, self._lopsided(5), 3)

        assert [(source, cid) for source, cid, _ in fused] == [
            ("alpha", "a0"),
            ("beta", "b0"),
            ("alpha", "a1"),
        ]


class TestFusingWhatARerankerReturns:
    @pytest.mark.asyncio
    async def test_a_reranker_returning_copies_is_named(self, tmp_path):
        """Candidates are mapped back to their database by identity, because
        chunk ids repeat between copies of one. A reranker that rebuilds its
        chunks loses that, and saying so beats a KeyError."""
        from haiku.rag.client.search import _fuse

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        class Rebuilds:
            async def rerank(self, query, chunks, top_n=10):
                return [(chunk.model_copy(), 1.0) for chunk in chunks[:top_n]]

        async with HaikuRAG(config=config) as rag:
            clients = await rag.clients_for(["alpha", "beta"])
            rag.__dict__["_own_reranker"] = Rebuilds()
            per_source = [
                await c.chunk_repository.search("cats", 5, "fts") for c in clients
            ]

            with pytest.raises(ValueError, match="objects from the list"):
                await _fuse(rag, clients, "cats", per_source, 5)


class TestComparingEmbedders:
    @pytest.mark.asyncio
    async def test_a_database_recording_no_embedder_is_not_compared(self, tmp_path):
        """A database whose settings never recorded one cannot disagree with a
        database that did, so there is nothing to reject."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            alpha, beta = await rag.clients_for(["alpha", "beta"])
            beta.store.stored_embedding = None

            rag._require_one_embedder([alpha, beta])


class TestOneNamedDatabase:
    @pytest.mark.asyncio
    async def test_a_single_named_database_keeps_its_name(self, tmp_path):
        """Named in config is named in results, even as the only entry."""
        config = _config(tmp_path, ["alpha"])
        await _seed(config, "alpha", ["alpha document about cats"])

        async with HaikuRAG(config=config) as rag:
            results = await rag.search("cats", search_type="fts")

        assert results
        assert all(r.source == "alpha" for r in results)

    @pytest.mark.asyncio
    async def test_selecting_nothing_at_construction_is_rejected(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])

        with pytest.raises(ValueError, match="selects no database"):
            async with HaikuRAG(config=config, sources=[]):
                pass

    @pytest.mark.asyncio
    async def test_selecting_nothing_means_the_same_with_one_database(self, tmp_path):
        """`sources=[]` selects nothing whether one database is configured or
        several, rather than raising on one path and returning nothing on the
        other."""
        config = _config(tmp_path, ["alpha"])
        await _seed(config, "alpha", ["alpha document about cats"])

        async with HaikuRAG(config=config) as rag:
            assert await rag.search("cats", search_type="fts", sources=[]) == []

    @pytest.mark.asyncio
    async def test_selecting_nothing_per_query_returns_nothing(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            assert await rag.search("cats", search_type="fts", sources=[]) == []
