import asyncio

import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from pydantic import ValidationError

from haiku.rag.client import HaikuRAG
from haiku.rag.config import get_config
from haiku.rag.config.models import AppConfig, LanceDBConfig
from haiku.rag.store.exceptions import SourceUnavailableError
from haiku.rag.store.models import Chunk, DocumentItem
from haiku.rag.utils import locate_database


class TestConfig:
    def test_databases_and_uri_are_mutually_exclusive(self):
        with pytest.raises(ValidationError, match="databases"):
            LanceDBConfig(
                uri="s3://b/one.lancedb", databases={"one": "s3://b/one.lancedb"}
            )

    def test_databases_alone_is_fine(self):
        config = LanceDBConfig(databases={"one": "s3://b/one.lancedb"})
        assert config.databases == {"one": "s3://b/one.lancedb"}

    def test_uri_alone_is_fine(self):
        assert LanceDBConfig(uri="s3://b/one.lancedb").databases == {}


def _config(tmp_path, names) -> AppConfig:
    return AppConfig(
        lancedb=LanceDBConfig(
            databases={n: str(tmp_path / f"{n}.lancedb") for n in names}
        )
    )


async def _seed(config, name, contents):
    """Precomputed embeddings and FTS queries keep the embedder out of the way:
    these tests are about fusion, not retrieval quality."""
    dim = get_config().embeddings.model.vector_dim
    async with HaikuRAG(config=config, create=True, sources=[name]) as rag:
        for content in contents:
            doc = DoclingDocument(name=content)
            doc.add_text(label=DocItemLabel.TEXT, text=content)
            await rag.import_document(
                doc,
                [Chunk(content=content, embedding=[0.1] * dim, order=0)],
                uri=f"test://{name}/{content}",
            )


class TestNamingADatabaseDirectly:
    @pytest.mark.asyncio
    async def test_an_explicit_db_path_wins_over_the_configured_set(
        self, tmp_path, temp_db_path
    ):
        """A caller that names a path means that database, not the configured
        set: the CLI resolves `--db` to one and must not fan out instead."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])

        async with HaikuRAG(temp_db_path, config=config, create=True) as rag:
            assert rag._federated == {}
            assert rag._source is None
            assert rag.store.db_path == temp_db_path

    @pytest.mark.asyncio
    async def test_one_configured_database_is_opened_by_name(self, tmp_path):
        """A set of one is not federated, and the client resolves it."""
        config = _config(tmp_path, ["alpha"])
        await _seed(config, "alpha", ["alpha document about cats"])

        async with HaikuRAG(config=config) as rag:
            assert rag._federated == {}
            assert rag._source == "alpha"
            results = await rag.search("cats", search_type="fts", limit=10)

        assert [r.source for r in results] == ["alpha"]


class TestOpeningDatabases:
    @pytest.mark.asyncio
    async def test_missing_databases_open_together(self, tmp_path):
        """A cold fan-out costs one open, not their sum. On object storage a
        serial loop is the difference between one round trip and N."""
        names = ["alpha", "beta", "gamma"]
        config = _config(tmp_path, names)
        for name in names:
            await _seed(config, name, [f"{name} document about cats"])

        async with HaikuRAG(config=config) as rag:
            barrier = asyncio.Barrier(len(names))
            open_one = rag._open_client

            async def gated(name: str, location: str):
                # Every open has to be in flight before any of them finishes, so
                # a serial loop cannot get past this and the wait times out.
                await barrier.wait()
                return await open_one(name, location)

            rag._open_client = gated
            clients = await asyncio.wait_for(rag.clients_for(names), timeout=15)

        assert {client._source for client in clients} == set(names)

    @pytest.mark.asyncio
    async def test_a_failed_open_does_not_leak_the_ones_that_worked(self, tmp_path):
        """Opening together means a failure has siblings already open. They are
        tracked before it is reported, so closing the set closes them."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        config.lancedb.databases["beta"] = str(tmp_path / "absent.lancedb")

        async with HaikuRAG(config=config) as rag:
            with pytest.raises(SourceUnavailableError, match="beta"):
                await rag.clients_for(["alpha", "beta"])

            assert set(rag._clients) == {"alpha"}

    @pytest.mark.asyncio
    async def test_a_database_named_twice_is_opened_once(self, tmp_path):
        """Fusion would count a repeated database as two rank lists."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            clients = await rag.clients_for(["alpha", "alpha", "beta"])

        assert [client._source for client in clients] == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_a_database_named_twice_returns_each_result_once(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            results = await rag.search(
                "cats", limit=10, search_type="fts", sources=["alpha", "alpha"]
            )

        assert [r.source for r in results] == ["alpha"]

    @pytest.mark.asyncio
    async def test_one_database_named_twice_is_still_that_database(self, tmp_path):
        """A client covering a single named database compares the selection
        against its own name, so repeats have to collapse first."""
        config = _config(tmp_path, ["alpha"])
        await _seed(config, "alpha", ["alpha document about cats"])

        async with HaikuRAG(config=config) as rag:
            covering = await rag.clients_covering(["alpha", "alpha"])

        assert [client._source for client in covering] == ["alpha"]


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
            with pytest.raises(KeyError, match="nope"):
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


class TestLocate:
    def test_a_scheme_is_a_uri(self):
        assert locate_database("s3://bucket/one.lancedb") == (
            "s3://bucket/one.lancedb",
            None,
        )

    def test_anything_else_is_a_local_path(self):
        uri, db_path = locate_database("/data/one.lancedb")
        assert uri == ""
        assert db_path is not None and str(db_path) == "/data/one.lancedb"


class TestSelection:
    @pytest.mark.asyncio
    async def test_unknown_source_at_construction_is_rejected(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])

        with pytest.raises(KeyError, match="nope"):
            async with HaikuRAG(config=config, sources=["nope"]):
                pass

    @pytest.mark.asyncio
    async def test_unknown_source_across_several_databases_is_rejected(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            with pytest.raises(KeyError, match="nope"):
                await rag.search("cats", search_type="fts", sources=["nope"])

    @pytest.mark.asyncio
    async def test_no_matches_anywhere_returns_nothing(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            assert await rag.search("aardvarks", search_type="fts") == []


class StubReranker:
    """Scores the union, reversing it so the ordering is unmistakably its own."""

    def __init__(self):
        self.seen: list[str] = []

    async def rerank(self, query, chunks, top_n):
        self.seen = [c.content for c in chunks]
        # Whatever the caller attached before handing them over.
        self.attached = {
            c.content.split()[0]: c._picture_data
            for c in chunks
            if getattr(c, "_picture_data", None)
        }
        return [(c, 1.0 - i) for i, c in enumerate(reversed(chunks))][:top_n]


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

        async def boom(exc_type, exc_val, exc_tb):
            raise RuntimeError("close failed")

        rag._clients["alpha"].__aexit__ = boom  # ty: ignore[invalid-assignment]

        await rag.__aexit__(None, None, None)

        assert rag._clients == {}

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


class TestLazyOpening:
    @pytest.mark.asyncio
    async def test_entering_opens_nothing(self, tmp_path):
        """25 configured databases queried a few at a time must not all open."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            assert rag._clients == {}

    @pytest.mark.asyncio
    async def test_only_the_selected_database_opens(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            await rag.search("cats", search_type="fts", sources=["alpha"])
            assert list(rag._clients) == ["alpha"]

    @pytest.mark.asyncio
    async def test_an_unselected_broken_database_does_not_break_a_query(self, tmp_path):
        """A database nobody asked for cannot fail a query."""
        config = _config(tmp_path, ["alpha", "missing"])
        await _seed(config, "alpha", ["alpha document about cats"])

        async with HaikuRAG(config=config) as rag:
            results = await rag.search("cats", search_type="fts", sources=["alpha"])

        assert [r.source for r in results] == ["alpha"]

    @pytest.mark.asyncio
    async def test_a_selected_broken_database_fails_the_query(self, tmp_path):
        config = _config(tmp_path, ["alpha", "missing"])
        await _seed(config, "alpha", ["alpha document about cats"])

        async with HaikuRAG(config=config) as rag:
            with pytest.raises(SourceUnavailableError, match="missing"):
                await rag.search("cats", search_type="fts")


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


class TestFailureNaming:
    @pytest.mark.asyncio
    async def test_a_single_named_database_is_reported_by_name(self, tmp_path):
        """One configured database is still a named one: it must not fall back to
        the raw error, which spells out the path."""
        config = _config(tmp_path, ["alpha"])

        with pytest.raises(SourceUnavailableError, match="alpha") as caught:
            async with HaikuRAG(config=config):
                pass

        assert str(tmp_path) not in str(caught.value)
        assert caught.value.__cause__ is None

    @pytest.mark.asyncio
    async def test_a_legacy_uri_client_keeps_its_error(self, tmp_path):
        """Nothing named it, so there is no name to report instead."""
        with pytest.raises(FileNotFoundError):
            async with HaikuRAG(tmp_path / "nope.lancedb"):
                pass

    @pytest.mark.asyncio
    async def test_the_location_is_absent_from_the_whole_chain(self, tmp_path):
        config = _config(tmp_path, ["alpha", "missing"])
        await _seed(config, "alpha", ["alpha document about cats"])

        with pytest.raises(SourceUnavailableError) as caught:
            async with HaikuRAG(config=config) as rag:
                await rag.search("cats", search_type="fts")

        rendered = str(caught.value)
        error = caught.value.__cause__ or caught.value.__context__
        assert "missing.lancedb" not in rendered
        assert error is None, "the location-bearing cause is still attached"
