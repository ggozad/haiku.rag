import asyncio

import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from pydantic import ValidationError

from haiku.rag.client import HaikuRAG
from haiku.rag.config import get_config
from haiku.rag.config.models import AppConfig, LanceDBConfig
from haiku.rag.store.exceptions import ConfigMismatchError, SourceUnavailableError
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


class TestNamingIsRequired:
    def test_a_blank_name_is_rejected(self):
        """An unnamed database is unreachable: every source check reads the
        empty name as no name at all."""
        with pytest.raises(ValidationError, match="entry with no name"):
            LanceDBConfig(databases={"": "/tmp/a.lancedb"})
        with pytest.raises(ValidationError, match="entry with no name"):
            LanceDBConfig(databases={"   ": "/tmp/a.lancedb"})

    def test_a_blank_location_is_rejected(self):
        """A blank location resolves to the working directory."""
        with pytest.raises(
            ValidationError, match=r"databases\[alpha\] has no location"
        ):
            LanceDBConfig(databases={"alpha": ""})


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


@pytest.fixture
def query_embedding(monkeypatch):
    """Vector search with no embedder behind it, recording the queries embedded.

    These tests are about which databases are asked and how often, not about
    retrieval quality, and CI has no embedding endpoint.
    """
    from haiku.rag.embeddings import EmbedderWrapper

    embedded: list[str] = []

    async def embed_query(self, text):
        embedded.append(text)
        return [0.1] * get_config().embeddings.model.vector_dim

    monkeypatch.setattr(EmbedderWrapper, "embed_query", embed_query)
    return embedded


async def _restore_embedder(config, name, *, provider=None, model_name=None):
    """Rewrite what one database records about the embedder that wrote it,
    standing in for a database built elsewhere with another model."""
    import json

    import lancedb

    _, db_path = locate_database(config.lancedb.databases[name])
    assert db_path is not None
    db = await lancedb.connect_async(str(db_path.resolve()))
    table = await db.open_table("settings")
    rows = (
        await table.query().where("id = 'settings'").limit(1).to_arrow()
    ).to_pylist()
    stored = json.loads(rows[0]["settings"])
    model = stored["embeddings"]["model"]
    if provider is not None:
        model["provider"] = provider
    if model_name is not None:
        model["name"] = model_name
    await table.update({"settings": json.dumps(stored)}, where="id = 'settings'")


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


class TestListingAcrossDatabases:
    """The chat TUI's document filter lists documents through the client, and a
    client covering a set has no repositories of its own."""

    @pytest.mark.asyncio
    async def test_listing_covers_every_database(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one", "alpha two"])
        await _seed(config, "beta", ["beta one"])

        async with HaikuRAG(config=config) as rag:
            docs = await rag.list_documents()

        assert {d.uri for d in docs} == {
            "test://alpha/alpha one",
            "test://alpha/alpha two",
            "test://beta/beta one",
        }

    @pytest.mark.asyncio
    async def test_counting_covers_every_database(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one", "alpha two"])
        await _seed(config, "beta", ["beta one"])

        async with HaikuRAG(config=config) as rag:
            assert await rag.count_documents() == 3

    @pytest.mark.asyncio
    async def test_a_limit_bounds_the_merged_listing(self, tmp_path):
        """A limit is that many documents in total, not that many per database."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one", "alpha two"])
        await _seed(config, "beta", ["beta one", "beta two"])

        async with HaikuRAG(config=config) as rag:
            assert len(await rag.list_documents(limit=3)) == 3
            assert len(await rag.list_documents(limit=2, offset=2)) == 2
            assert len(await rag.list_documents(offset=3)) == 1

    @pytest.mark.asyncio
    async def test_a_page_shows_every_database(self, tmp_path):
        """A window is taken across the databases, not filled from the first one:
        concatenating hides every database after whichever was listed first."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", [f"alpha {i}" for i in range(5)])
        await _seed(config, "beta", ["beta one"])

        async with HaikuRAG(config=config) as rag:
            page = await rag.list_documents(limit=3)

        assert len(page) == 3
        assert {(d.uri or "").split("/")[2] for d in page} == {"alpha", "beta"}

    @pytest.mark.asyncio
    async def test_a_filter_reaches_every_database(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])

        async with HaikuRAG(config=config) as rag:
            docs = await rag.list_documents(filter="uri LIKE 'test://beta/%'")

        assert [d.uri for d in docs] == ["test://beta/beta one"]


class TestLookupByIdentifier:
    """An id or a URI says nothing about which database holds it, and a client
    covering a set has no repositories of its own."""

    @pytest.mark.asyncio
    async def test_a_document_is_found_in_whichever_database_holds_it(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])

        async with HaikuRAG(config=config) as rag:
            beta = (await rag.clients_for(["beta"]))[0]
            [target] = await beta.document_repository.list_all(limit=1)
            assert target.id is not None

            found = await rag.get_document_by_id(target.id)
            by_uri = await rag.get_document_by_uri("test://alpha/alpha one")
            resolved = await rag.resolve_document(target.id)

        assert found is not None and found.uri == "test://beta/beta one"
        assert by_uri is not None and by_uri.uri == "test://alpha/alpha one"
        assert resolved is not None and resolved.uri == "test://beta/beta one"

    @pytest.mark.asyncio
    async def test_a_chunk_is_found_in_whichever_database_holds_it(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])

        async with HaikuRAG(config=config) as rag:
            beta = (await rag.clients_for(["beta"]))[0]
            [chunk] = await beta.chunk_repository.list_all(limit=1)
            assert chunk.id is not None

            found = await rag.get_chunk_by_id(chunk.id)

        assert found is not None and found.content == "beta one"

    @pytest.mark.asyncio
    async def test_an_unknown_identifier_is_absent_rather_than_an_error(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])

        async with HaikuRAG(config=config) as rag:
            assert (
                await rag.get_document_by_id("00000000-0000-4000-8000-000000000000")
                is None
            )
            assert (
                await rag.get_chunk_by_id("00000000-0000-4000-8000-000000000000")
                is None
            )
            assert await rag.get_document_by_uri("test://nowhere") is None


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


class TestReadOnlyMode:
    @pytest.mark.asyncio
    async def test_a_client_covering_a_set_reports_its_mode(self, tmp_path):
        """A client covering a set has no store of its own to ask."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])

        async with HaikuRAG(config=config, read_only=True) as rag:
            assert rag.is_read_only is True
        async with HaikuRAG(config=config) as rag:
            assert rag.is_read_only is False


class TestDocumentsNameTheirDatabase:
    """A listing that spans databases is unreadable when the documents do not
    say which one they came from, the same reason a search result carries one."""

    @pytest.mark.asyncio
    async def test_a_listing_names_each_document_s_database(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one", "alpha two"])
        await _seed(config, "beta", ["beta one"])

        async with HaikuRAG(config=config) as rag:
            docs = await rag.list_documents()

        assert {d.uri: d.source for d in docs} == {
            "test://alpha/alpha one": "alpha",
            "test://alpha/alpha two": "alpha",
            "test://beta/beta one": "beta",
        }

    @pytest.mark.asyncio
    async def test_a_looked_up_document_names_its_database(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])

        async with HaikuRAG(config=config) as rag:
            beta = (await rag.clients_for(["beta"]))[0]
            [target] = await beta.document_repository.list_all(limit=1)
            assert target.id is not None

            by_id = await rag.get_document_by_id(target.id)
            by_uri = await rag.get_document_by_uri("test://alpha/alpha one")
            resolved = await rag.resolve_document(target.id)

        assert by_id is not None and by_id.source == "beta"
        assert by_uri is not None and by_uri.source == "alpha"
        assert resolved is not None and resolved.source == "beta"

    @pytest.mark.asyncio
    async def test_one_named_database_still_names_itself(self, tmp_path):
        """`haiku-rag --database alpha list` opens one database, and its name is
        the whole reason the option exists."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])

        async with HaikuRAG(config=config, sources=["alpha"]) as rag:
            [listed] = await rag.list_documents()
            assert listed.id is not None
            by_id = await rag.get_document_by_id(listed.id)
            by_uri = await rag.get_document_by_uri("test://alpha/alpha one")

        assert listed.source == "alpha"
        assert by_id is not None and by_id.source == "alpha"
        assert by_uri is not None and by_uri.source == "alpha"

    @pytest.mark.asyncio
    async def test_one_database_leaves_the_source_unset(self, tmp_path, temp_db_path):
        """Nothing names the database when there is only one to name."""
        async with HaikuRAG(temp_db_path, create=True) as rag:
            dim = get_config().embeddings.model.vector_dim
            doc = DoclingDocument(name="solo")
            doc.add_text(label=DocItemLabel.TEXT, text="solo")
            await rag.import_document(
                doc,
                [Chunk(content="solo", embedding=[0.1] * dim, order=0)],
                uri="test://solo",
            )

            [listed] = await rag.list_documents()
            assert listed.source is None
            assert listed.id is not None
            assert (await rag.get_document_by_id(listed.id)).source is None


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
