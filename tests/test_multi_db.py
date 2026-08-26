import asyncio

import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from pydantic import ValidationError

from haiku.rag.client import HaikuRAG
from haiku.rag.client.scope import DatabaseScope
from haiku.rag.client.session import FederatedSession
from haiku.rag.config import get_config
from haiku.rag.config.models import AppConfig, LanceDBConfig
from haiku.rag.store.exceptions import (
    AmbiguousDatabaseError,
    ConfigMismatchError,
    SourceUnavailableError,
)
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
            assert not rag.covers_multiple
            assert rag.source is None
            assert rag.store.db_path == temp_db_path

    @pytest.mark.asyncio
    async def test_one_configured_database_is_opened_by_name(self, tmp_path):
        """A set of one is not federated, and the client resolves it."""
        config = _config(tmp_path, ["alpha"])
        await _seed(config, "alpha", ["alpha document about cats"])

        async with HaikuRAG(config=config) as rag:
            assert not rag.covers_multiple
            assert rag.source == "alpha"
            results = await rag.search("cats", search_type="fts", limit=10)

        assert [r.source for r in results] == ["alpha"]


class TestOneConfiguredLocation:
    """`lancedb.uri` places one unnamed database, at a URI or at a local path."""

    def _config(self, location) -> AppConfig:
        return AppConfig(lancedb=LanceDBConfig(uri=str(location)))

    @pytest.mark.asyncio
    async def test_a_local_uri_opens_the_configured_database(self, tmp_path):
        located = tmp_path / "notes.lancedb"
        config = self._config(located)

        async with HaikuRAG(config=config, create=True) as rag:
            assert rag.store.db_path == located
            # It places a database without naming one: only `lancedb.databases`
            # assigns the name results and citations carry.
            assert rag.source is None
        assert located.exists()

    @pytest.mark.asyncio
    async def test_an_explicit_path_overrides_a_local_uri(self, tmp_path):
        """`--db` overrides the configured location for one invocation."""
        config = self._config(tmp_path / "configured.lancedb")
        chosen = tmp_path / "chosen.lancedb"

        async with HaikuRAG(chosen, config=config, create=True) as rag:
            assert rag.store.db_path == chosen
        assert chosen.exists()
        assert not (tmp_path / "configured.lancedb").exists()

    @pytest.mark.asyncio
    async def test_a_local_uri_that_does_not_exist_is_refused(self, tmp_path):
        """A mistyped path fails instead of quietly becoming an empty database,
        which is what a value carrying a scheme would do."""
        config = self._config(tmp_path / "typo.lancedb")

        with pytest.raises(FileNotFoundError):
            async with HaikuRAG(config=config):
                pass
        assert not (tmp_path / "typo.lancedb").exists()

    def test_a_uri_with_a_scheme_stays_a_uri(self, tmp_path):
        """Object storage has no local path to check, and a location that does
        not exist yet is normal there."""
        from haiku.rag.store.engine import ConnectionMode

        config = self._config("s3://bucket/one.lancedb")

        [ref] = DatabaseScope.resolve(config).databases
        one, db_path = ref.connection(config)

        assert db_path is None
        assert ConnectionMode.from_config(one) == ConnectionMode.OBJECT_STORAGE


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
            assert isinstance(rag._session, FederatedSession)
            barrier = asyncio.Barrier(len(names))
            open_one = rag._session._open

            async def gated(ref):
                # Every open has to be in flight before any of them finishes, so
                # a serial loop cannot get past this and the wait times out.
                await barrier.wait()
                return await open_one(ref)

            rag._session._open = gated
            clients = await asyncio.wait_for(rag.clients_for(names), timeout=15)

        assert {client.source for client in clients} == set(names)

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

            assert isinstance(rag._session, FederatedSession)
            assert set(rag._session._sessions) == {"alpha"}

    @pytest.mark.asyncio
    async def test_a_database_named_twice_is_opened_once(self, tmp_path):
        """Fusion would count a repeated database as two rank lists."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            clients = await rag.clients_for(["alpha", "alpha", "beta"])

        assert [client.source for client in clients] == ["alpha", "beta"]

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

        assert [client.source for client in covering] == ["alpha"]


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
    async def test_a_document_held_by_two_databases_answers_from_the_first(
        self, tmp_path
    ):
        """A database copied from another holds the same ids. A read has an
        answer wherever it finds one, and which one it is has to be the
        configured order rather than whichever replied first."""
        import shutil

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        shutil.copytree(tmp_path / "alpha.lancedb", tmp_path / "beta.lancedb")

        async with HaikuRAG(config=config) as rag:
            beta = (await rag.clients_for(["beta"]))[0]
            [target] = await beta.document_repository.list_all(limit=1)
            assert target.id is not None

            found = await rag.get_document_by_id(target.id)

        assert found is not None and found.source == "alpha"

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


class TestClosingASet:
    @pytest.mark.asyncio
    async def test_every_database_opened_is_released(self, tmp_path):
        """A covered database owns an embedder and may owe a vacuum. Closing only
        its connection would leave both behind."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])

        released: list[str | None] = []
        drained: list[str | None] = []

        async with HaikuRAG(config=config, read_only=True) as rag:
            assert isinstance(rag._session, FederatedSession)
            await rag.clients_for(["alpha", "beta"])
            for name, session in rag._session._sessions.items():
                original = session.store.embedder.aclose
                drain = session.drain_vacuum

                async def release(_original=original, _name=name):
                    released.append(_name)
                    return await _original()

                async def drain_it(_drain=drain, _name=name):
                    drained.append(_name)
                    return await _drain()

                session.store.embedder.aclose = release
                session.drain_vacuum = drain_it

        assert sorted(released) == ["alpha", "beta"]
        assert sorted(drained) == ["alpha", "beta"]


class TestNamingOneOfTheSetOnTheCommandLine:
    """`--database NAME` reaches the application layer as a name, and every
    client it opens has to honour it — one that ignores it covers the set and
    quietly answers from the wrong database."""

    @pytest.mark.asyncio
    async def test_a_named_database_is_the_one_read(self, tmp_path, capsys):
        from haiku.rag.app import HaikuRAGApp

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        scope = DatabaseScope.resolve(config).select(["beta"])
        app = HaikuRAGApp(scope=scope, config=config, read_only=True)
        await app.list_documents()

        # Rich wraps long lines, so match the unwrapped part of the URI.
        printed = capsys.readouterr().out
        assert "test://beta/" in printed
        assert "test://alpha/" not in printed

    @pytest.mark.asyncio
    async def test_naming_none_of_them_covers_the_set(self, tmp_path, capsys):
        from haiku.rag.app import HaikuRAGApp

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        app = HaikuRAGApp(
            scope=DatabaseScope.resolve(config), config=config, read_only=True
        )
        await app.list_documents()

        printed = capsys.readouterr().out
        assert "test://alpha/" in printed
        assert "test://beta/" in printed


class TestPlacingADatabase:
    """What a client says about the databases it covers, so nothing outside has
    to read its private state to find out."""

    @pytest.mark.asyncio
    async def test_a_set_names_every_database_it_covers(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])

        async with HaikuRAG(config=config, read_only=True) as rag:
            assert rag.covers_multiple
            assert rag.source_names == ("alpha", "beta")
            assert rag.source is None

    @pytest.mark.asyncio
    async def test_one_named_database_names_itself(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])

        async with HaikuRAG(config=config, read_only=True, sources=["alpha"]) as rag:
            assert not rag.covers_multiple
            assert rag.source_names == ("alpha",)
            assert rag.source == "alpha"

    @pytest.mark.asyncio
    async def test_a_named_database_keeps_its_name_on_re_entry(self, tmp_path):
        """Entering derives a single-database configuration from what was
        configured. Deriving it from the last derivation loses the name."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])

        rag = HaikuRAG(config=config, read_only=True, sources=["alpha"])
        async with rag:
            assert rag.source == "alpha"
        async with rag:
            assert rag.source == "alpha"
            assert rag.source_names == ("alpha",)
            results = await rag.search("cats", search_type="fts")

        assert {r.source for r in results} == {"alpha"}

    @pytest.mark.asyncio
    async def test_an_unnamed_database_names_nothing(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True) as rag:
            assert rag.source_names == ()
            assert rag.source is None

    @pytest.mark.asyncio
    async def test_the_reader_for_a_database_is_the_client_holding_it(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])

        async with HaikuRAG(config=config, read_only=True) as rag:
            reader = await rag.reader_for("beta")

            assert reader is not None
            assert reader.source == "beta"
            # Asked twice, the same wrapper comes back.
            assert await rag.reader_for("beta") is reader

    @pytest.mark.asyncio
    async def test_a_client_reading_one_database_is_its_own_reader(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True) as rag:
            assert await rag.reader_for(None) is rag
            assert await rag.reader_for("anything") is rag

    @pytest.mark.asyncio
    async def test_a_set_cannot_place_evidence_that_names_no_database(self, tmp_path):
        """Evidence recorded before databases could be named carries no source."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])

        async with HaikuRAG(config=config, read_only=True) as rag:
            assert await rag.reader_for(None) is None


class TestBorrowedDatabases:
    """A client for one of a set wraps a database the set opened."""

    @pytest.mark.asyncio
    async def test_closing_a_borrowed_client_leaves_the_set_working(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            (alpha,) = await rag.clients_for(["alpha"])
            store = alpha.store

            alpha.close()
            assert store.db.is_open(), "close() closed a database it borrowed"

            await alpha.__aexit__(None, None, None)
            assert store.db.is_open(), "exit closed a database it borrowed"

            results = await rag.search("cats", search_type="fts")

        assert {r.source for r in results} == {"alpha", "beta"}
        assert not store.db.is_open(), "the set left a database open"

    @pytest.mark.asyncio
    async def test_entering_a_borrowed_client_reuses_its_database(self, tmp_path):
        """`async with` on a borrowed client is a plausible thing to write.
        Opening a second session would leak it, since teardown declines to close
        what this client did not open."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])

        async with HaikuRAG(config=config) as rag:
            (alpha,) = await rag.clients_for(["alpha"])
            borrowed = alpha.store

            async with alpha as entered:
                assert entered is alpha
                assert alpha.store is borrowed, "entry opened a second database"

            assert borrowed.db.is_open(), "exit closed a database it borrowed"
            assert alpha.store is borrowed

        assert not borrowed.db.is_open(), "the set left a database open"

    @pytest.mark.asyncio
    async def test_a_borrowed_client_releases_what_it_built(self, tmp_path):
        """Its reranker is its own; the database it wraps is not."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])

        closed: list[str] = []

        class Reranker:
            async def aclose(self):
                closed.append("reranker")

        async with HaikuRAG(config=config) as rag:
            (alpha,) = await rag.clients_for(["alpha"])
            alpha.__dict__["reranker"] = Reranker()

        assert closed == ["reranker"]


class TestDatabaseIndependentWork:
    """Converting, chunking and titling are functions of the configuration, not
    of a database, so covering a set does not stop them."""

    @pytest.mark.asyncio
    async def test_chunking_opens_no_database(self, tmp_path, monkeypatch):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])

        opened: list[str] = []

        async def refuse(self, ref):
            opened.append(ref.name)
            raise AssertionError("opened a database to chunk a document")

        monkeypatch.setattr(FederatedSession, "_open", refuse)

        doc = DoclingDocument(name="note")
        doc.add_text(
            label=DocItemLabel.TEXT, text="Boltzmann machines are energy based."
        )

        async with HaikuRAG(config=config, read_only=True) as rag:
            chunks = await rag.chunk(doc)

        assert opened == []
        assert [c.content for c in chunks]

    @pytest.mark.asyncio
    async def test_the_embedder_is_built_once_and_closed_once(
        self, tmp_path, monkeypatch
    ):
        """The parent owns the embedder it built, so leaving the context closes
        it, once."""
        from haiku.rag.embeddings import EmbedderWrapper

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])

        closed: list[object] = []
        original = EmbedderWrapper.aclose

        async def counting(self):
            closed.append(self)
            return await original(self)

        monkeypatch.setattr(EmbedderWrapper, "aclose", counting)

        rag = HaikuRAG(config=config, read_only=True)
        async with rag:
            built = rag.embedder
            assert rag.embedder is built

        assert closed == [built]

    @pytest.mark.asyncio
    async def test_re_entering_a_set_builds_a_fresh_embedder(self, tmp_path):
        """Teardown closes the embedder, so keeping it would hand the next
        context one that is already closed."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])

        rag = HaikuRAG(config=config, read_only=True)
        async with rag:
            first = rag.embedder
        async with rag:
            assert rag.embedder is not first

    @pytest.mark.asyncio
    async def test_re_entering_one_database_builds_a_fresh_embedder(self, temp_db_path):
        """One database opens a new store on re-entry, and the embedder is that
        store's."""
        rag = HaikuRAG(temp_db_path, create=True)
        async with rag:
            first = rag.embedder
        async with rag:
            assert rag.embedder is rag.store.embedder
            assert rag.embedder is not first

    @pytest.mark.asyncio
    async def test_a_set_nobody_asked_anything_of_builds_no_embedder(self, tmp_path):
        """Built on first use, so a client that answered nothing holds nothing."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])

        async with HaikuRAG(config=config, read_only=True) as rag:
            assert "embedder" not in rag.__dict__

    @pytest.mark.asyncio
    async def test_one_database_still_uses_its_store_s_embedder(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True) as rag:
            assert rag.embedder is rag.store.embedder


class TestCreatingNeedsOneDatabase:
    """Creating names a database. Covering a set, the flag had nothing to act on
    and was accepted anyway, leaving the first query to fail on whichever
    database turned out to be missing."""

    @pytest.mark.asyncio
    async def test_creating_a_set_is_refused(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])

        with pytest.raises(AmbiguousDatabaseError, match="alpha, beta"):
            async with HaikuRAG(config=config, create=True):
                pass

    @pytest.mark.asyncio
    async def test_naming_one_of_the_set_creates_it(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])

        async with HaikuRAG(config=config, create=True, sources=["alpha"]) as rag:
            assert await rag.count_documents() == 0

        assert (tmp_path / "alpha.lancedb").exists()
        assert not (tmp_path / "beta.lancedb").exists()

    @pytest.mark.asyncio
    async def test_covering_a_set_without_creating_is_unaffected(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])

        async with HaikuRAG(config=config) as rag:
            assert await rag.count_documents() == 2


class TestOperationsThatNeedOneDatabase:
    @pytest.mark.asyncio
    async def test_writing_names_the_databases_it_covers(self, tmp_path):
        """A domain error, so a caller can tell an unsupported selection from a
        missing attribute."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])

        async with HaikuRAG(config=config) as rag:
            with pytest.raises(AmbiguousDatabaseError, match="alpha, beta"):
                await rag.create_document("orphan")
            with pytest.raises(AmbiguousDatabaseError, match="clients_for"):
                await rag.vacuum()
            with pytest.raises(AmbiguousDatabaseError, match="close"):
                rag.close()

    @pytest.mark.asyncio
    async def test_a_set_has_no_store_of_its_own(self, tmp_path):
        """A store and its repositories belong to one database. `clients_for`
        reaches the one holding a given database."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])

        async with HaikuRAG(config=config) as rag:
            for name in (
                "store",
                "document_repository",
                "chunk_repository",
                "document_item_repository",
            ):
                with pytest.raises(AttributeError, match=name):
                    getattr(rag, name)

    @pytest.mark.asyncio
    async def test_a_selected_database_is_still_writable(self, tmp_path):
        """Naming one of the set is how a write picks its database."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])

        dim = get_config().embeddings.model.vector_dim
        written = DoclingDocument(name="written")
        written.add_text(label=DocItemLabel.TEXT, text="written")

        async with HaikuRAG(config=config) as rag:
            alpha = (await rag.clients_for(["alpha"]))[0]
            assert alpha.is_read_only is False
            document = await alpha.import_document(
                written,
                [Chunk(content="written", embedding=[0.1] * dim, order=0)],
                uri="test://alpha/written",
            )
            assert await alpha.count_documents() == 2

        assert document.id is not None


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
