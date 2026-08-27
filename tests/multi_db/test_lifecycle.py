"""Opening, borrowing and closing the databases a client covers."""

import asyncio

import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from haiku.rag.client import HaikuRAG
from haiku.rag.client.session import FederatedSession
from haiku.rag.config import get_config
from haiku.rag.store.exceptions import (
    AmbiguousDatabaseError,
    SourceUnavailableError,
)
from haiku.rag.store.models import Chunk
from tests.multi_db.helpers import (
    _config,
    _seed,
)


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

            async def gated(name):
                # Every open has to be in flight before any of them finishes, so
                # a serial loop cannot get past this and the wait times out.
                await barrier.wait()
                await open_one(name)

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
    async def test_a_cancelled_open_does_not_leak_the_ones_that_worked(self, tmp_path):
        """Cancellation discards the fan-out's results rather than returning them,
        so a database that opened while a sibling was still pending is reachable
        only because the opener recorded it."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            assert isinstance(rag._session, FederatedSession)
            open_one = rag._session._open
            alpha_open = asyncio.Event()

            async def staged(name):
                if name == "beta":
                    await asyncio.sleep(60)
                await open_one(name)
                alpha_open.set()

            rag._session._open = staged
            fanout = asyncio.create_task(rag.clients_for(["alpha", "beta"]))
            await asyncio.wait_for(alpha_open.wait(), timeout=15)
            fanout.cancel()
            with pytest.raises(asyncio.CancelledError):
                await fanout

            assert set(rag._session._sessions) == {"alpha"}
            alpha = rag._session._sessions["alpha"]

        assert not alpha.store.db.is_open()

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


class TestReportingWhereADatabaseIs:
    @pytest.mark.asyncio
    async def test_one_database_reports_its_location_and_a_set_none(self, tmp_path):
        """A set has no single location to report. What the CLI and the info
        modal print comes from here."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as covering:
            assert covering.location is None

        async with HaikuRAG(config=config, sources=["alpha"]) as one:
            assert one.location == tmp_path / "alpha.lancedb"


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
            alpha.__dict__["_own_reranker"] = Reranker()

        assert closed == ["reranker"]


class TestSharingTheReranker:
    @pytest.mark.asyncio
    async def test_the_set_builds_and_closes_one_reranker(self, tmp_path, monkeypatch):
        """A local reranker loads model weights, so one per database in a set
        would load the same weights that many times."""
        import haiku.rag.client as client_module

        built: list[object] = []
        closed: list[object] = []

        class Reranker:
            def __init__(self):
                built.append(self)

            async def aclose(self):
                closed.append(self)

        monkeypatch.setattr(client_module, "get_reranker", lambda config: Reranker())

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            alpha, beta = await rag.clients_for(["alpha", "beta"])

            assert alpha.reranker is beta.reranker is rag.reranker
            assert len(built) == 1

        assert closed == built


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


class TestDatabaseIndependentWork:
    """Converting, chunking and titling are functions of the configuration, not
    of a database, so covering a set does not stop them."""

    @pytest.mark.asyncio
    async def test_chunking_opens_no_database(self, tmp_path, monkeypatch):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])

        opened: list[str] = []

        async def refuse(self, name):
            opened.append(name)
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
