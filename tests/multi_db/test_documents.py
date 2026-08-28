"""Listing and looking up documents across databases."""

import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from haiku.rag.client import HaikuRAG
from haiku.rag.client.session import SingleDatabaseSession
from haiku.rag.config import get_config
from haiku.rag.store.exceptions import UnknownDatabaseError
from haiku.rag.store.models import Chunk
from haiku.rag.store.models.document_item import DocumentItem
from tests.multi_db.helpers import (
    _config,
    _seed,
)


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
    async def test_a_chunk_is_read_from_the_database_its_source_names(self, tmp_path):
        """One chunk id in two databases, holding different content. A result
        carries the database it came from, so a caller holding one must be able
        to say which of the two it means."""
        import shutil

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["shared body"])
        shutil.copytree(tmp_path / "alpha.lancedb", tmp_path / "beta.lancedb")

        async with HaikuRAG(config=config, sources=["beta"]) as beta:
            [held] = await beta.chunk_repository.list_all(limit=1)
            assert held.id is not None
            await beta.store.chunks_table.update(
                {"content": "only in beta"}, where=f"id = '{held.id}'"
            )

        async with HaikuRAG(config=config) as rag:
            from_alpha = await rag.get_chunk_by_id(held.id, "alpha")
            from_beta = await rag.get_chunk_by_id(held.id, "beta")
            unqualified = await rag.get_chunk_by_id(held.id)
            with pytest.raises(UnknownDatabaseError):
                await rag.get_chunk_by_id(held.id, "gamma")

        assert from_alpha is not None and from_alpha.content == "shared body"
        assert from_beta is not None and from_beta.content == "only in beta"
        # Configured order, as an unqualified lookup has always answered.
        assert unqualified is not None and unqualified.content == "shared body"

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

    @staticmethod
    async def _collided(tmp_path):
        """Two databases holding one document id, where only beta's answers to
        the title and URI asked for."""
        import shutil

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["shared body"])
        shutil.copytree(tmp_path / "alpha.lancedb", tmp_path / "beta.lancedb")

        async with HaikuRAG(config=config, sources=["beta"]) as beta:
            [target] = await beta.document_repository.list_all(limit=1)
            target.title = "only in beta"
            target.uri = "test://beta/only"
            await beta.document_repository.update_meta(target)
        return config

    @pytest.mark.asyncio
    async def test_a_title_match_is_read_from_the_database_that_matched(self, tmp_path):
        config = await self._collided(tmp_path)

        async with HaikuRAG(config=config) as rag:
            by_title = await rag.resolve_document("only in beta")
            by_uri = await rag.resolve_document("test://beta/only")

        assert by_title is not None
        assert (by_title.source, by_title.title) == ("beta", "only in beta")
        assert by_uri is not None
        assert (by_uri.source, by_uri.uri) == ("beta", "test://beta/only")

    @pytest.mark.asyncio
    async def test_a_partial_match_is_read_from_the_database_that_matched(
        self, tmp_path
    ):
        from haiku.rag.tools.document import find_document

        config = await self._collided(tmp_path)

        async with HaikuRAG(config=config) as rag:
            by_uri = await find_document(rag, "beta/onl")
            by_title = await find_document(rag, "only in bet")

        assert by_uri is not None and by_uri.source == "beta"
        assert by_title is not None and by_title.source == "beta"

    @pytest.mark.asyncio
    async def test_a_source_is_checked_against_what_the_client_covers(self, tmp_path):
        """A lookup naming a database the client does not cover is wrong rather
        than answerable from the one it does cover."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])

        async with HaikuRAG(config=config, sources=["alpha"]) as alpha:
            [target] = await alpha.document_repository.list_all(limit=1)
            assert target.id is not None

            await alpha.document_item_repository.create_all(
                [
                    DocumentItem(
                        document_id=target.id,
                        self_ref="#/pictures/0",
                        position=0,
                        label="picture",
                        text="",
                        picture_data=b"alpha-picture",
                    )
                ]
            )

            [held] = await alpha.chunk_repository.list_all(limit=1)
            assert held.id is not None

            found = await alpha.get_document_by_id(target.id, "alpha")
            picture = await alpha.get_picture_bytes(target.id, "#/pictures/0", "alpha")
            chunk = await alpha.get_chunk_by_id(held.id, "alpha")
            with pytest.raises(UnknownDatabaseError):
                await alpha.get_document_by_id(target.id, "beta")
            with pytest.raises(UnknownDatabaseError):
                await alpha.get_picture_bytes(target.id, "#/pictures/0", "beta")
            with pytest.raises(UnknownDatabaseError):
                await alpha.get_chunk_by_id(held.id, "beta")

        assert found is not None and found.uri == "test://alpha/alpha one"
        assert picture == b"alpha-picture"
        assert chunk is not None and chunk.content == "alpha one"

    @pytest.mark.asyncio
    async def test_an_unnamed_database_answers_to_no_name(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True) as rag:
            docling = DoclingDocument(name="one")
            docling.add_text(label=DocItemLabel.TEXT, text="body")
            dim = get_config().embeddings.model.vector_dim
            doc = await rag.import_document(
                docling,
                [Chunk(content="body", embedding=[0.1] * dim, order=0)],
                uri="test://one",
            )
            assert doc.id is not None

            [held] = await rag.chunk_repository.list_all(limit=1)
            assert held.id is not None

            assert await rag.get_document_by_id(doc.id) is not None
            assert await rag.get_chunk_by_id(held.id) is not None
            with pytest.raises(UnknownDatabaseError):
                await rag.get_document_by_id(doc.id, "alpha")
            with pytest.raises(UnknownDatabaseError):
                await rag.get_picture_bytes(doc.id, "#/pictures/0", "alpha")
            with pytest.raises(UnknownDatabaseError):
                await rag.get_chunk_by_id(held.id, "alpha")

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


class TestWritesNameTheirDatabase:
    """A write returns the document it wrote, and it came from a database. A
    read of the same document names it, so the write has to as well."""

    @staticmethod
    def _doc(text: str):
        from docling_core.types.doc.document import DoclingDocument
        from docling_core.types.doc.labels import DocItemLabel

        doc = DoclingDocument(name=text)
        doc.add_text(label=DocItemLabel.TEXT, text=text)
        return doc

    @pytest.mark.asyncio
    async def test_import_names_the_database(self, tmp_path):
        config = _config(tmp_path, ["alpha"])
        dim = get_config().embeddings.model.vector_dim

        async with HaikuRAG(config=config, create=True, sources=["alpha"]) as rag:
            written = await rag.import_document(
                self._doc("cats"),
                [Chunk(content="cats", embedding=[0.1] * dim, order=0)],
                uri="test://alpha/cats",
            )
            assert written.id is not None
            read = await rag.get_document_by_id(written.id)

        assert written.source == "alpha"
        assert read is not None and read.source == written.source

    @pytest.mark.asyncio
    async def test_a_batch_import_names_every_document(self, tmp_path):
        from haiku.rag.client.documents import DocumentImport

        config = _config(tmp_path, ["alpha"])
        dim = get_config().embeddings.model.vector_dim

        async with HaikuRAG(config=config, create=True, sources=["alpha"]) as rag:
            written = await rag.import_documents(
                [
                    DocumentImport(
                        docling_document=self._doc(text),
                        chunks=[Chunk(content=text, embedding=[0.1] * dim, order=0)],
                        uri=f"test://alpha/{text}",
                    )
                    for text in ("cats", "dogs")
                ]
            )

        assert [d.source for d in written] == ["alpha", "alpha"]

    @pytest.mark.asyncio
    async def test_a_metadata_only_update_names_the_database(self, tmp_path):
        """Changing only metadata rewrites the row without re-chunking, so it
        never reaches the paths that name a document on the way through."""
        config = _config(tmp_path, ["alpha"])
        dim = get_config().embeddings.model.vector_dim

        async with HaikuRAG(config=config, create=True, sources=["alpha"]) as rag:
            stored = await rag.import_document(
                self._doc("cats"),
                [Chunk(content="cats", embedding=[0.1] * dim, order=0)],
                uri="test://alpha/cats",
            )
            assert stored.id is not None

            updated = await rag.update_document(
                stored.id, title="Cats", metadata={"k": "v"}
            )

        assert updated is not None
        assert updated.title == "Cats"
        assert updated.source == "alpha"

    @pytest.mark.asyncio
    async def test_the_revision_short_circuit_names_the_database(self, tmp_path):
        """`create_document_from_source` refreshes metadata in place when the
        revision is unchanged, returning the document it rewrote."""
        from haiku.rag.client.documents import _refresh_doc_metadata

        config = _config(tmp_path, ["alpha"])
        dim = get_config().embeddings.model.vector_dim

        async with HaikuRAG(config=config, create=True, sources=["alpha"]) as rag:
            stored = await rag.import_document(
                self._doc("cats"),
                [Chunk(content="cats", embedding=[0.1] * dim, order=0)],
                uri="test://alpha/cats",
            )
            assert isinstance(rag._session, SingleDatabaseSession)
            refreshed = await _refresh_doc_metadata(
                rag._session,
                stored,
                title="Cats",
                user_metadata={"k": "v"},
                source_metadata=None,
            )

        assert refreshed.source == "alpha"


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
        """`haiku-rag --db-name alpha list` opens one database, and its name is
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
