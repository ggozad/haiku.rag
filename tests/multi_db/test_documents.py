"""Listing and looking up documents across databases."""

import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from haiku.rag.client import HaikuRAG
from haiku.rag.config import get_config
from haiku.rag.store.models import Chunk
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
