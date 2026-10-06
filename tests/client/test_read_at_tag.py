from dataclasses import replace

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.client.scope import DatabaseRef, DatabaseScope
from haiku.rag.store import AmbiguousDatabaseError, ReadOnlyError, TagError
from tests.multi_db.helpers import _config, _seed


async def _tag_between(config, name: str) -> None:
    """`before` in the database, the tag, then `after`."""
    await _seed(config, name, ["before"])
    async with HaikuRAG(config=config, sources=[name]) as rag:
        await rag.store.create_tag("tagged")
    await _seed(config, name, ["after"])


async def test_client_reads_the_tagged_state(tmp_path):
    config = _config(tmp_path, ["papers"])
    await _tag_between(config, "papers")

    async with HaikuRAG(config=config, tag="tagged") as rag:
        assert rag.is_read_only
        docs = await rag.list_documents()
        assert [d.uri for d in docs] == ["test://papers/before"]
        results = await rag.search("after", search_type="fts")
        assert results == []
        with pytest.raises(ReadOnlyError):
            await rag.delete_document(docs[0].id or "")


async def test_client_at_a_path_reads_the_tagged_state(tmp_path):
    config = _config(tmp_path, ["papers"])
    await _tag_between(config, "papers")

    async with HaikuRAG(tmp_path / "papers.lancedb", tag="tagged") as rag:
        assert [d.uri for d in await rag.list_documents()] == ["test://papers/before"]


async def test_client_at_a_tag_over_a_set_raises(tmp_path):
    config = _config(tmp_path, ["papers", "wiki"])
    await _tag_between(config, "papers")
    await _seed(config, "wiki", ["wiki"])

    with pytest.raises(AmbiguousDatabaseError, match="papers, wiki"):
        async with HaikuRAG(config=config, tag="tagged"):
            pass
    async with HaikuRAG(config=config, sources=["papers"], tag="tagged") as rag:
        assert len(await rag.list_documents()) == 1


async def test_client_over_a_tagged_scope_is_read_only(tmp_path):
    config = _config(tmp_path, ["papers"])
    await _tag_between(config, "papers")
    ref = replace(DatabaseRef.at(tmp_path / "papers.lancedb"), tag="tagged")

    rag = HaikuRAG._covering(DatabaseScope((ref,)), config)
    assert rag.is_read_only
    async with rag:
        assert rag.is_read_only
        assert [d.uri for d in await rag.list_documents()] == ["test://papers/before"]


async def test_client_at_a_missing_tag_names_the_database(tmp_path):
    config = _config(tmp_path, ["papers"])
    await _tag_between(config, "papers")

    with pytest.raises(TagError, match="database 'papers': Tag 'nope' does not"):
        async with HaikuRAG(config=config, tag="nope"):
            pass


def test_client_at_a_tag_cannot_create(tmp_path):
    with pytest.raises(ValueError, match="create"):
        HaikuRAG(tmp_path / "new.lancedb", create=True, tag="tagged")
