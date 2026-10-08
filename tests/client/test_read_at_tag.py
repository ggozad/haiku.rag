from dataclasses import replace

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.client.scope import DatabaseRef, DatabaseScope
from haiku.rag.config.models import AppConfig, DatabaseEntry
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


async def test_client_at_a_tag_cannot_create(tmp_path):
    with pytest.raises(ValueError, match="'new' is read at tag 'tagged'.*create"):
        async with HaikuRAG(tmp_path / "new.lancedb", create=True, tag="tagged"):
            pass
    assert not (tmp_path / "new.lancedb").exists()


def _tagged_entry(config, name: str, entry: str, tag: str = "tagged") -> AppConfig:
    """`config` plus `entry`, reading database `name` at `tag`."""
    databases = dict(config.lancedb.databases)
    databases[entry] = DatabaseEntry(location=databases[name], tag=tag)
    return config.model_copy(
        update={"lancedb": config.lancedb.model_copy(update={"databases": databases})}
    )


async def test_a_configured_tag_reads_the_tagged_state(tmp_path):
    config = _config(tmp_path, ["papers"])
    await _tag_between(config, "papers")
    config = _tagged_entry(config, "papers", "before")

    rag = HaikuRAG(config=config, sources=["before"])
    assert rag.is_read_only
    async with rag:
        assert rag.is_read_only
        docs = await rag.list_documents()
        assert [d.uri for d in docs] == ["test://papers/before"]
        with pytest.raises(ReadOnlyError):
            await rag.delete_document(docs[0].id or "")


async def test_live_and_tagged_entries_are_read_together(tmp_path):
    """Each entry reads its own state, and a document both hold comes back
    once from each."""
    config = _config(tmp_path, ["papers"])
    await _tag_between(config, "papers")
    config = _tagged_entry(config, "papers", "before")

    async with HaikuRAG(config=config) as rag:
        assert rag.source_names == ("papers", "before")
        before = await rag.search("before", search_type="fts")
        after = await rag.search("after", search_type="fts")

    assert sorted(r.source or "" for r in before) == ["before", "papers"]
    assert [r.source for r in after] == ["papers"]


def test_a_second_tag_on_a_configured_entry_is_refused(tmp_path):
    config = _tagged_entry(_config(tmp_path, ["papers"]), "papers", "before")

    with pytest.raises(AmbiguousDatabaseError, match="'tagged'.*'other'"):
        HaikuRAG(config=config, sources=["before"], tag="other").source_names


async def test_a_missing_configured_tag_names_the_entry(tmp_path):
    config = _config(tmp_path, ["papers"])
    await _tag_between(config, "papers")
    config = _tagged_entry(config, "papers", "before", tag="nope")

    with pytest.raises(TagError, match="database 'before': Tag 'nope' does not"):
        async with HaikuRAG(config=config, sources=["before"]):
            pass


async def test_a_configured_tag_cannot_create(tmp_path):
    config = _tagged_entry(_config(tmp_path, ["papers"]), "papers", "before")

    with pytest.raises(ValueError, match="'before' is read at tag 'tagged'.*create"):
        async with HaikuRAG(config=config, sources=["before"], create=True):
            pass
