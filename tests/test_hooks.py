import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.hooks import ENTRY_POINT_GROUP, Hook, build_hooks
from haiku.rag.store.models.chunk import Chunk
from tests.test_client import _docling_doc, _import


class RecordingHook(Hook):
    def __init__(self):
        self.events: list[tuple] = []

    async def after_ingest(self, client, document):
        self.events.append(("ingest", document.id, document.uri))

    async def after_delete(self, client, document_id):
        self.events.append(("delete", document_id))


class AppendTokenHook(Hook):
    def __init__(self, token: str = "expanded"):
        self.token = token

    async def before_search(self, client, query, filter):
        return f"{query} {self.token}", filter


class FilterHook(Hook):
    async def before_search(self, client, query, filter):
        return query, "uri = 'mem://hooked'"


class ReverseResultsHook(Hook):
    async def after_search(self, client, query, results):
        self.seen_query = query
        return list(reversed(results))


class _EntryPointStub:
    def __init__(self, factory):
        self._factory = factory

    def load(self):
        return self._factory


class _BrokenEntryPoint:
    def load(self):
        raise AssertionError("unreferenced entry point must not be loaded")


def test_build_hooks_unknown_name_raises():
    with pytest.raises(ValueError, match=ENTRY_POINT_GROUP):
        build_hooks(["missing"], {})


def test_build_hooks_loads_lazily_in_configured_order():
    discovered = {
        "recording": _EntryPointStub(RecordingHook),
        "append": _EntryPointStub(AppendTokenHook),
        "broken": _BrokenEntryPoint(),
    }
    hooks = build_hooks(["append", "recording"], discovered)
    assert [type(h) for h in hooks] == [AppendTokenHook, RecordingHook]


def test_client_init_unknown_hook_raises(temp_db_path):
    config = Config.model_copy(deep=True)
    config.hooks = ["missing"]
    with pytest.raises(ValueError, match="missing"):
        HaikuRAG(temp_db_path, config=config, create=True)


def test_client_builds_hooks_from_entry_points(temp_db_path, monkeypatch):
    class _NamedEntryPoint:
        name = "recording"

        def load(self):
            return RecordingHook

    def fake_entry_points(group):
        assert group == ENTRY_POINT_GROUP
        return [_NamedEntryPoint()]

    monkeypatch.setattr("haiku.rag.hooks.entry_points", fake_entry_points)
    config = Config.model_copy(deep=True)
    config.hooks = ["recording"]
    client = HaikuRAG(temp_db_path, config=config, create=True)
    assert len(client._hooks) == 1
    assert isinstance(client._hooks[0], RecordingHook)


def test_client_without_hooks_builds_none(temp_db_path):
    client = HaikuRAG(temp_db_path, create=True)
    assert client._hooks == []


async def _capture_repo_search(client):
    captured = {}

    async def fake_search(query, limit, search_type=None, filter=None, **kwargs):
        captured["query"] = query
        captured["filter"] = filter
        return []

    client.chunk_repository.search = fake_search
    return captured


@pytest.mark.asyncio
async def test_before_search_hooks_chain_in_order(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [AppendTokenHook("one"), AppendTokenHook("two"), FilterHook()]
        captured = await _capture_repo_search(client)

        await client.search("alpha")

        assert captured["query"] == "alpha one two"
        assert captured["filter"] == "uri = 'mem://hooked'"


class SpyBeforeSearchHook(Hook):
    def __init__(self):
        self.called: list[str] = []

    async def before_search(self, client, query, filter):
        self.called.append(query)
        return query, filter


@pytest.mark.asyncio
async def test_before_search_skips_non_text_queries(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        hook = SpyBeforeSearchHook()
        client._hooks = [hook]

        captured = {}

        async def fake_search(query, limit, search_type=None, filter=None, **kwargs):
            captured["query"] = query
            return []

        client.chunk_repository.search = fake_search

        async def fake_embed_image(image):
            return [0.1] * Config.embeddings.model.vector_dim

        client.store.embedder.embed_image = fake_embed_image
        client.store.embedder.supports_images = True

        await client.search(b"image-bytes")

        assert hook.called == []


@pytest.mark.asyncio
async def test_after_search_transforms_results(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        hook = ReverseResultsHook()
        client._hooks = [hook]

        chunks = [
            (Chunk(id="c1", content="first", document_id="d1", order=0), 0.9),
            (Chunk(id="c2", content="second", document_id="d1", order=1), 0.5),
        ]

        async def fake_search(query, limit, search_type=None, filter=None, **kwargs):
            return chunks

        client.chunk_repository.search = fake_search

        results = await client.search("alpha", include_images=False)

        assert [r.content for r in results] == ["second", "first"]
        assert hook.seen_query == "alpha"


@pytest.mark.asyncio
async def test_after_ingest_fires_on_import_batch_update(temp_db_path):
    spy = RecordingHook()
    dim = Config.embeddings.model.vector_dim

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [spy]

        doc = await client.import_document(
            _docling_doc("a", "Alpha body"),
            [Chunk(content="Alpha body", embedding=[0.1] * dim, order=0)],
            uri="mem://a",
            title="Alpha",
        )
        assert spy.events == [("ingest", doc.id, "mem://a")]

        spy.events.clear()
        batch = await client.import_documents(
            [
                _import("b", "Beta body", uri="mem://b", title="Beta"),
                _import("c", "Gamma body", uri="mem://c", title="Gamma"),
            ]
        )
        assert spy.events == [
            ("ingest", batch[0].id, "mem://b"),
            ("ingest", batch[1].id, "mem://c"),
        ]

        spy.events.clear()
        assert doc.id is not None
        await client.update_document(
            doc.id,
            docling_document=_docling_doc("a2", "Alpha updated"),
            chunks=[Chunk(content="Alpha updated", embedding=[0.2] * dim, order=0)],
        )
        assert spy.events == [("ingest", doc.id, "mem://a")]


@pytest.mark.asyncio
async def test_metadata_only_update_does_not_fire_after_ingest(temp_db_path):
    spy = RecordingHook()
    dim = Config.embeddings.model.vector_dim

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [spy]
        doc = await client.import_document(
            _docling_doc("a", "Alpha body"),
            [Chunk(content="Alpha body", embedding=[0.1] * dim, order=0)],
            uri="mem://a",
            title="Alpha",
        )
        assert doc.id is not None
        spy.events.clear()

        await client.update_document(doc.id, title="Renamed")

        assert spy.events == []


@pytest.mark.asyncio
async def test_after_delete_fires_for_cascade(temp_db_path):
    spy = RecordingHook()
    dim = Config.embeddings.model.vector_dim

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [spy]
        parent = await client.import_document(
            _docling_doc("p", "Parent body"),
            [Chunk(content="Parent body", embedding=[0.1] * dim, order=0)],
            uri="mem://parent",
            title="Parent",
        )
        child = await client.import_document(
            _docling_doc("k", "Child body"),
            [Chunk(content="Child body", embedding=[0.1] * dim, order=0)],
            uri="mem://child",
            title="Child",
            metadata={"parent_uri": "mem://parent"},
        )
        assert parent.id is not None and child.id is not None
        spy.events.clear()

        assert await client.delete_document(parent.id) is True

        deleted = {event[1] for event in spy.events}
        assert deleted == {parent.id, child.id}
        assert all(event[0] == "delete" for event in spy.events)


@pytest.mark.asyncio
async def test_delete_missing_document_fires_nothing(temp_db_path):
    spy = RecordingHook()
    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [spy]
        assert await client.delete_document("does-not-exist") is False
        assert spy.events == []
