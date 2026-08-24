import logging
from contextlib import asynccontextmanager

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.config import get_config
from haiku.rag.hooks import ENTRY_POINT_GROUP, Hook, build_hooks
from haiku.rag.store.models.chunk import Chunk
from tests.test_client import _docling_doc, _import


class RecordingHook(Hook):
    def __init__(self):
        self.events: list[tuple] = []

    async def after_ingest(self, client, event):
        self.events.append(
            ("ingest", event.operation, tuple((d.id, d.uri) for d in event.documents))
        )

    async def after_delete(self, client, event):
        self.events.append(("delete", tuple((d.id, d.uri) for d in event.documents)))


class AppendTokenHook(Hook):
    def __init__(self, token: str = "expanded"):
        self.token = token

    async def before_search(self, client, request):
        request.query = f"{request.query} {self.token}"
        return request


class FilterHook(Hook):
    async def before_search(self, client, request):
        request.filter = "uri = 'mem://hooked'"
        return request


class ClearSearchTypeHook(Hook):
    async def before_search(self, client, request):
        request.search_type = None
        return request


class SpyAfterSearchHook(Hook):
    def __init__(self):
        self.search_types: list[str | None] = []

    async def after_search(self, client, request, results):
        self.search_types.append(request.search_type)
        return results


class ReverseResultsHook(Hook):
    async def after_search(self, client, request, results):
        self.seen_query = request.query
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
    config = get_config().model_copy(deep=True)
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
    config = get_config().model_copy(deep=True)
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
        captured["search_type"] = search_type
        return []

    client.chunk_repository.search = fake_search
    return captured


@pytest.mark.asyncio
async def test_before_search_hooks_chain_in_order(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        spy = SpyBeforeSearchHook()
        client._hooks = [
            AppendTokenHook("one"),
            AppendTokenHook("two"),
            FilterHook(),
            spy,
        ]
        captured = await _capture_repo_search(client)

        await client.search("alpha")

        assert captured["query"] == "alpha one two"
        assert captured["filter"] == "uri = 'mem://hooked'"
        # The request carries the resolved search parameters.
        assert spy.requests == [("alpha one two", "hybrid", get_config().search.limit)]


class SpyBeforeSearchHook(Hook):
    def __init__(self):
        self.requests: list[tuple] = []

    async def before_search(self, client, request):
        self.requests.append((request.query, request.search_type, request.limit))
        return request


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
            return [0.1] * get_config().embeddings.model.vector_dim

        client.store.embedder.embed_image = fake_embed_image
        client.store.embedder.supports_images = True

        await client.search(b"image-bytes")

        assert hook.requests == []


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
    dim = get_config().embeddings.model.vector_dim

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [spy]

        doc = await client.import_document(
            _docling_doc("a", "Alpha body"),
            [Chunk(content="Alpha body", embedding=[0.1] * dim, order=0)],
            uri="mem://a",
            title="Alpha",
        )
        assert spy.events == [("ingest", "create", ((doc.id, "mem://a"),))]

        # A batch import arrives as one event carrying all documents.
        spy.events.clear()
        batch = await client.import_documents(
            [
                _import("b", "Beta body", uri="mem://b", title="Beta"),
                _import("c", "Gamma body", uri="mem://c", title="Gamma"),
            ]
        )
        assert spy.events == [
            (
                "ingest",
                "create",
                ((batch[0].id, "mem://b"), (batch[1].id, "mem://c")),
            )
        ]

        spy.events.clear()
        assert doc.id is not None
        await client.update_document(
            doc.id,
            docling_document=_docling_doc("a2", "Alpha updated"),
            chunks=[Chunk(content="Alpha updated", embedding=[0.2] * dim, order=0)],
        )
        assert spy.events == [("ingest", "update", ((doc.id, "mem://a"),))]

        # Creation against an already-stored URI updates in place.
        spy.events.clear()
        await client.import_document(
            _docling_doc("a3", "Alpha again"),
            [Chunk(content="Alpha again", embedding=[0.3] * dim, order=0)],
            uri="mem://a",
            title="Alpha",
        )
        assert spy.events == [("ingest", "update", ((doc.id, "mem://a"),))]


@pytest.mark.asyncio
async def test_metadata_only_update_does_not_fire_after_ingest(temp_db_path):
    spy = RecordingHook()
    dim = get_config().embeddings.model.vector_dim

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
    dim = get_config().embeddings.model.vector_dim

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

        # One event for the whole cascade, carrying the deleted documents'
        # last-known state (uri still resolvable).
        assert len(spy.events) == 1
        kind, deleted = spy.events[0]
        assert kind == "delete"
        assert set(deleted) == {(parent.id, "mem://parent"), (child.id, "mem://child")}


class AnnotateHook(Hook):
    async def after_search(self, client, request, results):
        for result in results:
            result.annotations = ["XMT: transmit"]
        return results


@pytest.mark.asyncio
async def test_after_search_hook_annotations_render_for_agent(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [AnnotateHook()]

        async def fake_search(query, limit, search_type=None, filter=None, **kwargs):
            return [(Chunk(id="c1", content="XMT lamp check", order=0), 0.9)]

        client.chunk_repository.search = fake_search

        results = await client.search("lamp", include_images=False)

        assert results[0].annotations == ["XMT: transmit"]
        assert "Note: XMT: transmit" in results[0].format_for_agent()


def test_format_for_agent_without_annotations_has_no_notes():
    from haiku.rag.store.models.chunk import SearchResult

    result = SearchResult(content="plain", score=0.5)
    assert "Note:" not in result.format_for_agent()


@pytest.mark.asyncio
async def test_annotations_survive_context_expansion(temp_db_path):
    from haiku.rag.context import expand_with_items, window_for
    from haiku.rag.store.models.chunk import SearchResult
    from haiku.rag.store.models.document_item import DocumentItem

    async with HaikuRAG(temp_db_path, create=True) as client:
        items = [
            DocumentItem(
                document_id="doc-1",
                position=i,
                self_ref=f"#/texts/{i}",
                label="text",
                text=f"Paragraph {i}. " * 10,
            )
            for i in range(5)
        ]
        await client.document_item_repository.create_items("doc-1", items)

        r1 = SearchResult(
            content="Paragraph 1.",
            score=0.9,
            chunk_id="c1",
            document_id="doc-1",
            doc_item_refs=["#/texts/1"],
            annotations=["XMT: transmit", "shared note"],
        )
        r2 = SearchResult(
            content="Paragraph 3.",
            score=0.85,
            chunk_id="c2",
            document_id="doc-1",
            doc_item_refs=["#/texts/3"],
            annotations=["RCV: receive", "shared note"],
        )

        repo = client.document_item_repository
        positions = (
            await repo.resolve_refs_grouped(
                {"doc-1": [ref for r in (r1, r2) for ref in r.doc_item_refs]}
            )
        )["doc-1"]
        window_items = (
            await repo.get_items_in_ranges({"doc-1": window_for(positions)})
        )["doc-1"]
        expanded = expand_with_items([r1, r2], 5000, positions, window_items)

        assert len(expanded) == 1
        assert expanded[0].annotations == [
            "XMT: transmit",
            "shared note",
            "RCV: receive",
        ]


class ThrowingHook(Hook):
    async def after_ingest(self, client, event):
        raise RuntimeError("ingest hook boom")

    async def after_delete(self, client, event):
        raise RuntimeError("delete hook boom")

    async def before_search(self, client, request):
        raise RuntimeError("search hook boom")


@pytest.mark.asyncio
async def test_after_ingest_hook_failure_is_logged_not_raised(temp_db_path, caplog):
    spy = RecordingHook()
    dim = get_config().embeddings.model.vector_dim

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [ThrowingHook(), spy]

        with caplog.at_level(logging.ERROR, logger="haiku.rag.hooks"):
            doc = await client.import_document(
                _docling_doc("a", "Alpha body"),
                [Chunk(content="Alpha body", embedding=[0.1] * dim, order=0)],
                uri="mem://a",
                title="Alpha",
            )

        assert doc.id is not None
        stored = await client.get_document_by_id(doc.id)
        assert stored is not None

        # Subsequent hooks still run after a failing one.
        assert spy.events == [("ingest", "create", ((doc.id, "mem://a"),))]

        record = next(r for r in caplog.records if "after_ingest" in r.message)
        assert "tests.test_hooks.ThrowingHook" in record.message
        assert str(doc.id) in record.message


@pytest.mark.asyncio
async def test_after_delete_hook_failure_is_logged_not_raised(temp_db_path, caplog):
    spy = RecordingHook()
    dim = get_config().embeddings.model.vector_dim

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.import_document(
            _docling_doc("a", "Alpha body"),
            [Chunk(content="Alpha body", embedding=[0.1] * dim, order=0)],
            uri="mem://a",
            title="Alpha",
        )
        assert doc.id is not None
        client._hooks = [ThrowingHook(), spy]

        with caplog.at_level(logging.ERROR, logger="haiku.rag.hooks"):
            assert await client.delete_document(doc.id) is True

        assert await client.get_document_by_id(doc.id) is None
        assert spy.events == [("delete", ((doc.id, "mem://a"),))]

        record = next(r for r in caplog.records if "after_delete" in r.message)
        assert "tests.test_hooks.ThrowingHook" in record.message
        assert str(doc.id) in record.message


@pytest.mark.asyncio
async def test_before_search_hook_failure_propagates(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [ThrowingHook()]

        with pytest.raises(RuntimeError, match="search hook boom"):
            await client.search("alpha")


@pytest.mark.asyncio
async def test_delete_missing_document_fires_nothing(temp_db_path):
    spy = RecordingHook()
    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [spy]
        assert await client.delete_document("does-not-exist") is False
        assert spy.events == []


class LifespanHook(Hook):
    """Records lifespan transitions into a shared log, so ordering across
    several hooks is observable."""

    def __init__(self, name: str, log: list[str]):
        self.name = name
        self.log = log

    @asynccontextmanager
    async def lifespan(self, client):
        self.log.append(f"enter {self.name}")
        try:
            yield
        finally:
            self.log.append(f"exit {self.name}")


class ExceptionRecordingHook(Hook):
    """Records whatever exception its lifespan exit was told about, then lets
    it continue on its way."""

    def __init__(self, seen: list[str]):
        self.seen = seen

    @asynccontextmanager
    async def lifespan(self, client):
        try:
            yield
        except Exception as exc:
            self.seen.append(str(exc))
            raise


@pytest.mark.asyncio
async def test_lifespans_enter_in_order_and_exit_in_reverse(temp_db_path):
    log: list[str] = []
    client = HaikuRAG(temp_db_path, create=True)
    client._hooks = [LifespanHook("a", log), LifespanHook("b", log)]

    async with client:
        assert log == ["enter a", "enter b"]

    assert log == ["enter a", "enter b", "exit b", "exit a"]


@pytest.mark.asyncio
async def test_lifespan_can_use_the_store_on_entry_and_exit(temp_db_path):
    counts: list[int] = []

    class _StoreUsingHook(Hook):
        @asynccontextmanager
        async def lifespan(self, client):
            counts.append(len(await client.list_documents()))
            try:
                yield
            finally:
                counts.append(len(await client.list_documents()))

    client = HaikuRAG(temp_db_path, create=True)
    client._hooks = [_StoreUsingHook()]
    dim = get_config().embeddings.model.vector_dim

    async with client:
        await client.import_document(
            _docling_doc("a", "Alpha body"),
            [Chunk(content="Alpha body", embedding=[0.1] * dim, order=0)],
            uri="mem://lifespan",
            title="Alpha",
        )

    assert counts == [0, 1]


@pytest.mark.asyncio
async def test_lifespan_startup_failure_unwinds_started_hooks(temp_db_path):
    log: list[str] = []

    class _FailingStartHook(Hook):
        @asynccontextmanager
        async def lifespan(self, client):
            raise RuntimeError("cannot start")
            yield  # unreachable; asynccontextmanager needs a generator

    client = HaikuRAG(temp_db_path, create=True)
    client._hooks = [LifespanHook("a", log), _FailingStartHook()]

    with pytest.raises(RuntimeError, match="cannot start"):
        async with client:
            pass

    assert log == ["enter a", "exit a"]
    assert not client.store.db.is_open()


@pytest.mark.asyncio
async def test_lifespan_teardown_failure_is_logged_and_suppressed(temp_db_path, caplog):
    log: list[str] = []

    class _FailingExitHook(Hook):
        @asynccontextmanager
        async def lifespan(self, client):
            yield
            raise RuntimeError("cannot stop")

    client = HaikuRAG(temp_db_path, create=True)
    client._hooks = [LifespanHook("a", log), _FailingExitHook()]

    with caplog.at_level(logging.ERROR, logger="haiku.rag.hooks"):
        async with client:
            pass

    # The surviving hook still exits, and teardown does not raise.
    assert log == ["enter a", "exit a"]
    assert any("lifespan" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_lifespans_see_the_exception_being_unwound(temp_db_path):
    seen: list[str] = []

    client = HaikuRAG(temp_db_path, create=True)
    client._hooks = [ExceptionRecordingHook(seen)]

    with pytest.raises(ValueError, match="from the body"):
        async with client:
            raise ValueError("from the body")

    assert seen == ["from the body"]


@pytest.mark.asyncio
async def test_a_swallowing_lifespan_hides_nothing_from_anyone(temp_db_path):
    """A hook that eats the exception in its own teardown must neither
    suppress it for the caller nor make the hooks unwound after it believe
    the shutdown was clean."""
    seen: list[str] = []

    class _SwallowingHook(Hook):
        @asynccontextmanager
        async def lifespan(self, client):
            try:
                yield
            except Exception:
                pass

    client = HaikuRAG(temp_db_path, create=True)
    # The swallowing hook is entered last, so it unwinds first.
    client._hooks = [ExceptionRecordingHook(seen), _SwallowingHook()]

    with pytest.raises(ValueError, match="from the body"):
        async with client:
            raise ValueError("from the body")

    assert seen == ["from the body"]


@pytest.mark.asyncio
async def test_default_lifespan_is_a_noop(temp_db_path):
    spy = RecordingHook()
    client = HaikuRAG(temp_db_path, create=True)
    client._hooks = [spy]
    dim = get_config().embeddings.model.vector_dim

    async with client:
        doc = await client.import_document(
            _docling_doc("a", "Alpha body"),
            [Chunk(content="Alpha body", embedding=[0.1] * dim, order=0)],
            uri="mem://default-lifespan",
            title="Alpha",
        )

    # A hook that overrides no lifespan still reaches its other hook points.
    assert spy.events == [("ingest", "create", ((doc.id, "mem://default-lifespan"),))]


@pytest.mark.asyncio
async def test_startup_failure_is_forwarded_to_started_lifespans(temp_db_path):
    """A hook that fails to start is an unwind like any other: the lifespans
    already running are told what went wrong, not handed a clean shutdown."""
    seen: list[str] = []

    class _FailingStartHook(Hook):
        @asynccontextmanager
        async def lifespan(self, client):
            raise RuntimeError("cannot start")
            yield  # unreachable; asynccontextmanager needs a generator

    client = HaikuRAG(temp_db_path, create=True)
    client._hooks = [ExceptionRecordingHook(seen), _FailingStartHook()]

    with pytest.raises(RuntimeError, match="cannot start"):
        async with client:
            pass

    assert seen == ["cannot start"]


@pytest.mark.asyncio
async def test_after_search_sees_the_search_type_that_ran(temp_db_path):
    """A before_search hook may leave search_type unset. Retrieval falls back
    to hybrid, so the request after_search reads must say hybrid too."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        spy = SpyAfterSearchHook()
        client._hooks = [ClearSearchTypeHook(), spy]
        captured = await _capture_repo_search(client)

        await client.search("alpha", include_images=False)

        assert captured["search_type"] == "hybrid"
        assert spy.search_types == ["hybrid"]


@pytest.mark.asyncio
async def test_after_search_reports_vector_for_image_queries(temp_db_path):
    """Image queries run vector-only whatever the caller asked for, so the
    request must not still be advertising the caller's choice."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        spy = SpyAfterSearchHook()
        client._hooks = [spy]

        async def fake_search(query, limit, search_type=None, filter=None, **kwargs):
            return []

        client.chunk_repository.search = fake_search

        async def fake_embed_image(image):
            return [0.1] * get_config().embeddings.model.vector_dim

        client.store.embedder.embed_image = fake_embed_image
        client.store.embedder.supports_images = True

        await client.search(b"image-bytes", search_type="fts", include_images=False)

        assert spy.search_types == ["vector"]
