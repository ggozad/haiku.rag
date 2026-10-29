import asyncio
import hashlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic import ValidationError

from haiku.rag.client import HaikuRAG
from haiku.rag.config import get_config
from haiku.rag.config.models import AppConfig, HookConfig
from haiku.rag.hooks import ENTRY_POINT_GROUP, Hook, build_hooks
from haiku.rag.store.models.chunk import Chunk
from tests.client.test_client import _docling_doc, _import
from tests.platform import converts_pdf_locally, requires_symlinks


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


def test_config_accepts_a_hook_name_or_a_name_with_options():
    config = AppConfig.model_validate(
        {"hooks": ["plain", {"name": "tuned", "options": {"token": "syn"}}]}
    )
    assert config.hooks == [
        "plain",
        HookConfig(name="tuned", options={"token": "syn"}),
    ]
    assert HookConfig(name="bare").options == {}


def test_config_rejects_an_unknown_key_in_a_hook_entry():
    with pytest.raises(ValidationError, match="opts"):
        AppConfig.model_validate({"hooks": [{"name": "tuned", "opts": {}}]})


def test_build_hooks_passes_options_to_the_factory():
    discovered = {"append": _EntryPointStub(AppendTokenHook)}
    hooks = build_hooks(
        ["append", HookConfig(name="append", options={"token": "syn"})], discovered
    )
    assert len(hooks) == 2
    assert [h.token for h in hooks if isinstance(h, AppendTokenHook)] == [
        "expanded",
        "syn",
    ]


def test_build_hooks_unknown_name_with_options_raises():
    with pytest.raises(ValueError, match="missing"):
        build_hooks([HookConfig(name="missing", options={"x": 1})], {})


def test_build_hooks_names_a_hook_whose_factory_rejects_its_options():
    discovered = {"append": _EntryPointStub(AppendTokenHook)}
    with pytest.raises(ValueError, match="hook 'append': .*tokn") as excinfo:
        build_hooks([HookConfig(name="append", options={"tokn": "x"})], discovered)
    assert isinstance(excinfo.value.__cause__, TypeError)


def test_build_hooks_names_a_hook_whose_module_fails_to_import():
    class _UnimportableEntryPoint:
        def load(self):
            raise ImportError("No module named 'missing_dependency'")

    with pytest.raises(ValueError, match="hook 'broken': .*missing_dependency"):
        build_hooks(["broken"], {"broken": _UnimportableEntryPoint()})


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
    class ContentHook(Hook):
        contents: dict[str, str] = {}

        async def after_delete(self, client, event):
            self.contents = {d.uri or "": d.content for d in event.documents}

    spy = RecordingHook()
    content_spy = ContentHook()
    dim = get_config().embeddings.model.vector_dim

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [spy, content_spy]
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
        )
        child.metadata = {"parent_uri": "mem://parent"}
        child = await client.document_repository.update_meta(child)
        assert parent.id is not None and child.id is not None
        spy.events.clear()

        assert await client.delete_document(parent.id) is True

        # One event for the whole cascade, carrying the deleted documents'
        # last-known state (uri still resolvable).
        assert len(spy.events) == 1
        kind, deleted = spy.events[0]
        assert kind == "delete"
        assert set(deleted) == {(parent.id, "mem://parent"), (child.id, "mem://child")}
        assert content_spy.contents == {
            "mem://parent": "Parent body",
            "mem://child": "Child body",
        }


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


class CancelledTeardownHook(Hook):
    @asynccontextmanager
    async def lifespan(self, client):
        try:
            yield
        finally:
            raise asyncio.CancelledError


@pytest.mark.asyncio
async def test_a_cancelled_lifespan_teardown_still_closes_the_client(temp_db_path):
    client = HaikuRAG(temp_db_path, create=True)
    client._hooks = [CancelledTeardownHook()]

    with pytest.raises(asyncio.CancelledError):
        async with client:
            pass

    assert not client.store.db.is_open()


@pytest.mark.asyncio
async def test_a_cancelled_unwind_after_a_startup_failure_still_closes_the_client(
    temp_db_path,
):
    class _FailingStartHook(Hook):
        @asynccontextmanager
        async def lifespan(self, client):
            raise RuntimeError("cannot start")
            yield  # unreachable; asynccontextmanager needs a generator

    client = HaikuRAG(temp_db_path, create=True)
    client._hooks = [CancelledTeardownHook(), _FailingStartHook()]

    with pytest.raises(asyncio.CancelledError):
        async with client:
            pass

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
async def test_startup_failure_closes_a_reranker_a_lifespan_built(
    temp_db_path, monkeypatch
):
    closed: list[str] = []

    class _Reranker:
        async def aclose(self):
            closed.append("reranker")

    monkeypatch.setattr(
        "haiku.rag.client.client.get_reranker", lambda config: _Reranker()
    )

    class _RerankerUsingHook(Hook):
        @asynccontextmanager
        async def lifespan(self, client):
            assert client.reranker is not None
            yield

    class _FailingStartHook(Hook):
        @asynccontextmanager
        async def lifespan(self, client):
            raise RuntimeError("cannot start")
            yield  # unreachable; asynccontextmanager needs a generator

    client = HaikuRAG(temp_db_path, create=True)
    client._hooks = [_RerankerUsingHook(), _FailingStartHook()]

    with pytest.raises(RuntimeError, match="cannot start"):
        async with client:
            pass

    assert closed == ["reranker"]


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


def _md5(data: bytes) -> str:
    import hashlib

    return hashlib.md5(data, usedforsecurity=False).hexdigest()


class RewriteHook(Hook):
    """Rewrites "original" to "rewritten" in non-PDF bodies, tags every
    document, tries to forge a reserved key, and records each request."""

    def __init__(self):
        self.requests: list[tuple[str, str | None, str | None, bytes]] = []

    async def before_ingest(self, client, request):
        self.requests.append(
            (request.uri, request.source_id, request.content_type, request.body)
        )
        if request.content_type != "application/pdf":
            request.body = request.body.replace(b"original", b"rewritten")
        request.metadata["team"] = "hook"
        request.metadata["md5"] = "forged"
        return request


class ProviderStub:
    def __init__(self):
        self.bodies: list[bytes] = []

    async def __call__(self, source_id, uri, result):
        self.bodies.append(result.body)
        return {"team": "provider", "provided": "yes"}


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_before_ingest_rewrites_the_stored_document(temp_db_path, tmp_path):
    from haiku.rag.sources.fs import FSSource

    original = b"# Notes\n\nThe original text.\n"
    path = tmp_path / "doc.md"
    path.write_bytes(original)
    source = FSSource(root=tmp_path, source_id="fs:docs")
    hook = RewriteHook()
    provider = ProviderStub()

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [hook]
        doc = await client.create_document_from_source(
            path,
            metadata={"caller": "yes"},
            sources=[source],
            source_id=source.source_id,
            metadata_provider=provider,
        )

    assert not isinstance(doc, list)
    assert "rewritten text" in doc.content
    assert "original" not in doc.content
    assert hook.requests == [
        (doc.uri, "fs:docs", doc.metadata["content_type"], original)
    ]
    assert provider.bodies == [original]
    assert doc.metadata["md5"] == _md5(original.replace(b"original", b"rewritten"))
    assert doc.metadata["source_revision"] == str(path.stat().st_mtime_ns)
    assert doc.metadata["source_id"] == "fs:docs"
    assert doc.metadata["team"] == "hook"
    assert doc.metadata["provided"] == "yes"
    assert doc.metadata["caller"] == "yes"


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_before_ingest_metadata_wins_over_source_metadata(temp_db_path, tmp_path):
    import os

    from haiku.rag.sources.fs import FSSource

    class TaggingSource(FSSource):
        async def fetch(self, uri):
            result = await super().fetch(uri)
            result.extra_metadata["team"] = "source"
            return result

    path = tmp_path / "doc.md"
    path.write_bytes(b"# Notes\n\nThe original text.\n")
    source = TaggingSource(root=tmp_path, source_id="fs:docs")

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [RewriteHook()]
        doc = await client.create_document_from_source(
            path, sources=[source], source_id=source.source_id
        )
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        refreshed = await client.create_document_from_source(
            path, sources=[source], source_id=source.source_id
        )

    for stored in (doc, refreshed):
        assert not isinstance(stored, list)
        assert stored.metadata["team"] == "hook"
        assert stored.metadata["md5"] != "forged"
    assert not isinstance(refreshed, list)
    assert refreshed.metadata["source_revision"] == str(path.stat().st_mtime_ns)


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_before_ingest_output_short_circuits_unchanged_bytes(
    temp_db_path, tmp_path
):
    import os

    path = tmp_path / "doc.md"
    path.write_bytes(b"# Notes\n\nThe original text.\n")
    hook = RewriteHook()
    recording = RecordingHook()

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [hook, recording]
        await client.create_document_from_source(path)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        await client.create_document_from_source(path)

    assert len(hook.requests) == 2
    assert [event[0] for event in recording.events] == ["ingest"]


@converts_pdf_locally
@pytest.mark.asyncio
@pytest.mark.vcr()
@pytest.mark.usefixtures("docling_local_models")
async def test_before_ingest_runs_for_each_pdf_attachment(temp_db_path, tmp_path):
    from haiku.rag.sources.fs import FSSource
    from tests.conftest import build_pdf

    notes = b"The original notes."
    path = tmp_path / "parent.pdf"
    path.write_bytes(build_pdf([("notes.txt", notes)]))
    source = FSSource(root=tmp_path, source_id="fs:attachments")
    hook = RewriteHook()
    provider = ProviderStub()

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [hook]
        parent = await client.create_document_from_source(
            path,
            sources=[source],
            source_id=source.source_id,
            metadata_provider=provider,
        )
        assert not isinstance(parent, list)
        child_uri = f"{parent.uri}#attachment=notes.txt"
        child = await client.get_document_by_uri(child_uri)

    assert child is not None
    assert [(uri, source_id) for uri, source_id, _, _ in hook.requests] == [
        (parent.uri, "fs:attachments"),
        (child_uri, "fs:attachments"),
    ]
    assert hook.requests[1][3] == notes
    assert provider.bodies[1] == notes
    assert "rewritten notes" in child.content
    assert child.metadata["md5"] == _md5(b"The rewritten notes.")
    assert child.metadata["team"] == "hook"
    assert child.metadata["provided"] == "yes"
    assert "source_id" not in child.metadata
    assert parent.metadata["team"] == "hook"


class RaisingIngestHook(Hook):
    async def before_ingest(self, client, request):
        raise RuntimeError("hook refused")


@pytest.mark.asyncio
@pytest.mark.parametrize("uri", [None, "custom://notes"])
async def test_a_raising_before_ingest_stores_nothing(temp_db_path, tmp_path, uri):
    from haiku.rag.hooks import HookError

    path = tmp_path / "doc.md"
    path.write_bytes(b"# Notes\n\nBody.\n")

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [RaisingIngestHook()]
        with pytest.raises(HookError) as raised:
            await client.create_document_from_source(path, uri=uri)
        assert await client.list_documents() == []

    assert str(raised.value) == (
        "tests.test_hooks.RaisingIngestHook.before_ingest failed for "
        f"{uri or path.as_uri()}: RuntimeError: hook refused"
    )
    assert isinstance(raised.value.__cause__, RuntimeError)


class LookupIngestHook(Hook):
    """Looks up the stored document by the request's URI, then rejects."""

    def __init__(self):
        self.found: list = []

    async def before_ingest(self, client, request):
        from haiku.rag.hooks import DocumentRejected

        self.found.append(await client.get_document_by_uri(request.uri))
        raise DocumentRejected("looked up")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "through",
    [
        "explicit_uri",
        pytest.param("symlinked_path", marks=requires_symlinks),
        pytest.param("symlinked_file_uri", marks=requires_symlinks),
    ],
)
async def test_before_ingest_request_uri_finds_the_stored_document(
    temp_db_path, tmp_path, through
):
    from haiku.rag.hooks import DocumentRejected

    target = tmp_path / "target.md"
    target.write_bytes(b"# Notes\n\nBody.\n")
    if through == "explicit_uri":
        source: Path | str = target
        uri = stored_uri = "custom://notes"
    else:
        link = tmp_path / "link.md"
        link.symlink_to(target)
        source = link if through == "symlinked_path" else link.as_uri()
        uri, stored_uri = None, link.as_uri()
    hook = LookupIngestHook()

    async with HaikuRAG(temp_db_path, create=True) as client:
        stored = await client.import_document(
            _docling_doc("notes", "Body"),
            [
                Chunk(
                    content="Body",
                    embedding=[0.1] * get_config().embeddings.model.vector_dim,
                    order=0,
                )
            ],
            uri=stored_uri,
        )
        client._hooks = [hook]
        with pytest.raises(DocumentRejected) as raised:
            await client.create_document_from_source(source, uri=uri)

    assert [doc.id for doc in hook.found] == [stored.id]
    assert raised.value.uri == stored_uri


class NoReturnIngestHook(Hook):
    async def before_ingest(self, client, request):
        request.metadata["team"] = "hook"


class TextBodyIngestHook(Hook):
    async def before_ingest(self, client, request):
        request.body = request.body.decode()
        return request


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hook, error",
    [
        (NoReturnIngestHook(), "TypeError: returned NoneType"),
        (TextBodyIngestHook(), "TypeError: body is str"),
    ],
)
async def test_invalid_before_ingest_output_fails_as_a_hook_error(
    temp_db_path, tmp_path, hook, error
):
    from haiku.rag.hooks import HookError

    path = tmp_path / "doc.md"
    path.write_bytes(b"# Notes\n\nBody.\n")

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [hook]
        with pytest.raises(HookError) as raised:
            await client.create_document_from_source(path)
        assert await client.list_documents() == []

    assert f"before_ingest failed for {path.as_uri()}: {error}" in str(raised.value)


class DatetimeMetadataHook(Hook):
    async def before_ingest(self, client, request):
        from datetime import datetime

        request.metadata["seen_at"] = datetime(2026, 1, 1)
        return request


@pytest.mark.asyncio
async def test_before_ingest_metadata_that_is_not_json_fails_before_conversion(
    temp_db_path, tmp_path
):
    from haiku.rag.hooks import HookError

    path = tmp_path / "doc.md"
    path.write_bytes(b"# Notes\n\nBody.\n")

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [DatetimeMetadataHook()]
        with pytest.raises(HookError, match="DatetimeMetadataHook.*TypeError"):
            await client.create_document_from_source(path)
        assert await client.list_documents() == []


class ReservedDatetimeHook(Hook):
    async def before_ingest(self, client, request):
        from datetime import datetime

        request.metadata["source_revision"] = datetime(2026, 1, 1)
        return request


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_before_ingest_ignores_a_reserved_key_that_is_not_json(
    temp_db_path, tmp_path
):
    path = tmp_path / "doc.md"
    path.write_bytes(b"# Notes\n\nBody.\n")

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [ReservedDatetimeHook()]
        doc = await client.create_document_from_source(path)

    assert not isinstance(doc, list)
    assert doc.metadata["source_revision"] == str(path.stat().st_mtime_ns)


@pytest.mark.asyncio
async def test_full_rebuild_propagates_a_before_ingest_failure(temp_db_path, tmp_path):
    from haiku.rag.client import RebuildMode
    from haiku.rag.hooks import HookError

    path = tmp_path / "doc.md"
    path.write_bytes(b"# Notes\n\nBody.\n")

    async with HaikuRAG(temp_db_path, create=True) as client:
        [doc] = await client.import_documents(
            [_import("doc", "Body.", uri=path.as_uri())]
        )
        client._hooks = [RaisingIngestHook()]
        with pytest.raises(HookError):
            async for _ in client.rebuild_database(mode=RebuildMode.FULL):
                pass
        assert doc.id is not None
        stored = await client.get_document_by_id(doc.id)

    assert stored is not None
    assert stored.content == doc.content


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_full_rebuild_leaves_other_clients_hooks_firing(temp_db_path, tmp_path):
    from haiku.rag.client import RebuildMode

    path = tmp_path / "doc.md"
    path.write_bytes(b"# Notes\n\nThe original text.\n")
    recording = RecordingHook()

    async with HaikuRAG(tmp_path / "other.lancedb", create=True) as other:
        other._hooks = [recording]

        class _WritesElsewhere(Hook):
            async def before_ingest(self, client, request):
                await other.import_documents([_import("side", "Side note.")])
                return request

        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.import_documents([_import("doc", "Body.", uri=path.as_uri())])
            client._hooks = [_WritesElsewhere()]
            async for _ in client.rebuild_database(mode=RebuildMode.FULL):
                pass

    assert [event[0] for event in recording.events] == ["ingest"]


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_full_rebuild_runs_before_ingest_but_not_after_ingest(
    temp_db_path, tmp_path
):
    from haiku.rag.client import RebuildMode

    path = tmp_path / "doc.md"
    path.write_bytes(b"# Notes\n\nThe original text.\n")
    hook = RewriteHook()
    recording = RecordingHook()

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document_from_source(path)
        assert not isinstance(doc, list)
        assert "original text" in doc.content

        client._hooks = [hook, recording]
        rebuilt = [
            doc_id async for doc_id in client.rebuild_database(mode=RebuildMode.FULL)
        ]
        assert doc.id is not None
        refreshed = await client.get_document_by_id(doc.id)

    assert rebuilt == [doc.id]
    assert refreshed is not None
    assert "rewritten text" in refreshed.content
    assert refreshed.metadata["team"] == "hook"
    assert len(hook.requests) == 1
    assert recording.events == []


class RejectHook(Hook):
    """Rejects every document whose URI ends with ``suffix``."""

    def __init__(self, suffix: str):
        self.suffix = suffix

    async def before_ingest(self, client, request):
        from haiku.rag.hooks import DocumentRejected

        if request.uri.endswith(self.suffix):
            raise DocumentRejected("draft")
        return request


def test_a_rejection_reads_as_its_reason_until_ingestion_names_the_document():
    from haiku.rag.hooks import DocumentRejected

    assert str(DocumentRejected("draft")) == "draft"


@pytest.mark.asyncio
async def test_a_rejected_document_raises_and_stores_nothing(temp_db_path, tmp_path):
    from haiku.rag.hooks import DocumentRejected

    path = tmp_path / "doc.md"
    path.write_bytes(b"# Notes\n\nBody.\n")

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [RejectHook("doc.md")]
        with pytest.raises(DocumentRejected) as raised:
            await client.create_document_from_source(path)
        assert await client.list_documents() == []

    assert raised.value.reason == "draft"
    assert raised.value.uri == path.as_uri()
    assert str(raised.value) == f"{path.as_uri()} rejected: draft"


@pytest.mark.asyncio
async def test_a_directory_ingest_passes_a_rejected_file(
    temp_db_path, tmp_path, monkeypatch
):
    from tests.conftest import capture_logs, fake_ingest_fetch_result

    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result", fake_ingest_fetch_result
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_bytes(b"# A\n\nFirst.\n")
    (docs / "b.md").write_bytes(b"# B\n\nSecond.\n")

    with capture_logs(
        logging.getLogger("haiku.rag.client.documents"), logging.WARNING
    ) as records:
        async with HaikuRAG(temp_db_path, create=True) as client:
            client._hooks = [RejectHook("a.md")]
            stored = await client.create_document_from_source(docs)
            listed = await client.list_documents()

    assert isinstance(stored, list)
    assert [d.uri for d in stored] == [(docs / "b.md").as_uri()]
    assert [d.uri for d in listed] == [(docs / "b.md").as_uri()]
    assert [r.getMessage() for r in records] == [
        f"{(docs / 'a.md').as_uri()} rejected: draft"
    ]


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_full_rebuild_keeps_a_rejected_documents_stored_content(
    temp_db_path, tmp_path
):
    from haiku.rag.client import RebuildMode

    path = tmp_path / "doc.md"
    path.write_bytes(b"# Notes\n\nBody.\n")

    async with HaikuRAG(temp_db_path, create=True) as client:
        [doc] = await client.import_documents(
            [_import("doc", "Body.", uri=path.as_uri())]
        )
        client._hooks = [RejectHook("doc.md")]
        rebuilt = [
            doc_id async for doc_id in client.rebuild_database(mode=RebuildMode.FULL)
        ]
        assert doc.id is not None
        stored = await client.get_document_by_id(doc.id)

    assert rebuilt == [doc.id]
    assert stored is not None
    assert stored.content == doc.content


async def _reconcile_twice(client, first: bytes, second: bytes, **kwargs):
    """Reconcile a stored parent's attachments from ``first``, then ``second``."""
    from haiku.rag.client.documents import _reconcile_pdf_attachments
    from haiku.rag.store.models.document import Document
    from tests.conftest import writing

    parent = await client.document_repository.create(
        Document(
            content="",
            uri="file:///fixtures/parent.pdf",
            metadata={"content_type": "application/pdf"},
        )
    )
    await _reconcile_pdf_attachments(writing(client), client, parent, first, depth=0)
    await _reconcile_pdf_attachments(
        writing(client), client, parent, second, depth=0, **kwargs
    )


@pytest.mark.asyncio
async def test_removed_attachments_fire_one_after_delete(temp_db_path, monkeypatch):
    from tests.conftest import build_pdf, fake_ingest_fetch_result

    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result", fake_ingest_fetch_result
    )
    recording = RecordingHook()

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [recording]
        await _reconcile_twice(
            client,
            build_pdf([("a.txt", b"A"), ("b.txt", b"B"), ("c.txt", b"C")]),
            build_pdf([("a.txt", b"A")]),
        )

    [(kind, documents)] = recording.events
    assert kind == "delete"
    assert sorted(uri for _, uri in documents) == [
        "file:///fixtures/parent.pdf#attachment=b.txt",
        "file:///fixtures/parent.pdf#attachment=c.txt",
    ]


@pytest.mark.asyncio
async def test_removed_attachments_fire_nothing_without_notify_hooks(
    temp_db_path, monkeypatch
):
    from tests.conftest import build_pdf, fake_ingest_fetch_result

    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result", fake_ingest_fetch_result
    )
    recording = RecordingHook()

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [recording]
        await _reconcile_twice(
            client,
            build_pdf([("a.txt", b"A"), ("b.txt", b"B")]),
            build_pdf([("a.txt", b"A")]),
            notify_hooks=False,
        )
        remaining = await client.list_documents()

    assert recording.events == []
    assert len(remaining) == 2


@pytest.mark.asyncio
async def test_attachments_deleted_before_a_failure_still_fire_after_delete(
    temp_db_path, monkeypatch
):
    from haiku.rag.client.documents import _reconcile_pdf_attachments
    from haiku.rag.client.session import SingleDatabaseSession
    from tests.conftest import build_pdf, fake_ingest_fetch_result, writing

    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result", fake_ingest_fetch_result
    )
    real_delete = SingleDatabaseSession.delete_document
    calls = 0

    async def fail_second_delete(self, document_id, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        return await real_delete(self, document_id, **kwargs)

    monkeypatch.setattr(SingleDatabaseSession, "delete_document", fail_second_delete)
    recording = RecordingHook()
    three = build_pdf([("a.txt", b"A"), ("b.txt", b"B"), ("c.txt", b"C")])
    one = build_pdf([("a.txt", b"A")])

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [recording]
        with pytest.raises(OSError, match="disk full"):
            await _reconcile_twice(client, three, one)
        [first] = recording.events

        parent = await client.get_document_by_uri("file:///fixtures/parent.pdf")
        assert parent is not None
        await _reconcile_pdf_attachments(writing(client), client, parent, one, depth=0)

    [second] = recording.events[1:]
    assert first[0] == second[0] == "delete"
    assert len(first[1]) == len(second[1]) == 1
    assert {first[1][0][1], second[1][0][1]} == {
        "file:///fixtures/parent.pdf#attachment=b.txt",
        "file:///fixtures/parent.pdf#attachment=c.txt",
    }


@pytest.mark.asyncio
async def test_a_rejected_attachment_is_skipped_and_its_stored_copy_kept(
    temp_db_path, monkeypatch
):
    from haiku.rag.client.documents import (
        _reconcile_pdf_attachments,
        parent_uri_filter,
    )
    from haiku.rag.store.models.document import Document
    from tests.conftest import (
        build_pdf,
        capture_logs,
        fake_ingest_fetch_result,
        writing,
    )

    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result", fake_ingest_fetch_result
    )
    parent_uri = "file:///fixtures/parent.pdf"

    with capture_logs(
        logging.getLogger("haiku.rag.client.documents"), logging.WARNING
    ) as records:
        async with HaikuRAG(temp_db_path, create=True) as client:
            parent = await client.document_repository.create(
                Document(
                    content="",
                    uri=parent_uri,
                    metadata={"content_type": "application/pdf"},
                )
            )
            await _reconcile_pdf_attachments(
                writing(client),
                client,
                parent,
                build_pdf([("a.txt", b"A"), ("b.txt", b"B")]),
                depth=0,
            )
            client._hooks = [RejectHook("#attachment=a.txt")]
            await _reconcile_pdf_attachments(
                writing(client),
                client,
                parent,
                build_pdf([("a.txt", b"A2"), ("c.txt", b"C")]),
                depth=0,
            )
            children = await client.list_documents(filter=parent_uri_filter(parent_uri))

    by_uri = {c.uri: c for c in children}
    assert sorted(by_uri) == [
        f"{parent_uri}#attachment=a.txt",
        f"{parent_uri}#attachment=c.txt",
    ]
    assert by_uri[f"{parent_uri}#attachment=a.txt"].metadata["md5"] == (
        hashlib.md5(b"A").hexdigest()
    )
    assert [r.getMessage() for r in records] == [
        f"{parent_uri}#attachment=a.txt rejected: draft"
    ]
