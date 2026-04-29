import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, MonitorConfig, S3MonitorEntry
from haiku.rag.store.models.document import Document


@pytest.fixture
def fake_aioboto3(monkeypatch):
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "aioboto3", fake)
    return fake


@pytest.fixture
def s3_paginator():
    """Return (paginator_mock, set_pages) — `set_pages([page, ...])` rewires the iterator."""
    pages: list[dict] = []

    async def _paginate(**_kwargs):
        for page in pages:
            yield page

    paginator = MagicMock()
    paginator.paginate.side_effect = lambda **kw: _paginate(**kw)

    def set_pages(new_pages):
        pages.clear()
        pages.extend(new_pages)

    return paginator, set_pages


@pytest.fixture
def fake_s3_client(fake_aioboto3, s3_paginator):
    paginator, set_pages = s3_paginator
    s3_client = MagicMock()
    s3_client.get_paginator.return_value = paginator

    client_ctx = AsyncMock()
    client_ctx.__aenter__.return_value = s3_client
    client_ctx.__aexit__.return_value = None

    session = MagicMock()
    session.client.return_value = client_ctx
    fake_aioboto3.Session.return_value = session

    return s3_client, set_pages


def _entry(**kwargs):
    base = {
        "uri": "s3://my-bucket/incoming/",
        "poll_interval": 60,
        "delete_orphans": False,
        "ignore_patterns": [],
        "include_patterns": [],
        "storage_options": {},
    }
    base.update(kwargs)
    return S3MonitorEntry(**base)


def _doc(uri: str, etag: str, doc_id: str | None = None) -> Document:
    return Document(
        id=doc_id or uri,
        content="...",
        uri=uri,
        metadata={"etag": etag, "md5": "deadbeef"},
    )


@pytest.mark.asyncio
async def test_s3_watcher_refresh_upserts_new_objects(fake_s3_client):
    s3, set_pages = fake_s3_client
    set_pages(
        [
            {
                "Contents": [
                    {"Key": "incoming/a.txt", "ETag": '"abc"'},
                    {"Key": "incoming/b.txt", "ETag": '"def"'},
                ]
            }
        ]
    )

    from haiku.rag.monitor import S3Watcher

    rag = AsyncMock(spec=HaikuRAG)
    rag.list_documents.return_value = []
    rag.create_document_from_source.return_value = Document(
        id="x", content="...", uri="s3://my-bucket/incoming/a.txt"
    )

    watcher = S3Watcher(
        client=rag, entry=_entry(), supported_extensions=[".txt", ".md", ".pdf"]
    )
    await watcher.refresh()

    assert rag.create_document_from_source.await_count == 2
    called_uris = {c.args[0] for c in rag.create_document_from_source.await_args_list}
    assert called_uris == {
        "s3://my-bucket/incoming/a.txt",
        "s3://my-bucket/incoming/b.txt",
    }


@pytest.mark.asyncio
async def test_s3_watcher_skips_unchanged_etag(fake_s3_client):
    s3, set_pages = fake_s3_client
    set_pages([{"Contents": [{"Key": "incoming/a.txt", "ETag": '"abc"'}]}])

    from haiku.rag.monitor import S3Watcher

    rag = AsyncMock(spec=HaikuRAG)
    rag.list_documents.return_value = [_doc("s3://my-bucket/incoming/a.txt", "abc")]

    watcher = S3Watcher(client=rag, entry=_entry(), supported_extensions=[".txt"])
    await watcher.refresh()

    rag.create_document_from_source.assert_not_awaited()


@pytest.mark.asyncio
async def test_s3_watcher_upserts_when_etag_differs(fake_s3_client):
    s3, set_pages = fake_s3_client
    set_pages([{"Contents": [{"Key": "incoming/a.txt", "ETag": '"new"'}]}])

    from haiku.rag.monitor import S3Watcher

    rag = AsyncMock(spec=HaikuRAG)
    rag.list_documents.return_value = [_doc("s3://my-bucket/incoming/a.txt", "old")]
    rag.create_document_from_source.return_value = Document(
        id="x", content="...", uri="s3://my-bucket/incoming/a.txt"
    )

    watcher = S3Watcher(client=rag, entry=_entry(), supported_extensions=[".txt"])
    await watcher.refresh()

    rag.create_document_from_source.assert_awaited_once_with(
        "s3://my-bucket/incoming/a.txt", storage_options={}
    )


@pytest.mark.asyncio
async def test_s3_watcher_strips_etag_quotes(fake_s3_client):
    s3, set_pages = fake_s3_client
    set_pages([{"Contents": [{"Key": "incoming/a.txt", "ETag": '"abc"'}]}])

    from haiku.rag.monitor import S3Watcher

    rag = AsyncMock(spec=HaikuRAG)
    rag.list_documents.return_value = [
        _doc("s3://my-bucket/incoming/a.txt", "abc")  # already stripped in storage
    ]

    watcher = S3Watcher(client=rag, entry=_entry(), supported_extensions=[".txt"])
    await watcher.refresh()

    rag.create_document_from_source.assert_not_awaited()


@pytest.mark.asyncio
async def test_s3_watcher_deletes_orphans_when_enabled(fake_s3_client):
    s3, set_pages = fake_s3_client
    set_pages([{"Contents": [{"Key": "incoming/a.txt", "ETag": '"abc"'}]}])

    from haiku.rag.monitor import S3Watcher

    a_doc = _doc("s3://my-bucket/incoming/a.txt", "abc", doc_id="a-id")
    orphan = _doc("s3://my-bucket/incoming/old.txt", "stale", doc_id="orphan-id")

    rag = AsyncMock(spec=HaikuRAG)
    rag.list_documents.return_value = [a_doc, orphan]
    rag.get_document_by_uri.return_value = orphan

    watcher = S3Watcher(
        client=rag,
        entry=_entry(delete_orphans=True),
        supported_extensions=[".txt"],
    )
    await watcher.refresh()

    rag.delete_document.assert_awaited_once_with("orphan-id")


@pytest.mark.asyncio
async def test_s3_watcher_does_not_delete_orphans_when_disabled(fake_s3_client):
    s3, set_pages = fake_s3_client
    set_pages([{"Contents": []}])

    from haiku.rag.monitor import S3Watcher

    orphan = _doc("s3://my-bucket/incoming/old.txt", "stale", doc_id="orphan-id")

    rag = AsyncMock(spec=HaikuRAG)
    rag.list_documents.return_value = [orphan]

    watcher = S3Watcher(
        client=rag,
        entry=_entry(delete_orphans=False),
        supported_extensions=[".txt"],
    )
    await watcher.refresh()

    rag.delete_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_s3_watcher_orphan_scope_is_per_entry(fake_s3_client):
    """A doc under a different bucket prefix must not be touched."""
    s3, set_pages = fake_s3_client
    set_pages([{"Contents": []}])

    from haiku.rag.monitor import S3Watcher

    rag = AsyncMock(spec=HaikuRAG)
    rag.list_documents.return_value = []  # filter scopes to my-bucket

    watcher = S3Watcher(
        client=rag,
        entry=_entry(delete_orphans=True),
        supported_extensions=[".txt"],
    )
    await watcher.refresh()

    rag.list_documents.assert_awaited_once()
    filter_kwarg = rag.list_documents.await_args.kwargs["filter"]
    assert filter_kwarg == "uri LIKE 's3://my-bucket/incoming/%'"


@pytest.mark.asyncio
async def test_s3_watcher_applies_include_and_ignore_patterns(fake_s3_client):
    s3, set_pages = fake_s3_client
    set_pages(
        [
            {
                "Contents": [
                    {"Key": "incoming/keep.md", "ETag": '"1"'},
                    {"Key": "incoming/draft.md", "ETag": '"2"'},
                    {"Key": "incoming/skip.txt", "ETag": '"3"'},
                ]
            }
        ]
    )

    from haiku.rag.monitor import S3Watcher

    rag = AsyncMock(spec=HaikuRAG)
    rag.list_documents.return_value = []
    rag.create_document_from_source.return_value = Document(
        id="x", content="...", uri="s3://my-bucket/incoming/keep.md"
    )

    watcher = S3Watcher(
        client=rag,
        entry=_entry(include_patterns=["*.md"], ignore_patterns=["draft*"]),
        supported_extensions=[".md", ".txt"],
    )
    await watcher.refresh()

    assert rag.create_document_from_source.await_count == 1
    assert (
        rag.create_document_from_source.await_args.args[0]
        == "s3://my-bucket/incoming/keep.md"
    )


@pytest.mark.asyncio
async def test_s3_watcher_observe_survives_transient_list_failure(fake_s3_client):
    """First refresh succeeds; second refresh raises; loop survives and recovers."""
    s3, set_pages = fake_s3_client

    pages_initial = [{"Contents": [{"Key": "incoming/a.txt", "ETag": '"abc"'}]}]
    pages_after = [{"Contents": [{"Key": "incoming/a.txt", "ETag": '"abc"'}]}]

    set_pages(pages_initial)
    paginate_calls = {"n": 0}

    async def paginate_side_effect(**_kwargs):
        paginate_calls["n"] += 1
        if paginate_calls["n"] == 2:
            raise RuntimeError("transient list failure")
        for page in pages_after if paginate_calls["n"] > 1 else pages_initial:
            yield page

    s3.get_paginator.return_value.paginate.side_effect = paginate_side_effect

    from haiku.rag.monitor import S3Watcher

    rag = AsyncMock(spec=HaikuRAG)
    rag.list_documents.return_value = []
    rag.create_document_from_source.return_value = Document(
        id="x", content="...", uri="s3://my-bucket/incoming/a.txt"
    )

    watcher = S3Watcher(
        client=rag,
        entry=_entry(poll_interval=0),
        supported_extensions=[".txt"],
    )
    task = asyncio.create_task(watcher.observe())

    # Let three iterations run: initial refresh, transient failure, recovery.
    for _ in range(20):
        await asyncio.sleep(0)
        if paginate_calls["n"] >= 3:
            break

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert paginate_calls["n"] >= 3  # loop kept going past the failure


@pytest.mark.asyncio
async def test_s3_watcher_invalid_uri_rejected():
    from haiku.rag.monitor import S3Watcher

    rag = AsyncMock(spec=HaikuRAG)
    with pytest.raises(ValueError, match="Invalid S3 monitor URI"):
        S3Watcher(
            client=rag,
            entry=S3MonitorEntry(uri="s3://"),
            supported_extensions=[".txt"],
        )


@pytest.mark.asyncio
async def test_serve_starts_one_s3_task_per_entry(monkeypatch, fake_s3_client):
    """`serve` wires one S3Watcher task per MonitorConfig.s3 entry."""
    from haiku.rag import app as app_module

    captured_tasks: list = []
    original_create_task = asyncio.create_task

    def tracking_create_task(coro, *args, **kwargs):
        captured_tasks.append(coro)
        return original_create_task(coro, *args, **kwargs)

    monkeypatch.setattr(app_module.asyncio, "create_task", tracking_create_task)

    config = AppConfig(
        monitor=MonitorConfig(
            s3=[
                S3MonitorEntry(uri="s3://bucket-a/x/"),
                S3MonitorEntry(uri="s3://bucket-b/y/"),
            ]
        )
    )

    fw_observe_calls = {"n": 0}

    async def fake_fw_observe(self):
        fw_observe_calls["n"] += 1

    monkeypatch.setattr(app_module.FileWatcher, "observe", fake_fw_observe)

    s3_observe_calls = {"n": 0}

    async def fake_s3_observe(self):
        s3_observe_calls["n"] += 1

    monkeypatch.setattr(app_module.S3Watcher, "observe", fake_s3_observe)

    # Provide a dummy supported_extensions to skip docling import.
    class _Conv:
        supported_extensions = [".txt"]

    monkeypatch.setattr("haiku.rag.converters.get_converter", lambda cfg: _Conv())

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "db.lancedb"
        app = app_module.HaikuRAGApp(db_path=db_path, config=config)
        async with HaikuRAG(db_path, config=config, create=True):
            pass  # create the database

        await app.serve(enable_monitor=True, enable_mcp=False)

    # Both S3 entries should have triggered observe(), plus the FileWatcher.
    assert fw_observe_calls["n"] == 1
    assert s3_observe_calls["n"] == 2
