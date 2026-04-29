import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, MonitorConfig, S3MonitorEntry
from haiku.rag.store.models.document import Document


@pytest.fixture
def s3_listing(monkeypatch):
    """Patch `obstore.list_obs` with an async-iterator returning controllable batches.

    Returns `(set_batches, list_mock)`. `set_batches([[meta, ...], ...])`
    seeds the next call's pages.
    """
    import obstore

    batches: list[list[MagicMock]] = []

    def list_obs(_store, *_, **__):
        async def _iter():
            for batch in batches:
                yield batch

        return _iter()

    list_mock = MagicMock(side_effect=list_obs)
    monkeypatch.setattr(obstore, "list", list_mock)

    def set_batches(new_batches):
        batches.clear()
        batches.extend(new_batches)

    return set_batches, list_mock


def _meta(path: str, etag: str) -> dict:
    # Real obstore ObjectMeta is a TypedDict; raw S3 ETags include quotes.
    return {
        "path": path,
        "e_tag": f'"{etag}"',
        "size": 0,
        "last_modified": None,
    }


def _entry(**kwargs) -> S3MonitorEntry:
    return S3MonitorEntry(
        uri=kwargs.pop("uri", "s3://my-bucket/incoming/"),
        poll_interval=kwargs.pop("poll_interval", 60),
        delete_orphans=kwargs.pop("delete_orphans", False),
        ignore_patterns=kwargs.pop("ignore_patterns", []),
        include_patterns=kwargs.pop("include_patterns", []),
        storage_options=kwargs.pop("storage_options", {}),
        **kwargs,
    )


def _doc(uri: str, etag: str, doc_id: str | None = None) -> Document:
    return Document(
        id=doc_id or uri,
        content="...",
        uri=uri,
        metadata={"etag": etag, "md5": "deadbeef"},
    )


@pytest.mark.asyncio
async def test_s3_watcher_refresh_upserts_new_objects(s3_listing):
    set_batches, _ = s3_listing
    set_batches([[_meta("incoming/a.txt", "abc"), _meta("incoming/b.txt", "def")]])

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
async def test_s3_watcher_skips_unchanged_etag(s3_listing):
    set_batches, _ = s3_listing
    set_batches([[_meta("incoming/a.txt", "abc")]])

    from haiku.rag.monitor import S3Watcher

    rag = AsyncMock(spec=HaikuRAG)
    rag.list_documents.return_value = [_doc("s3://my-bucket/incoming/a.txt", "abc")]

    watcher = S3Watcher(client=rag, entry=_entry(), supported_extensions=[".txt"])
    await watcher.refresh()

    rag.create_document_from_source.assert_not_awaited()


@pytest.mark.asyncio
async def test_s3_watcher_upserts_when_etag_differs(s3_listing):
    set_batches, _ = s3_listing
    set_batches([[_meta("incoming/a.txt", "new")]])

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
async def test_s3_watcher_strips_etag_quotes(s3_listing):
    set_batches, _ = s3_listing
    set_batches([[_meta("incoming/a.txt", "abc")]])

    from haiku.rag.monitor import S3Watcher

    rag = AsyncMock(spec=HaikuRAG)
    rag.list_documents.return_value = [
        _doc("s3://my-bucket/incoming/a.txt", "abc")  # already stripped in storage
    ]

    watcher = S3Watcher(client=rag, entry=_entry(), supported_extensions=[".txt"])
    await watcher.refresh()

    rag.create_document_from_source.assert_not_awaited()


@pytest.mark.asyncio
async def test_s3_watcher_deletes_orphans_when_enabled(s3_listing):
    set_batches, _ = s3_listing
    set_batches([[_meta("incoming/a.txt", "abc")]])

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
async def test_s3_watcher_does_not_delete_orphans_when_disabled(s3_listing):
    set_batches, _ = s3_listing
    set_batches([[]])

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
async def test_s3_watcher_orphan_scope_is_per_entry(s3_listing):
    """A doc under a different bucket prefix must not be touched."""
    set_batches, _ = s3_listing
    set_batches([[]])

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
async def test_s3_watcher_applies_include_and_ignore_patterns(s3_listing):
    set_batches, _ = s3_listing
    set_batches(
        [
            [
                _meta("incoming/keep.md", "1"),
                _meta("incoming/draft.md", "2"),
                _meta("incoming/skip.txt", "3"),
            ]
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
async def test_s3_watcher_observe_survives_transient_list_failure(s3_listing):
    """First refresh succeeds; second refresh raises; loop survives and recovers."""
    set_batches, list_mock = s3_listing

    pages_initial = [[_meta("incoming/a.txt", "abc")]]
    pages_after = [[_meta("incoming/a.txt", "abc")]]

    paginate_calls = {"n": 0}

    def list_obs_side_effect(_store, *_, **__):
        paginate_calls["n"] += 1
        if paginate_calls["n"] == 2:
            raise RuntimeError("transient list failure")

        async def _iter():
            for batch in pages_after if paginate_calls["n"] > 1 else pages_initial:
                yield batch

        return _iter()

    list_mock.side_effect = list_obs_side_effect

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
async def test_serve_starts_one_s3_task_per_entry(monkeypatch, s3_listing):
    """`serve` wires one S3Watcher task per MonitorConfig.s3 entry."""
    from haiku.rag import app as app_module

    original_create_task = asyncio.create_task

    def tracking_create_task(coro, *args, **kwargs):
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

    assert fw_observe_calls["n"] == 1
    assert s3_observe_calls["n"] == 2
