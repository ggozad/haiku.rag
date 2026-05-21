import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from haiku.rag.ingester.sources.base import SourceEventKind
from haiku.rag.ingester.sources.s3 import S3Source


@pytest.fixture
def fake_obstore_io(monkeypatch):
    import obstore

    head_async = AsyncMock()
    get_async = AsyncMock()
    monkeypatch.setattr(obstore, "head_async", head_async)
    monkeypatch.setattr(obstore, "get_async", get_async)
    return head_async, get_async


@pytest.fixture
def fake_s3_listing(monkeypatch):
    import obstore

    batches: list[list[dict]] = []

    def list_obs(_store, *_, **__):
        async def _iter():
            for batch in batches:
                yield batch

        return _iter()

    monkeypatch.setattr(obstore, "list", MagicMock(side_effect=list_obs))

    def set_batches(new_batches):
        batches.clear()
        batches.extend(new_batches)

    return set_batches


def _meta(path: str, etag: str = "abc") -> dict:
    return {"path": path, "e_tag": f'"{etag}"', "size": 0, "last_modified": None}


def _get_result(data: bytes) -> MagicMock:
    result = MagicMock()
    result.bytes_async = AsyncMock(return_value=data)
    return result


def test_supports_s3_only():
    src = S3Source(uri="s3://bucket/")
    assert src.supports("s3://bucket/a.md")
    assert not src.supports("https://example.com/a.md")
    assert not src.supports("file:///tmp/a.md")


def test_supports_scopes_to_prefix():
    src = S3Source(uri="s3://bucket/incoming/")
    assert src.supports("s3://bucket/incoming/a.md")
    assert not src.supports("s3://bucket/other/a.md")


def test_source_id_uses_bucket_and_prefix():
    assert S3Source(uri="s3://bucket/prefix/").source_id == "s3:bucket/prefix/"
    assert S3Source(uri="s3://bucket/").source_id == "s3:bucket/"


def test_invalid_uri_raises():
    with pytest.raises(ValueError):
        S3Source(uri="https://example.com/")
    with pytest.raises(ValueError):
        S3Source(uri="s3:///key")


@pytest.mark.asyncio
async def test_fetch_returns_bytes_md5_etag(fake_obstore_io):
    head_async, get_async = fake_obstore_io
    body = b"S3 hosted content"
    head_async.return_value = {"e_tag": '"abc123"'}
    get_async.return_value = _get_result(body)

    src = S3Source(uri="s3://bucket/")
    result = await src.fetch("s3://bucket/folder/file.txt")

    assert result.uri == "s3://bucket/folder/file.txt"
    assert result.body == body
    assert result.content_hash == hashlib.md5(body, usedforsecurity=False).hexdigest()
    assert result.content_type == "text/plain"
    assert result.revision == "abc123"
    assert result.extra_metadata["etag"] == "abc123"
    head_async.assert_awaited_once()
    get_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_handles_missing_etag(fake_obstore_io):
    head_async, get_async = fake_obstore_io
    head_async.return_value = {}
    get_async.return_value = _get_result(b"x")

    src = S3Source(uri="s3://bucket/")
    result = await src.fetch("s3://bucket/file.txt")
    assert result.revision is None
    assert "etag" not in result.extra_metadata


@pytest.mark.asyncio
async def test_fetch_invalid_uri_raises(fake_obstore_io):
    src = S3Source(uri="s3://bucket/")
    with pytest.raises(ValueError):
        await src.fetch("s3:///no-bucket")


@pytest.mark.asyncio
async def test_discover_yields_upsert_for_new_keys(fake_s3_listing):
    fake_s3_listing(
        [
            [
                _meta("file1.md", "abc"),
                _meta("subfolder/file2.md", "def"),
            ]
        ]
    )

    src = S3Source(uri="s3://bucket/", supported_extensions=[".md"])
    events = [e async for e in src.discover()]
    uris = {e.uri for e in events}
    assert uris == {
        "s3://bucket/file1.md",
        "s3://bucket/subfolder/file2.md",
    }
    assert all(e.kind is SourceEventKind.UPSERT for e in events)
    assert {e.revision for e in events} == {"abc", "def"}


@pytest.mark.asyncio
async def test_discover_unchanged_against_matching_snapshot(fake_s3_listing):
    fake_s3_listing([[_meta("file1.md", "abc")]])
    src = S3Source(uri="s3://bucket/", supported_extensions=[".md"])
    events = [e async for e in src.discover(since={"s3://bucket/file1.md": "abc"})]
    assert len(events) == 1
    assert events[0].kind is SourceEventKind.UNCHANGED


@pytest.mark.asyncio
async def test_discover_emits_delete_for_missing_keys(fake_s3_listing):
    fake_s3_listing([[_meta("file1.md", "abc")]])
    src = S3Source(uri="s3://bucket/", supported_extensions=[".md"])
    events = [e async for e in src.discover(since={"s3://bucket/gone.md": "old"})]
    deletes = [e for e in events if e.kind is SourceEventKind.DELETE]
    assert len(deletes) == 1
    assert deletes[0].uri == "s3://bucket/gone.md"


@pytest.mark.asyncio
async def test_discover_respects_prefix(fake_s3_listing):
    fake_s3_listing([[_meta("incoming/file1.md", "abc")]])
    src = S3Source(uri="s3://bucket/incoming/", supported_extensions=[".md"])
    events = [e async for e in src.discover()]
    assert len(events) == 1
    assert events[0].uri == "s3://bucket/incoming/file1.md"


@pytest.mark.asyncio
async def test_discover_respects_extension_filter(fake_s3_listing):
    fake_s3_listing(
        [
            [
                _meta("a.md", "x"),
                _meta("b.log", "y"),
            ]
        ]
    )
    src = S3Source(uri="s3://bucket/", supported_extensions=[".md"])
    events = [e async for e in src.discover()]
    assert {e.uri for e in events} == {"s3://bucket/a.md"}


@pytest.mark.asyncio
async def test_discover_respects_ignore_patterns(fake_s3_listing):
    fake_s3_listing(
        [
            [
                _meta("a.md", "x"),
                _meta("draft-b.md", "y"),
            ]
        ]
    )
    src = S3Source(
        uri="s3://bucket/",
        supported_extensions=[".md"],
        ignore_patterns=["draft-*"],
    )
    events = [e async for e in src.discover()]
    assert {e.uri for e in events} == {"s3://bucket/a.md"}
