import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from haiku.rag.sources.base import FileTooLargeError, SourceEventKind
from haiku.rag.sources.s3 import S3Source


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


def _get_result(data: bytes, etag: str | None = '"abc123"') -> MagicMock:
    result = MagicMock()
    result.meta = {"e_tag": etag, "size": len(data)}
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


async def test_head_returns_etag(fake_obstore_io):
    head_async, _ = fake_obstore_io
    head_async.return_value = {"e_tag": '"abc123"'}
    src = S3Source(uri="s3://bucket/")
    assert await src.head("s3://bucket/file.txt") == "abc123"
    head_async.assert_awaited_once()


async def test_head_returns_none_when_no_etag(fake_obstore_io):
    head_async, _ = fake_obstore_io
    head_async.return_value = {}
    src = S3Source(uri="s3://bucket/")
    assert await src.head("s3://bucket/file.txt") is None


async def test_fetch_returns_bytes_md5_etag(fake_obstore_io):
    head_async, get_async = fake_obstore_io
    body = b"S3 hosted content"
    head_async.return_value = {"e_tag": '"different-revision"'}
    get_async.return_value = _get_result(body)

    src = S3Source(uri="s3://bucket/")
    result = await src.fetch("s3://bucket/folder/file.txt")

    assert result.uri == "s3://bucket/folder/file.txt"
    assert result.body == body
    assert result.content_hash == hashlib.md5(body, usedforsecurity=False).hexdigest()
    assert result.content_type == "text/plain"
    assert result.revision == "abc123"
    assert result.extra_metadata == {}
    head_async.assert_not_awaited()
    get_async.assert_awaited_once()


@pytest.mark.parametrize("etag", [None, "", '""'])
async def test_fetch_handles_missing_etag(fake_obstore_io, etag):
    head_async, get_async = fake_obstore_io
    head_async.return_value = {"e_tag": '"different-revision"'}
    get_async.return_value = _get_result(b"x", etag)

    src = S3Source(uri="s3://bucket/")
    result = await src.fetch("s3://bucket/file.txt")
    assert result.revision is None
    assert result.extra_metadata == {}


async def test_fetch_invalid_uri_raises(fake_obstore_io):
    src = S3Source(uri="s3://bucket/")
    with pytest.raises(ValueError):
        await src.fetch("s3:///no-bucket")


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


async def test_discover_unchanged_against_matching_snapshot(fake_s3_listing):
    fake_s3_listing([[_meta("file1.md", "abc")]])
    src = S3Source(uri="s3://bucket/", supported_extensions=[".md"])
    events = [e async for e in src.discover(since={"s3://bucket/file1.md": "abc"})]
    assert len(events) == 1
    assert events[0].kind is SourceEventKind.UNCHANGED


async def test_discover_emits_delete_for_missing_keys(fake_s3_listing):
    fake_s3_listing([[_meta("file1.md", "abc")]])
    src = S3Source(uri="s3://bucket/", supported_extensions=[".md"])
    events = [e async for e in src.discover(known_uris={"s3://bucket/gone.md"})]
    deletes = [e for e in events if e.kind is SourceEventKind.DELETE]
    assert len(deletes) == 1
    assert deletes[0].uri == "s3://bucket/gone.md"


async def test_discover_respects_prefix(fake_s3_listing):
    fake_s3_listing([[_meta("incoming/file1.md", "abc")]])
    src = S3Source(uri="s3://bucket/incoming/", supported_extensions=[".md"])
    events = [e async for e in src.discover()]
    assert len(events) == 1
    assert events[0].uri == "s3://bucket/incoming/file1.md"


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


@pytest.mark.parametrize(
    "known_uris, kind",
    [
        ({"s3://bucket/file.md"}, SourceEventKind.UNCHANGED),
        (None, SourceEventKind.UPSERT),
    ],
)
async def test_discover_without_etag_upserts_only_unknown_keys(
    fake_s3_listing, known_uris, kind
):
    """An S3 object with no ETag should not cause re-ingestion every sweep
    once the key has been ingested. A new key still UPSERTs on first sight."""
    fake_s3_listing([[{"path": "file.md", "size": 0, "last_modified": None}]])
    src = S3Source(uri="s3://bucket/", supported_extensions=[".md"])
    events = [e async for e in src.discover(known_uris=known_uris)]
    assert [e.kind for e in events] == [kind]


@pytest.mark.parametrize("body, fits", [(b"x" * 5000, False), (b"small", True)])
async def test_fetch_enforces_max_size(fake_obstore_io, body, fits):
    head_async, get_async = fake_obstore_io
    head_async.return_value = {"e_tag": '"abc"', "size": 5}
    response = _get_result(body)
    get_async.return_value = response

    src = S3Source(uri="s3://bucket/", max_file_size=1000)
    if fits:
        assert (await src.fetch("s3://bucket/file.txt")).body == body
    else:
        with pytest.raises(
            FileTooLargeError, match="file size 5000 bytes exceeds limit of 1000 bytes"
        ):
            await src.fetch("s3://bucket/file.txt")
        response.bytes_async.assert_not_awaited()
