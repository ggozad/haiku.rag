import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from haiku.rag.client import HaikuRAG


@pytest.fixture
def fake_obstore_io(monkeypatch):
    """Patch `obstore.head_async` and `obstore.get_async` with controllable mocks.

    Returns a tuple `(head_async, get_async)`. Tests assign `.return_value` to
    each to seed responses. Real `obstore.store.S3Store` is left intact —
    constructing the store still exercises the storage_options path.
    """
    import obstore

    head_async = AsyncMock()
    get_async = AsyncMock()
    monkeypatch.setattr(obstore, "head_async", head_async)
    monkeypatch.setattr(obstore, "get_async", get_async)
    return head_async, get_async


def _meta(etag: str) -> dict:
    return {"e_tag": etag, "path": "ignored", "size": 0, "last_modified": None}


def _get_result(data: bytes) -> MagicMock:
    result = MagicMock()
    result.bytes_async = AsyncMock(return_value=data)
    return result


def test_make_s3_store_accepts_lancedb_keys():
    from haiku.rag.s3 import make_s3_store

    store = make_s3_store(
        "my-bucket",
        {
            "endpoint": "http://seaweed:8333",
            "region": "us-east-1",
            "aws_access_key_id": "AKIA",
            "aws_secret_access_key": "secret",
            "allow_http": "true",
        },
    )
    assert store is not None  # construction must not raise


def test_make_s3_store_no_options_uses_default_chain():
    from haiku.rag.s3 import make_s3_store

    store = make_s3_store("my-bucket", None)
    assert store is not None


def test_make_s3_store_missing_obstore_raises_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "obstore.store", None)

    from haiku.rag.s3 import make_s3_store

    with pytest.raises(ImportError, match=r"haiku\.rag-slim\[s3\]"):
        make_s3_store("my-bucket", {})


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_create_document_from_s3_new(fake_obstore_io, temp_db_path):
    head_async, get_async = fake_obstore_io
    text = b"S3 hosted content"
    head_async.return_value = _meta('"abc123"')
    get_async.return_value = _get_result(text)

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document_from_source("s3://my-bucket/folder/file.txt")

    assert doc.uri == "s3://my-bucket/folder/file.txt"
    assert doc.metadata["etag"] == "abc123"  # quotes stripped
    assert doc.metadata["md5"]  # real content MD5
    assert doc.metadata["md5"] != "abc123"
    head_async.assert_awaited_once()
    get_async.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_create_document_from_s3_skips_when_etag_unchanged(
    fake_obstore_io, temp_db_path
):
    head_async, get_async = fake_obstore_io
    text = b"S3 hosted content"
    head_async.return_value = _meta('"abc123"')
    get_async.return_value = _get_result(text)

    async with HaikuRAG(temp_db_path, create=True) as client:
        first = await client.create_document_from_source("s3://my-bucket/file.txt")

        # Second call with the same ETag must not GET.
        get_async.reset_mock()
        get_async.return_value = _get_result(text)  # re-arm just in case
        second = await client.create_document_from_source("s3://my-bucket/file.txt")

    assert second.id == first.id
    assert head_async.await_count == 2
    get_async.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_create_document_from_s3_etag_changed_md5_same_skips_rechunk(
    fake_obstore_io, temp_db_path
):
    """Multipart re-upload of same content: etag changes, MD5 doesn't.

    Expected: GET runs to verify, but no re-chunk; only metadata.etag updates.
    """
    head_async, get_async = fake_obstore_io
    text = b"S3 hosted content"
    head_async.return_value = _meta('"abc123"')
    get_async.return_value = _get_result(text)

    async with HaikuRAG(temp_db_path, create=True) as client:
        first = await client.create_document_from_source("s3://my-bucket/file.txt")
        original_md5 = first.metadata["md5"]
        original_updated_at = first.updated_at

        # Same bytes, different ETag (multipart re-upload).
        head_async.return_value = _meta('"def456-2"')
        get_async.return_value = _get_result(text)

        second = await client.create_document_from_source("s3://my-bucket/file.txt")

    assert second.id == first.id
    assert second.metadata["md5"] == original_md5
    assert second.metadata["etag"] == "def456-2"
    assert second.updated_at >= original_updated_at
    assert get_async.await_count == 2  # initial create + etag-changed compare


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_create_document_from_s3_etag_changed_md5_changed_rechunks(
    fake_obstore_io, temp_db_path
):
    head_async, get_async = fake_obstore_io
    head_async.return_value = _meta('"abc123"')
    get_async.return_value = _get_result(b"original text")

    async with HaikuRAG(temp_db_path, create=True) as client:
        first = await client.create_document_from_source("s3://my-bucket/file.txt")

        head_async.return_value = _meta('"new999"')
        get_async.return_value = _get_result(b"different text now")

        second = await client.create_document_from_source("s3://my-bucket/file.txt")

    assert second.id == first.id
    assert second.metadata["md5"] != first.metadata["md5"]
    assert second.metadata["etag"] == "new999"
    assert "different text now" in second.content


@pytest.mark.asyncio
async def test_create_document_from_s3_rejects_invalid_uri(
    fake_obstore_io, temp_db_path
):
    async with HaikuRAG(temp_db_path, create=True) as client:
        with pytest.raises(ValueError, match="Invalid S3 URI"):
            await client.create_document_from_source("s3://only-bucket-no-key")
