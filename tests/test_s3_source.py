import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from haiku.rag.client import HaikuRAG


@pytest.fixture
def fake_aioboto3(monkeypatch):
    """Install a fake aioboto3 module in sys.modules.

    Returns the module itself; callers configure `Session` to control behavior.
    """
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "aioboto3", fake)
    return fake


@pytest.fixture
def fake_s3_client(fake_aioboto3):
    """Configure fake_aioboto3.Session to return a controllable S3 client.

    Returns a MagicMock representing the S3 client (head_object, get_object).
    Tests override its return values per-case.
    """
    s3_client = MagicMock()
    s3_client.head_object = AsyncMock()
    s3_client.get_object = AsyncMock()

    client_ctx = AsyncMock()
    client_ctx.__aenter__.return_value = s3_client
    client_ctx.__aexit__.return_value = None

    session = MagicMock()
    session.client.return_value = client_ctx

    fake_aioboto3.Session.return_value = session
    return s3_client


def _streaming_body(data: bytes) -> AsyncMock:
    body = AsyncMock()
    body.read.return_value = data
    return body


def test_make_s3_session_translates_lancedb_keys(fake_aioboto3):
    from haiku.rag.s3 import make_s3_session

    storage_options = {
        "endpoint": "http://seaweed:8333",
        "region": "us-east-1",
        "aws_access_key_id": "AKIA",
        "aws_secret_access_key": "secret",
        "allow_http": "true",
    }

    session, client_kwargs = make_s3_session(storage_options)

    fake_aioboto3.Session.assert_called_once_with(
        aws_access_key_id="AKIA",
        aws_secret_access_key="secret",
        region_name="us-east-1",
    )
    assert client_kwargs == {
        "endpoint_url": "http://seaweed:8333",
        "use_ssl": False,
    }
    assert session is fake_aioboto3.Session.return_value


def test_make_s3_session_accepts_native_aliases(fake_aioboto3):
    from haiku.rag.s3 import make_s3_session

    _, client_kwargs = make_s3_session(
        {"region_name": "eu-west-1", "endpoint_url": "https://s3"}
    )

    fake_aioboto3.Session.assert_called_once_with(region_name="eu-west-1")
    assert client_kwargs == {"endpoint_url": "https://s3"}


def test_make_s3_session_no_options_uses_default_chain(fake_aioboto3):
    from haiku.rag.s3 import make_s3_session

    session, client_kwargs = make_s3_session(None)

    fake_aioboto3.Session.assert_called_once_with()
    assert client_kwargs == {}


def test_make_s3_session_allow_http_only_when_truthy(fake_aioboto3):
    from haiku.rag.s3 import make_s3_session

    _, ck = make_s3_session({"allow_http": "false"})
    assert "use_ssl" not in ck

    _, ck = make_s3_session({"allow_http": "True"})
    assert ck["use_ssl"] is False


def test_make_s3_session_missing_aioboto3_raises_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "aioboto3", None)

    from haiku.rag.s3 import make_s3_session

    with pytest.raises(ImportError, match=r"haiku\.rag-slim\[s3\]"):
        make_s3_session({})


@pytest.mark.asyncio
async def test_create_document_from_s3_new(fake_s3_client, temp_db_path):
    text = b"S3 hosted content"
    fake_s3_client.head_object.return_value = {
        "ETag": '"abc123"',
        "ContentType": "text/plain",
    }
    fake_s3_client.get_object.return_value = {"Body": _streaming_body(text)}

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document_from_source("s3://my-bucket/folder/file.txt")

    assert doc.uri == "s3://my-bucket/folder/file.txt"
    assert doc.metadata["etag"] == "abc123"  # quotes stripped
    assert doc.metadata["md5"]  # real content MD5
    assert doc.metadata["md5"] != "abc123"
    fake_s3_client.head_object.assert_awaited_once()
    fake_s3_client.get_object.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_document_from_s3_skips_when_etag_unchanged(
    fake_s3_client, temp_db_path
):
    text = b"S3 hosted content"
    fake_s3_client.head_object.return_value = {
        "ETag": '"abc123"',
        "ContentType": "text/plain",
    }
    fake_s3_client.get_object.return_value = {"Body": _streaming_body(text)}

    async with HaikuRAG(temp_db_path, create=True) as client:
        first = await client.create_document_from_source("s3://my-bucket/file.txt")

        # Second call with the same ETag must not GetObject.
        fake_s3_client.get_object.reset_mock()
        # Re-arm body so a stray call would still produce something readable.
        fake_s3_client.get_object.return_value = {"Body": _streaming_body(text)}
        second = await client.create_document_from_source("s3://my-bucket/file.txt")

    assert second.id == first.id
    assert fake_s3_client.head_object.await_count == 2
    fake_s3_client.get_object.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_document_from_s3_etag_changed_md5_same_skips_rechunk(
    fake_s3_client, temp_db_path
):
    """Multipart re-upload of same content: etag changes, MD5 doesn't.

    Expected: GetObject runs to verify, but no re-chunk; only metadata.etag updates.
    """
    text = b"S3 hosted content"
    fake_s3_client.head_object.return_value = {
        "ETag": '"abc123"',
        "ContentType": "text/plain",
    }
    fake_s3_client.get_object.return_value = {"Body": _streaming_body(text)}

    async with HaikuRAG(temp_db_path, create=True) as client:
        first = await client.create_document_from_source("s3://my-bucket/file.txt")
        original_md5 = first.metadata["md5"]
        original_updated_at = first.updated_at

        # Same bytes, different ETag (simulating a multipart re-upload).
        fake_s3_client.head_object.return_value = {
            "ETag": '"def456-2"',  # multipart-style ETag
            "ContentType": "text/plain",
        }
        fake_s3_client.get_object.return_value = {"Body": _streaming_body(text)}

        second = await client.create_document_from_source("s3://my-bucket/file.txt")

    assert second.id == first.id
    assert second.metadata["md5"] == original_md5  # MD5 unchanged
    assert second.metadata["etag"] == "def456-2"  # ETag refreshed
    assert second.updated_at >= original_updated_at
    # GetObject ran once (initial create) plus once more for the etag-changed compare.
    assert fake_s3_client.get_object.await_count == 2


@pytest.mark.asyncio
async def test_create_document_from_s3_etag_changed_md5_changed_rechunks(
    fake_s3_client, temp_db_path
):
    fake_s3_client.head_object.return_value = {
        "ETag": '"abc123"',
        "ContentType": "text/plain",
    }
    fake_s3_client.get_object.return_value = {"Body": _streaming_body(b"original text")}

    async with HaikuRAG(temp_db_path, create=True) as client:
        first = await client.create_document_from_source("s3://my-bucket/file.txt")

        fake_s3_client.head_object.return_value = {
            "ETag": '"new999"',
            "ContentType": "text/plain",
        }
        fake_s3_client.get_object.return_value = {
            "Body": _streaming_body(b"different text now")
        }

        second = await client.create_document_from_source("s3://my-bucket/file.txt")

    assert second.id == first.id
    assert second.metadata["md5"] != first.metadata["md5"]
    assert second.metadata["etag"] == "new999"
    assert "different text now" in second.content


@pytest.mark.asyncio
async def test_create_document_from_s3_rejects_invalid_uri(
    fake_s3_client, temp_db_path
):
    async with HaikuRAG(temp_db_path, create=True) as client:
        with pytest.raises(ValueError, match="Invalid S3 URI"):
            await client.create_document_from_source("s3://only-bucket-no-key")


@pytest.mark.asyncio
async def test_create_document_from_s3_passes_storage_options(
    fake_aioboto3, fake_s3_client, temp_db_path
):
    fake_s3_client.head_object.return_value = {
        "ETag": '"abc"',
        "ContentType": "text/plain",
    }
    fake_s3_client.get_object.return_value = {"Body": _streaming_body(b"hello")}

    async with HaikuRAG(temp_db_path, create=True) as client:
        await client.create_document_from_source(
            "s3://bucket/key.txt",
            storage_options={
                "endpoint": "http://seaweed:8333",
                "region": "us-east-1",
                "allow_http": "true",
                "aws_access_key_id": "AKIA",
                "aws_secret_access_key": "secret",
            },
        )

    fake_aioboto3.Session.assert_called_with(
        aws_access_key_id="AKIA",
        aws_secret_access_key="secret",
        region_name="us-east-1",
    )
    fake_aioboto3.Session.return_value.client.assert_called_with(
        "s3", endpoint_url="http://seaweed:8333", use_ssl=False
    )
