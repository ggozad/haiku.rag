# Start SeaweedFS before running:
#   docker compose -f tests/docker/docker-compose.s3.yml up -d
# Stop after:
#   docker compose -f tests/docker/docker-compose.s3.yml down -v

import importlib.util
import socket
from uuid import uuid4

import pytest

from haiku.rag.app import HaikuRAGApp
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig, LanceDBConfig, S3MonitorEntry
from haiku.rag.store.engine import Store

HAS_AIOBOTO3 = importlib.util.find_spec("aioboto3") is not None

S3_ENDPOINT = "http://localhost:8333"
S3_BUCKET = "test-bucket"
S3_STORAGE_OPTIONS = {
    "endpoint": S3_ENDPOINT,
    "region": "us-east-1",
    "allow_http": "true",
    "aws_access_key_id": "testkey",
    "aws_secret_access_key": "testsecret",
}


def _s3_available() -> bool:
    try:
        s = socket.create_connection(("localhost", 8333), timeout=1)
        s.close()
        return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _s3_available(), reason="SeaweedFS not running on localhost:8333"
    ),
]


def _make_config() -> AppConfig:
    unique_prefix = uuid4().hex[:8]
    return AppConfig(
        lancedb=LanceDBConfig(
            uri=f"s3://{S3_BUCKET}/test-{unique_prefix}",
            storage_options=S3_STORAGE_OPTIONS,
        )
    )


@pytest.mark.asyncio
async def test_store_connect_and_create(tmp_path):
    from haiku.rag.store.engine import get_database_stats

    config = _make_config()
    async with Store(tmp_path / "unused", config=config, create=True) as store:
        stats = await get_database_stats(store.db)
        assert stats["documents"]["exists"]
        assert stats["chunks"]["exists"]


@pytest.mark.asyncio
async def test_store_vacuum(tmp_path):
    config = _make_config()
    async with Store(tmp_path / "unused", config=config, create=True) as store:
        await store.vacuum()


@pytest.mark.asyncio
async def test_store_add_document(tmp_path):
    from haiku.rag.store.engine import DocumentRecord, get_database_stats

    config = _make_config()
    async with Store(tmp_path / "unused", config=config, create=True) as store:
        doc = DocumentRecord(content="The quick brown fox jumps over the lazy dog.")
        await store.documents_table.add([doc])

        stats = await get_database_stats(store.db)
        assert stats["documents"]["num_rows"] == 1


@pytest.mark.asyncio
async def test_client_create_document(tmp_path):
    config = _make_config()
    async with HaikuRAG(tmp_path / "unused", config=config, create=True) as rag:
        doc = await rag.create_document(
            "Python is a programming language.", uri="test://python"
        )
        assert doc.id
        assert doc.uri == "test://python"


@pytest.mark.asyncio
async def test_client_list_documents(tmp_path):
    config = _make_config()
    async with HaikuRAG(tmp_path / "unused", config=config, create=True) as rag:
        await rag.create_document("First document.", uri="test://first")
        await rag.create_document("Second document.", uri="test://second")

        docs = await rag.list_documents()
        assert len(docs) == 2


@pytest.mark.asyncio
async def test_client_search(tmp_path):
    config = _make_config()
    async with HaikuRAG(tmp_path / "unused", config=config, create=True) as rag:
        await rag.create_document(
            "The Eiffel Tower is located in Paris, France.", uri="test://eiffel"
        )
        results = await rag.search("Eiffel Tower")
        assert len(results) > 0
        assert "Eiffel" in results[0].content


@pytest.mark.asyncio
async def test_client_delete_document(tmp_path):
    config = _make_config()
    async with HaikuRAG(tmp_path / "unused", config=config, create=True) as rag:
        doc = await rag.create_document("Temporary document.", uri="test://temp")
        await rag.delete_document(doc.id)
        docs = await rag.list_documents()
        assert len(docs) == 0


@pytest.mark.asyncio
async def test_app_info(tmp_path, capsys):
    config = _make_config()
    async with HaikuRAG(tmp_path / "unused", config=config, create=True) as rag:
        await rag.create_document("Info test document.", uri="test://info")

    app = HaikuRAGApp(db_path=tmp_path / "unused", config=config)
    await app.info()

    out = capsys.readouterr().out
    assert "path:" in out
    assert config.lancedb.uri in out
    assert "documents: 1" in out


@pytest.mark.asyncio
async def test_app_info_empty_db(tmp_path, capsys):
    config = _make_config()
    app = HaikuRAGApp(db_path=tmp_path / "unused", config=config)
    await app.info()

    out = capsys.readouterr().out
    assert "Database is empty" in out


# ----------------------- S3 watcher integration tests ----------------------- #
# These exercise S3Watcher against the live SeaweedFS instance. Documents are
# uploaded as raw S3 objects under a unique per-test prefix; the watcher's
# refresh() is invoked directly so tests stay deterministic. LanceDB stays
# local — these tests verify the watcher path, not LanceDB-on-S3.


_aioboto3_required = pytest.mark.skipif(
    not HAS_AIOBOTO3,
    reason="aioboto3 not installed (uv sync --extra s3)",
)


async def _put_object(prefix: str, key: str, body: bytes) -> None:
    import aioboto3  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]

    session = aioboto3.Session(
        aws_access_key_id=S3_STORAGE_OPTIONS["aws_access_key_id"],
        aws_secret_access_key=S3_STORAGE_OPTIONS["aws_secret_access_key"],
        region_name=S3_STORAGE_OPTIONS["region"],
    )
    async with session.client(
        "s3", endpoint_url=S3_STORAGE_OPTIONS["endpoint"], use_ssl=False
    ) as s3:
        try:
            await s3.create_bucket(Bucket=S3_BUCKET)
        except Exception:
            pass  # bucket already exists
        await s3.put_object(Bucket=S3_BUCKET, Key=f"{prefix}/{key}", Body=body)


async def _delete_object(prefix: str, key: str) -> None:
    import aioboto3  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]

    session = aioboto3.Session(
        aws_access_key_id=S3_STORAGE_OPTIONS["aws_access_key_id"],
        aws_secret_access_key=S3_STORAGE_OPTIONS["aws_secret_access_key"],
        region_name=S3_STORAGE_OPTIONS["region"],
    )
    async with session.client(
        "s3", endpoint_url=S3_STORAGE_OPTIONS["endpoint"], use_ssl=False
    ) as s3:
        await s3.delete_object(Bucket=S3_BUCKET, Key=f"{prefix}/{key}")


def _watcher_entry(prefix: str, **overrides) -> S3MonitorEntry:
    return S3MonitorEntry(
        uri=overrides.pop("uri", f"s3://{S3_BUCKET}/{prefix}/"),
        storage_options=overrides.pop("storage_options", S3_STORAGE_OPTIONS),
        include_patterns=overrides.pop("include_patterns", ["*.txt"]),
        delete_orphans=overrides.pop("delete_orphans", False),
        poll_interval=overrides.pop("poll_interval", 60),
        **overrides,
    )


@_aioboto3_required
@pytest.mark.asyncio
async def test_s3_watcher_initial_sweep(tmp_path):
    from haiku.rag.monitor import S3Watcher

    prefix = f"watcher-init-{uuid4().hex[:8]}"
    await _put_object(prefix, "alpha.txt", b"alpha content")
    await _put_object(prefix, "beta.txt", b"beta content")

    async with HaikuRAG(tmp_path / "db.lancedb", create=True) as rag:
        watcher = S3Watcher(
            client=rag,
            entry=_watcher_entry(prefix),
            supported_extensions=[".txt", ".md", ".pdf"],
        )
        await watcher.refresh()

        docs = await rag.list_documents()
        uris = sorted(d.uri or "" for d in docs)

    assert uris == [
        f"s3://{S3_BUCKET}/{prefix}/alpha.txt",
        f"s3://{S3_BUCKET}/{prefix}/beta.txt",
    ]


@_aioboto3_required
@pytest.mark.asyncio
async def test_s3_watcher_detects_new_object(tmp_path):
    from haiku.rag.monitor import S3Watcher

    prefix = f"watcher-new-{uuid4().hex[:8]}"
    await _put_object(prefix, "first.txt", b"first content")

    async with HaikuRAG(tmp_path / "db.lancedb", create=True) as rag:
        watcher = S3Watcher(
            client=rag,
            entry=_watcher_entry(prefix),
            supported_extensions=[".txt"],
        )
        await watcher.refresh()
        assert await rag.count_documents() == 1

        await _put_object(prefix, "second.txt", b"second content")
        await watcher.refresh()
        assert await rag.count_documents() == 2


@_aioboto3_required
@pytest.mark.asyncio
async def test_s3_watcher_detects_modified_object(tmp_path):
    from haiku.rag.monitor import S3Watcher

    prefix = f"watcher-mod-{uuid4().hex[:8]}"
    uri = f"s3://{S3_BUCKET}/{prefix}/file.txt"

    await _put_object(prefix, "file.txt", b"original content")

    async with HaikuRAG(tmp_path / "db.lancedb", create=True) as rag:
        watcher = S3Watcher(
            client=rag,
            entry=_watcher_entry(prefix),
            supported_extensions=[".txt"],
        )
        await watcher.refresh()

        first = await rag.get_document_by_uri(uri)
        assert first is not None
        first_md5 = first.metadata["md5"]

        await _put_object(prefix, "file.txt", b"new content body")
        await watcher.refresh()

        second = await rag.get_document_by_uri(uri)
        assert second is not None
        assert second.id == first.id
        assert second.metadata["md5"] != first_md5
        assert "new content body" in second.content


@_aioboto3_required
@pytest.mark.asyncio
async def test_s3_watcher_orphan_deletion(tmp_path):
    from haiku.rag.monitor import S3Watcher

    prefix = f"watcher-orphan-{uuid4().hex[:8]}"
    await _put_object(prefix, "kept.txt", b"keep me")
    await _put_object(prefix, "doomed.txt", b"will be deleted")

    async with HaikuRAG(tmp_path / "db.lancedb", create=True) as rag:
        watcher = S3Watcher(
            client=rag,
            entry=_watcher_entry(prefix, delete_orphans=True),
            supported_extensions=[".txt"],
        )
        await watcher.refresh()
        assert await rag.count_documents() == 2

        await _delete_object(prefix, "doomed.txt")
        await watcher.refresh()

        docs = await rag.list_documents()
        assert len(docs) == 1
        assert docs[0].uri == f"s3://{S3_BUCKET}/{prefix}/kept.txt"
