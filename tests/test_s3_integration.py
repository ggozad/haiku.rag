# Start SeaweedFS before running:
#   docker compose -f tests/docker/docker-compose.s3.yml up -d
# Stop after:
#   docker compose -f tests/docker/docker-compose.s3.yml down -v

import socket
from uuid import uuid4

import pytest

from haiku.rag.app import HaikuRAGApp
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig, LanceDBConfig
from haiku.rag.store.engine import Store

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


def test_store_connect_and_create(tmp_path):
    config = _make_config()
    store = Store(tmp_path / "unused", config=config, create=True)
    stats = store.get_stats()
    assert stats["documents"]["exists"]
    assert stats["chunks"]["exists"]
    store.close()


@pytest.mark.asyncio
async def test_store_vacuum(tmp_path):
    config = _make_config()
    store = Store(tmp_path / "unused", config=config, create=True)
    await store.vacuum()
    store.close()


def test_store_add_document(tmp_path):
    from haiku.rag.store.engine import DocumentRecord

    config = _make_config()
    store = Store(tmp_path / "unused", config=config, create=True)

    doc = DocumentRecord(content="The quick brown fox jumps over the lazy dog.")
    store.documents_table.add([doc])

    stats = store.get_stats()
    assert stats["documents"]["num_rows"] == 1
    store.close()


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
