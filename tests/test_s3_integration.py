# Start the services before running:
#   docker compose -f tests/docker/docker-compose.yml up -d
# Stop after:
#   docker compose -f tests/docker/docker-compose.yml down -v

from uuid import uuid4

import obstore
import pytest

from haiku.rag.app import HaikuRAGApp
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig, LanceDBConfig
from haiku.rag.s3 import make_s3_store
from haiku.rag.store.engine import Store
from tests.services import reachable

S3_ENDPOINT = "http://localhost:8333"
S3_BUCKET = "test-bucket"
S3_STORAGE_OPTIONS = {
    "endpoint": S3_ENDPOINT,
    "region": "us-east-1",
    "allow_http": "true",
    "aws_access_key_id": "testkey",
    "aws_secret_access_key": "testsecret",
}


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not reachable("localhost", 8333),
        reason="SeaweedFS not running on localhost:8333",
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


@pytest.fixture
def config():
    """A config pointing at a unique S3 prefix, cleaned up after the test.

    SeaweedFS reclaims volume space lazily, so leaving each run's data behind
    eventually fills the volume server and seals its volumes read-only, which
    makes every subsequent write block forever.
    """
    config = _make_config()
    yield config

    bucket, _, prefix = config.lancedb.uri.removeprefix("s3://").partition("/")
    store = make_s3_store(bucket, S3_STORAGE_OPTIONS)
    paths = [obj["path"] for batch in store.list(prefix=f"{prefix}/") for obj in batch]
    if paths:
        obstore.delete(store, paths)


@pytest.mark.asyncio
async def test_store_connect_and_create(tmp_path, config):
    from haiku.rag.store.engine import get_database_stats

    async with Store(tmp_path / "unused", config=config, create=True) as store:
        stats = await get_database_stats(store.db)
        assert stats["documents"]["exists"]
        assert stats["chunks"]["exists"]


@pytest.mark.asyncio
async def test_store_vacuum(tmp_path, config):
    async with Store(tmp_path / "unused", config=config, create=True) as store:
        await store.vacuum()


@pytest.mark.asyncio
async def test_store_add_document(tmp_path, config):
    from haiku.rag.store.engine import DocumentRecord, get_database_stats

    async with Store(tmp_path / "unused", config=config, create=True) as store:
        doc = DocumentRecord(content="The quick brown fox jumps over the lazy dog.")
        await store.documents_table.add([doc])

        stats = await get_database_stats(store.db)
        assert stats["documents"]["num_rows"] == 1


@pytest.mark.asyncio
async def test_client_create_document(tmp_path, config):
    async with HaikuRAG(tmp_path / "unused", config=config, create=True) as rag:
        doc = await rag.create_document(
            "Python is a programming language.", uri="test://python"
        )
        assert doc.id
        assert doc.uri == "test://python"


@pytest.mark.asyncio
async def test_client_list_documents(tmp_path, config):
    async with HaikuRAG(tmp_path / "unused", config=config, create=True) as rag:
        await rag.create_document("First document.", uri="test://first")
        await rag.create_document("Second document.", uri="test://second")

        docs = await rag.list_documents()
        assert len(docs) == 2


@pytest.mark.asyncio
async def test_client_search(tmp_path, config):
    async with HaikuRAG(tmp_path / "unused", config=config, create=True) as rag:
        await rag.create_document(
            "The Eiffel Tower is located in Paris, France.", uri="test://eiffel"
        )
        results = await rag.search("Eiffel Tower")
        assert len(results) > 0
        assert "Eiffel" in results[0].content


@pytest.mark.asyncio
async def test_client_delete_document(tmp_path, config):
    async with HaikuRAG(tmp_path / "unused", config=config, create=True) as rag:
        doc = await rag.create_document("Temporary document.", uri="test://temp")
        await rag.delete_document(doc.id)
        docs = await rag.list_documents()
        assert len(docs) == 0


@pytest.mark.asyncio
async def test_app_info(tmp_path, capsys, config):
    async with HaikuRAG(tmp_path / "unused", config=config, create=True) as rag:
        await rag.create_document("Info test document.", uri="test://info")

    app = HaikuRAGApp(db_path=tmp_path / "unused", config=config)
    await app.info()

    out = capsys.readouterr().out
    assert "path:" in out
    assert config.lancedb.uri in out
    assert "documents: 1" in out


@pytest.mark.asyncio
async def test_app_info_empty_db(tmp_path, capsys, config):
    app = HaikuRAGApp(db_path=tmp_path / "unused", config=config)
    await app.info()

    out = capsys.readouterr().out
    assert "Database is empty" in out
