import asyncio
import random
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from haiku.rag.client import DocumentImport, HaikuRAG
from haiku.rag.config import get_config
from haiku.rag.ingester.maintenance import (
    MaintenanceError,
    MaintenanceResult,
    has_vector_index,
    maintain,
)
from haiku.rag.store.models.chunk import Chunk
from tests.documents import docling_doc

DIM = 16


@pytest.fixture
async def client(temp_db_path):
    config = get_config().model_copy(deep=True)
    config.embeddings.model.vector_dim = DIM
    config.storage.auto_vacuum = False
    async with HaikuRAG(temp_db_path, config=config, create=True) as rag:
        yield rag


async def add_chunks(client: HaikuRAG, count: int) -> None:
    rng = random.Random(count)
    chunks = [
        Chunk(
            content=f"chunk {i}",
            embedding=[rng.uniform(-1, 1) for _ in range(DIM)],
            order=i,
        )
        for i in range(count)
    ]
    await client.import_documents(
        [
            DocumentImport(
                docling_document=docling_doc("d", "body"),
                chunks=chunks,
                uri=f"mem://{count}",
            )
        ]
    )


async def test_a_small_table_is_vacuumed_and_left_unindexed(client):
    await add_chunks(client, 10)

    assert await maintain(client) == MaintenanceResult(vacuumed=True, index_built=False)


async def test_a_table_without_an_index_gets_one(client):
    await add_chunks(client, 300)
    assert await has_vector_index(client.store) is False

    assert await maintain(client) == MaintenanceResult(vacuumed=True, index_built=True)
    assert await has_vector_index(client.store) is True


async def test_an_indexed_table_is_vacuumed_and_not_rebuilt(client):
    await add_chunks(client, 300)
    await maintain(client)
    await add_chunks(client, 1000)

    assert await maintain(client) == MaintenanceResult(vacuumed=True, index_built=False)
    stats = await client.store.chunks_table.index_stats("vector_idx")
    assert stats is not None
    assert stats.num_unindexed_rows == 0


async def test_a_vacuum_that_does_not_complete_fails_the_pass(client):
    await add_chunks(client, 300)
    async with client.store._vacuum_lock:
        with pytest.raises(MaintenanceError, match="vacuum"):
            await asyncio.wait_for(maintain(client), timeout=5)

    assert await has_vector_index(client.store) is False


async def test_a_cloud_database_is_left_alone():
    store = AsyncMock(location="db://cloud")
    cloud = SimpleNamespace(store=store, create_vector_index=AsyncMock())

    assert await maintain(cloud) == MaintenanceResult(vacuumed=False, index_built=False)  # ty: ignore[invalid-argument-type]
    store.vacuum.assert_not_awaited()
    store.chunks_table.list_indices.assert_not_awaited()
    cloud.create_vector_index.assert_not_awaited()
