import asyncio
import random
from unittest.mock import patch

import lancedb
import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from lancedb.index import FTS, BTree

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import (
    AppConfig,
    EmbeddingModelConfig,
    EmbeddingsConfig,
    StorageConfig,
)
from haiku.rag.store.engine import ConnectionMode, Store
from haiku.rag.store.exceptions import ReadOnlyError
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.store.schema import has_vector_index

DIM = 32
CONFIG = AppConfig(
    embeddings=EmbeddingsConfig(model=EmbeddingModelConfig(vector_dim=DIM)),
    storage=StorageConfig(auto_vacuum=False),
)


async def _import(client: HaikuRAG, chunks: int, uri: str) -> None:
    rng = random.Random(uri)
    doc = DoclingDocument(name=uri)
    doc.add_text(label=DocItemLabel.TEXT, text=uri)
    await client.import_document(
        doc,
        [
            Chunk(
                content=f"{uri} {i}",
                embedding=[rng.random() for _ in range(DIM)],
                order=i,
            )
            for i in range(chunks)
        ],
        uri=uri,
    )


async def _vector_index_stats(client: HaikuRAG):
    return await client.store.chunks_table.index_stats("vector_idx")


class Gate:
    """Holds a vector index build on the chunks table until released."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.error: Exception | None = None

    def __enter__(self):
        create_index = lancedb.AsyncTable.create_index
        gate = self

        async def gated(self, column, *args, **kwargs):
            if self.name == "chunks" and column == "vector":
                gate.started.set()
                await gate.release.wait()
                if gate.error is not None:
                    raise gate.error
            return await create_index(self, column, *args, **kwargs)

        self._patch = patch.object(lancedb.AsyncTable, "create_index", gated)
        self._patch.start()
        return self

    def __exit__(self, *_exc):
        self.release.set()
        self._patch.stop()


def _bounded(awaitable):
    return asyncio.wait_for(awaitable, timeout=10)


async def _not_done(task: asyncio.Task) -> bool:
    for _ in range(20):
        await asyncio.sleep(0)
    await asyncio.sleep(0.05)
    return not task.done()


async def test_builds_an_index_and_reports_it(temp_db_path):
    async with HaikuRAG(temp_db_path, CONFIG, create=True) as client:
        await _import(client, 256, "test://a")

        assert await client.create_vector_index() is True

        stats = await _vector_index_stats(client)
        assert stats is not None
        assert stats.num_indexed_rows == 256


async def test_builds_nothing_below_256_chunks(temp_db_path):
    async with HaikuRAG(temp_db_path, CONFIG, create=True) as client:
        await _import(client, 255, "test://a")

        assert await client.create_vector_index() is False

        assert not await has_vector_index(client.store.chunks_table)


async def test_retrains_an_existing_index(temp_db_path):
    async with HaikuRAG(temp_db_path, CONFIG, create=True) as client:
        await _import(client, 256, "test://a")
        await client.create_vector_index()
        await _import(client, 10, "test://b")

        assert await client.create_vector_index() is True

        indexes = await client.store.chunks_table.list_indices()
        assert [i.name for i in indexes if "vector" in i.columns] == ["vector_idx"]
        stats = await _vector_index_stats(client)
        assert stats is not None
        assert stats.num_indexed_rows == 266


async def test_builds_nothing_on_lancedb_cloud(temp_db_path):
    async with HaikuRAG(temp_db_path, CONFIG, create=True) as client:
        await _import(client, 256, "test://a")
        with patch.object(Store, "_connection_mode", ConnectionMode.CLOUD):
            assert await client.create_vector_index() is False

        assert not await has_vector_index(client.store.chunks_table)


async def test_refuses_a_read_only_client(temp_db_path):
    async with HaikuRAG(temp_db_path, CONFIG, create=True) as client:
        await _import(client, 256, "test://a")

    async with HaikuRAG(temp_db_path, CONFIG, read_only=True) as client:
        with pytest.raises(ReadOnlyError):
            await client.create_vector_index()


async def test_a_write_waits_for_the_build(temp_db_path):
    async with HaikuRAG(temp_db_path, CONFIG, create=True) as client:
        await _import(client, 256, "test://a")

        with Gate() as gate:
            build = asyncio.create_task(client.create_vector_index())
            await _bounded(gate.started.wait())
            assert client.store._write_lock.locked()
            write = asyncio.create_task(_import(client, 10, "test://b"))
            assert await _not_done(write)
            gate.release.set()
            assert await _bounded(build) is True
            await _bounded(write)

        stats = await _vector_index_stats(client)
        assert stats is not None
        assert (stats.num_indexed_rows, stats.num_unindexed_rows) == (256, 10)


async def test_a_vacuum_waits_for_the_build(temp_db_path):
    async with HaikuRAG(temp_db_path, CONFIG, create=True) as client:
        await _import(client, 256, "test://a")

        with Gate() as gate:
            build = asyncio.create_task(client.create_vector_index())
            await _bounded(gate.started.wait())
            vacuum = asyncio.create_task(client.vacuum(retention_seconds=0))
            assert await _not_done(vacuum)
            gate.release.set()
            assert await _bounded(build) is True
            await _bounded(vacuum)

        assert await has_vector_index(client.store.chunks_table)


async def test_a_rebuild_waits_for_the_build_and_retrains_it(temp_db_path, monkeypatch):
    """An embed-only rebuild recreates the chunks table, so it runs after the
    build and retrains the new index."""
    from haiku.rag import embeddings
    from haiku.rag.client import RebuildMode

    rng = random.Random(0)

    async def random_embeddings(chunks, embedder, config=None):
        for chunk in chunks:
            chunk.embedding = [rng.random() for _ in range(DIM)]
        return chunks

    monkeypatch.setattr(embeddings, "embed_chunks", random_embeddings)

    async def rebuild():
        async for _ in client.rebuild_database(mode=RebuildMode.EMBED_ONLY):
            pass

    async with HaikuRAG(temp_db_path, CONFIG, create=True) as client:
        await _import(client, 256, "test://a")

        with Gate() as gate:
            build = asyncio.create_task(client.create_vector_index())
            await _bounded(gate.started.wait())
            rebuilding = asyncio.create_task(rebuild())
            assert await _not_done(rebuilding)
            gate.release.set()
            assert await _bounded(build) is True
            await _bounded(rebuilding)

        stats = await _vector_index_stats(client)
        assert stats is not None
        assert (stats.num_indexed_rows, stats.num_unindexed_rows) == (256, 0)


async def test_a_tag_operation_during_the_build_fails_fast(temp_db_path):
    async with HaikuRAG(temp_db_path, CONFIG, create=True) as client:
        await _import(client, 256, "test://a")

        with Gate() as gate:
            build = asyncio.create_task(client.create_vector_index())
            await _bounded(gate.started.wait())
            with pytest.raises(ValueError, match="in progress"):
                await asyncio.wait_for(
                    client.store.create_tag("during-build"), timeout=5
                )
            gate.release.set()
            await _bounded(build)


@pytest.mark.parametrize("ending", ["fails", "cancelled"])
async def test_an_unfinished_build_releases_its_locks(temp_db_path, ending):
    async with HaikuRAG(temp_db_path, CONFIG, create=True) as client:
        await _import(client, 256, "test://a")

        with Gate() as gate:
            build = asyncio.create_task(client.create_vector_index())
            await _bounded(gate.started.wait())
            if ending == "fails":
                gate.error = RuntimeError("index build failed")
                gate.release.set()
                with pytest.raises(RuntimeError, match="index build failed"):
                    await _bounded(build)
            else:
                build.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await build

        await _bounded(_import(client, 10, "test://b"))
        await _bounded(client.store.create_tag("after-build"))


class TestHasVectorIndex:
    async def test_an_index_on_the_vector_column(self, temp_db_path):
        async with HaikuRAG(temp_db_path, CONFIG, create=True) as client:
            await _import(client, 256, "test://a")
            await client.create_vector_index()

            assert await has_vector_index(client.store.chunks_table)

    async def test_scalar_and_full_text_indexes_only(self, temp_db_path):
        async with HaikuRAG(temp_db_path, CONFIG, create=True) as client:
            await _import(client, 10, "test://a")
            indexes = await client.store.chunks_table.list_indices()
            assert {i.index_type for i in indexes} >= {
                type(FTS()).__name__,
                type(BTree()).__name__,
            }

            assert not await has_vector_index(client.store.chunks_table)
