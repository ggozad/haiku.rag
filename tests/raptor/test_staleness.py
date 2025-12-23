import pytest


async def _exhaust_build(builder):
    """Helper to run build() generator and return total_nodes."""
    async for _ in builder.build():
        pass
    return builder.total_nodes


@pytest.mark.asyncio
class TestRaptorStaleness:
    async def test_is_raptor_stale_returns_none_when_never_built(self, temp_db_path):
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document(content="Test content", uri="doc://1")

            # Never built RAPTOR
            assert client.store.is_raptor_stale() is None

    async def test_is_raptor_stale_returns_false_when_fresh(self, temp_db_path):
        from haiku.rag.client import HaikuRAG
        from haiku.rag.raptor.builder import RaptorTreeBuilder

        async with HaikuRAG(temp_db_path, create=True) as client:
            # Need enough documents to create RAPTOR nodes
            for i in range(10):
                await client.create_document(
                    content=f"Document {i} about topic {i % 3}. Some content here.",
                    uri=f"doc://{i}",
                )

            # Build RAPTOR
            builder = RaptorTreeBuilder(client)
            node_count = await _exhaust_build(builder)
            assert node_count > 0, "RAPTOR should create nodes"

            # Should be fresh immediately after build
            assert client.store.is_raptor_stale() is False

    async def test_is_raptor_stale_returns_true_after_doc_added(self, temp_db_path):
        from haiku.rag.client import HaikuRAG
        from haiku.rag.raptor.builder import RaptorTreeBuilder

        async with HaikuRAG(temp_db_path, create=True) as client:
            for i in range(10):
                await client.create_document(
                    content=f"Document {i} about topic {i % 3}. Some content here.",
                    uri=f"doc://{i}",
                )

            # Build RAPTOR
            builder = RaptorTreeBuilder(client)
            node_count = await _exhaust_build(builder)
            assert node_count > 0, "RAPTOR should create nodes"

            # Add a new document
            await client.create_document(content="New document", uri="doc://new")

            # Should be stale now
            assert client.store.is_raptor_stale() is True

    async def test_is_raptor_stale_returns_true_after_doc_updated(self, temp_db_path):
        from haiku.rag.client import HaikuRAG
        from haiku.rag.raptor.builder import RaptorTreeBuilder

        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await client.create_document(
                content="Original content about topic. Some content here.",
                uri="doc://1",
            )
            for i in range(9):
                await client.create_document(
                    content=f"Document {i} about topic {i % 3}. Some content here.",
                    uri=f"doc://{i + 2}",
                )

            # Build RAPTOR
            builder = RaptorTreeBuilder(client)
            node_count = await _exhaust_build(builder)
            assert node_count > 0, "RAPTOR should create nodes"

            # Update a document
            assert doc.id is not None
            await client.update_document(
                doc.id, content="Updated content about topic. Some content here."
            )

            # Should be stale now
            assert client.store.is_raptor_stale() is True

    async def test_is_raptor_stale_returns_false_after_rebuild(self, temp_db_path):
        from haiku.rag.client import HaikuRAG
        from haiku.rag.raptor.builder import RaptorTreeBuilder

        async with HaikuRAG(temp_db_path, create=True) as client:
            for i in range(10):
                await client.create_document(
                    content=f"Document {i} about topic {i % 3}. Some content here.",
                    uri=f"doc://{i}",
                )

            # Build RAPTOR
            builder = RaptorTreeBuilder(client)
            node_count = await _exhaust_build(builder)
            assert node_count > 0, "RAPTOR should create nodes"

            # Add a new document (makes it stale)
            await client.create_document(content="New document", uri="doc://new")
            assert client.store.is_raptor_stale() is True

            # Rebuild RAPTOR
            builder = RaptorTreeBuilder(client)
            await _exhaust_build(builder)

            # Should be fresh again
            assert client.store.is_raptor_stale() is False
