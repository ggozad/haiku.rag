import pytest


async def _exhaust_build(builder):
    """Helper to run build() generator and return total_nodes."""
    async for _ in builder.build():
        pass
    return builder.total_nodes


@pytest.mark.asyncio
class TestRaptorTreeBuilder:
    async def test_build_tree_creates_nodes(self, temp_db_path):
        from haiku.rag.client import HaikuRAG
        from haiku.rag.raptor.builder import RaptorTreeBuilder

        async with HaikuRAG(temp_db_path, create=True) as client:
            # Add enough documents to form clusters
            for i in range(10):
                await client.create_document(
                    content=f"Document {i} about topic {i % 3}. Some content here.",
                    uri=f"doc://{i}",
                )

            builder = RaptorTreeBuilder(client)
            node_count = await _exhaust_build(builder)

            assert node_count > 0

    async def test_build_tree_with_few_chunks_creates_no_nodes(self, temp_db_path):
        from haiku.rag.client import HaikuRAG
        from haiku.rag.raptor.builder import RaptorTreeBuilder

        async with HaikuRAG(temp_db_path, create=True) as client:
            # Add only 2 documents - not enough for clustering
            await client.create_document(content="First document.", uri="doc://1")
            await client.create_document(content="Second document.", uri="doc://2")

            builder = RaptorTreeBuilder(client)
            node_count = await _exhaust_build(builder)

            # Too few chunks to cluster
            assert node_count == 0

    async def test_build_tree_respects_max_depth(self, temp_db_path):
        from haiku.rag.client import HaikuRAG
        from haiku.rag.config import Config
        from haiku.rag.raptor.builder import RaptorTreeBuilder
        from haiku.rag.store.repositories.raptor import RaptorNodeRepository

        # Set max_depth to 1
        Config.raptor.max_depth = 1

        async with HaikuRAG(temp_db_path, create=True) as client:
            for i in range(15):
                await client.create_document(
                    content=f"Document {i} with content about various topics.",
                    uri=f"doc://{i}",
                )

            builder = RaptorTreeBuilder(client)
            await _exhaust_build(builder)

            repo = RaptorNodeRepository(client.store)
            nodes = await repo.list_all()

            # All nodes should be layer 1 (no layer 2+)
            assert all(n.layer == 1 for n in nodes)

        # Reset
        Config.raptor.max_depth = 5

    async def test_build_clears_existing_nodes(self, temp_db_path):
        from haiku.rag.client import HaikuRAG
        from haiku.rag.raptor.builder import RaptorTreeBuilder
        from haiku.rag.store.repositories.raptor import RaptorNodeRepository

        async with HaikuRAG(temp_db_path, create=True) as client:
            for i in range(10):
                await client.create_document(
                    content=f"Document {i} about topic {i % 3}.",
                    uri=f"doc://{i}",
                )

            builder = RaptorTreeBuilder(client)

            # Build twice
            await _exhaust_build(builder)
            second_count = await _exhaust_build(builder)

            # Second build should have similar count (not doubled)
            repo = RaptorNodeRepository(client.store)
            total = await repo.count()

            assert total == second_count
