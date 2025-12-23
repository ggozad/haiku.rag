import pytest

from haiku.rag.store.models.raptor import RaptorNode


class TestRaptorNode:
    def test_create_raptor_node(self):
        node = RaptorNode(
            id="node-1",
            content="Summary of cluster content",
            layer=1,
            cluster_id=0,
            source_chunk_ids=["chunk-1", "chunk-2", "chunk-3"],
        )
        assert node.id == "node-1"
        assert node.content == "Summary of cluster content"
        assert node.layer == 1
        assert node.cluster_id == 0
        assert node.source_chunk_ids == ["chunk-1", "chunk-2", "chunk-3"]
        assert node.embedding is None

    def test_node_with_embedding(self):
        embedding = [0.1, 0.2, 0.3, 0.4]
        node = RaptorNode(
            id="node-1",
            content="Content",
            layer=1,
            cluster_id=0,
            source_chunk_ids=["c1"],
            embedding=embedding,
        )
        assert node.embedding == embedding


@pytest.mark.asyncio
class TestRaptorNodeRepository:
    async def test_create_single_node(self, temp_db_path):
        from haiku.rag.store.engine import Store
        from haiku.rag.store.repositories.raptor import RaptorNodeRepository

        store = Store(temp_db_path, create=True)
        repo = RaptorNodeRepository(store)

        node = RaptorNode(
            content="Test summary",
            layer=1,
            cluster_id=0,
            source_chunk_ids=["c1", "c2"],
            embedding=[0.1] * store.embedder._vector_dim,
        )

        created = await repo.create(node)
        assert created.id is not None
        assert created.content == "Test summary"

    async def test_create_batch_nodes(self, temp_db_path):
        from haiku.rag.store.engine import Store
        from haiku.rag.store.repositories.raptor import RaptorNodeRepository

        store = Store(temp_db_path, create=True)
        repo = RaptorNodeRepository(store)

        nodes = [
            RaptorNode(
                content=f"Summary {i}",
                layer=1,
                cluster_id=i,
                source_chunk_ids=[f"c{i}"],
                embedding=[0.1] * store.embedder._vector_dim,
            )
            for i in range(3)
        ]

        created = await repo.create(nodes)
        assert len(created) == 3
        assert all(n.id is not None for n in created)

    async def test_get_by_id(self, temp_db_path):
        from haiku.rag.store.engine import Store
        from haiku.rag.store.repositories.raptor import RaptorNodeRepository

        store = Store(temp_db_path, create=True)
        repo = RaptorNodeRepository(store)

        node = RaptorNode(
            content="Test content",
            layer=2,
            cluster_id=0,
            source_chunk_ids=["c1"],
            embedding=[0.1] * store.embedder._vector_dim,
        )
        created = await repo.create(node)
        assert created.id is not None

        retrieved = await repo.get_by_id(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.content == "Test content"
        assert retrieved.layer == 2

    async def test_get_by_layer(self, temp_db_path):
        from haiku.rag.store.engine import Store
        from haiku.rag.store.repositories.raptor import RaptorNodeRepository

        store = Store(temp_db_path, create=True)
        repo = RaptorNodeRepository(store)

        # Create nodes at different layers (1, 2, 3 - no layer 0 since those are chunks)
        for layer in range(1, 4):
            for i in range(2):
                node = RaptorNode(
                    content=f"Layer {layer} node {i}",
                    layer=layer,
                    cluster_id=i,
                    source_chunk_ids=[f"c{i}"],
                    embedding=[0.1] * store.embedder._vector_dim,
                )
                await repo.create(node)

        layer_2_nodes = await repo.get_by_layer(2)
        assert len(layer_2_nodes) == 2
        assert all(n.layer == 2 for n in layer_2_nodes)

    async def test_delete_all(self, temp_db_path):
        from haiku.rag.store.engine import Store
        from haiku.rag.store.repositories.raptor import RaptorNodeRepository

        store = Store(temp_db_path, create=True)
        repo = RaptorNodeRepository(store)

        # Create some nodes
        for i in range(3):
            node = RaptorNode(
                content=f"Node {i}",
                layer=1,
                cluster_id=i,
                source_chunk_ids=[f"c{i}"],
                embedding=[0.1] * store.embedder._vector_dim,
            )
            await repo.create(node)

        # Delete all
        await repo.delete_all()

        # Verify empty
        all_nodes = await repo.list_all()
        assert len(all_nodes) == 0

    async def test_count(self, temp_db_path):
        from haiku.rag.store.engine import Store
        from haiku.rag.store.repositories.raptor import RaptorNodeRepository

        store = Store(temp_db_path, create=True)
        repo = RaptorNodeRepository(store)

        # Initially empty
        assert await repo.count() == 0

        # Create some nodes (layers 1 and 2)
        for i in range(5):
            node = RaptorNode(
                content=f"Node {i}",
                layer=(i % 2) + 1,
                cluster_id=i,
                source_chunk_ids=[f"c{i}"],
                embedding=[0.1] * store.embedder._vector_dim,
            )
            await repo.create(node)

        assert await repo.count() == 5

    async def test_search(self, temp_db_path):
        from haiku.rag.store.engine import Store
        from haiku.rag.store.repositories.raptor import RaptorNodeRepository

        store = Store(temp_db_path, create=True)
        repo = RaptorNodeRepository(store)

        # Create nodes with different content
        contents = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning and artificial intelligence",
            "Database systems and query optimization",
        ]

        for i, content in enumerate(contents):
            embedding = await store.embedder.embed(content)
            node = RaptorNode(
                content=content,
                layer=1,
                cluster_id=i,
                source_chunk_ids=[f"c{i}"],
                embedding=embedding,
            )
            await repo.create(node)

        # Search for AI-related content
        results = await repo.search("artificial intelligence", limit=2)
        assert len(results) > 0
        # Results should be (RaptorNode, score) tuples
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
