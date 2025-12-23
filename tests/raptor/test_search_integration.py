import pytest


async def _exhaust_build(builder):
    """Helper to run build() generator and return total_nodes."""
    async for _ in builder.build():
        pass
    return builder.total_nodes


@pytest.mark.asyncio
class TestRaptorSearchIntegration:
    async def test_search_includes_raptor_results(self, temp_db_path):
        from haiku.rag.client import HaikuRAG
        from haiku.rag.raptor.builder import RaptorTreeBuilder

        async with HaikuRAG(temp_db_path, create=True) as client:
            # Add documents
            for i in range(10):
                await client.create_document(
                    content=f"Document {i} about topic {i % 3}. Some content here.",
                    uri=f"doc://{i}",
                )

            # Build RAPTOR tree
            builder = RaptorTreeBuilder(client)
            node_count = await _exhaust_build(builder)
            assert node_count > 0

            # Search should return both chunks and RAPTOR summaries
            results = await client.search("topic", limit=20)
            assert len(results) > 0

            # Check we have both types
            chunk_results = [r for r in results if r.chunk_id is not None]
            raptor_results = [r for r in results if r.raptor_node_id is not None]

            assert len(chunk_results) > 0, "Should have chunk results"
            assert len(raptor_results) > 0, "Should have RAPTOR summary results"

    async def test_search_without_raptor_nodes(self, temp_db_path):
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as client:
            # Add documents but don't build RAPTOR
            await client.create_document(
                content="Document about testing.",
                uri="doc://1",
            )

            # Search should still work, returning only chunks
            results = await client.search("testing", limit=5)
            assert len(results) > 0
            assert all(r.chunk_id is not None for r in results)
            assert all(r.raptor_node_id is None for r in results)

    async def test_search_results_chunks_sorted_summaries_appended(self, temp_db_path):
        from haiku.rag.client import HaikuRAG
        from haiku.rag.raptor.builder import RaptorTreeBuilder

        async with HaikuRAG(temp_db_path, create=True) as client:
            for i in range(10):
                await client.create_document(
                    content=f"Document {i} about topic {i % 3}.",
                    uri=f"doc://{i}",
                )

            builder = RaptorTreeBuilder(client)
            await _exhaust_build(builder)

            results = await client.search("topic", limit=10)

            # Separate chunks and summaries
            chunk_results = [r for r in results if r.chunk_id is not None]
            raptor_results = [r for r in results if r.raptor_node_id is not None]

            # Chunks should be sorted by score descending
            chunk_scores = [r.score for r in chunk_results]
            assert chunk_scores == sorted(chunk_scores, reverse=True)

            # Summaries should come after chunks in the result list
            if raptor_results:
                last_chunk_idx = max(
                    i for i, r in enumerate(results) if r.chunk_id is not None
                )
                first_raptor_idx = min(
                    i for i, r in enumerate(results) if r.raptor_node_id is not None
                )
                assert last_chunk_idx < first_raptor_idx

    async def test_format_for_agent_handles_raptor(self, temp_db_path):
        from haiku.rag.client import HaikuRAG
        from haiku.rag.raptor.builder import RaptorTreeBuilder

        async with HaikuRAG(temp_db_path, create=True) as client:
            for i in range(10):
                await client.create_document(
                    content=f"Document {i} about topic {i % 3}.",
                    uri=f"doc://{i}",
                )

            builder = RaptorTreeBuilder(client)
            await _exhaust_build(builder)

            results = await client.search("topic", limit=20)

            for result in results:
                formatted = result.format_for_agent()
                if result.raptor_node_id:
                    assert formatted.startswith("[Summary]")
                else:
                    assert formatted.startswith(f"[{result.chunk_id}]")
