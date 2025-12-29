import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.graph.research.dependencies import ResearchContext
from haiku.rag.graph.research.graph import build_research_graph
from haiku.rag.graph.research.state import ResearchDeps, ResearchState


@pytest.fixture
async def client_with_docs(temp_db_path):
    """Create a client with two distinct documents."""
    client = HaikuRAG(temp_db_path, create=True)

    # Add two documents with distinct content
    doc1 = await client.create_document(
        "Document about cats: Cats are small furry mammals that purr.",
        title="Cat Facts",
    )
    doc2 = await client.create_document(
        "Document about dogs: Dogs are loyal companions that bark.",
        title="Dog Facts",
    )

    yield client, doc1.id, doc2.id

    client.close()


@pytest.mark.vcr()
async def test_search_filter_restricts_results(client_with_docs):
    """Test that search_filter restricts search to specified documents."""
    client, doc1_id, doc2_id = client_with_docs

    # Search without filter - should find both
    results_all = await client.search("animals mammals companions")
    assert len(results_all) >= 1

    # Search with filter for doc1 only
    filter_doc1 = f"id = '{doc1_id}'"
    results_filtered = await client.search(
        "animals mammals companions", filter=filter_doc1
    )

    # All results should be from doc1
    for result in results_filtered:
        assert result.document_id == doc1_id


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_research_graph_uses_search_filter(
    allow_model_requests, client_with_docs
):
    """Test that research graph passes search_filter to search operations."""
    client, doc1_id, doc2_id = client_with_docs

    # Track search calls to verify filter is passed
    search_calls = []
    original_search = client.search

    async def tracking_search(query, limit=None, search_type="hybrid", filter=None):
        search_calls.append({"query": query, "filter": filter})
        return await original_search(query, limit, search_type, filter)

    client.search = tracking_search

    graph = build_research_graph()

    # Create state with search_filter
    filter_clause = f"id = '{doc1_id}'"
    state = ResearchState(
        context=ResearchContext(original_question="Tell me about animals"),
        max_iterations=1,
        confidence_threshold=0.5,
        search_filter=filter_clause,
    )

    deps = ResearchDeps(client=client)

    await graph.run(state=state, deps=deps)

    # Verify search was called with the filter
    assert len(search_calls) > 0, "Expected search to be called"
    for call in search_calls:
        assert call["filter"] == filter_clause, (
            f"Expected filter '{filter_clause}', got '{call['filter']}'"
        )


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_search_filter_none_searches_all(allow_model_requests, client_with_docs):
    """Test that search_filter=None searches all documents."""
    client, doc1_id, doc2_id = client_with_docs

    # Track search calls
    search_calls = []
    original_search = client.search

    async def tracking_search(query, limit=None, search_type="hybrid", filter=None):
        search_calls.append({"query": query, "filter": filter})
        return await original_search(query, limit, search_type, filter)

    client.search = tracking_search

    graph = build_research_graph()

    # Create state without search_filter (None)
    state = ResearchState(
        context=ResearchContext(original_question="Tell me about animals"),
        max_iterations=1,
        confidence_threshold=0.5,
        search_filter=None,
    )

    deps = ResearchDeps(client=client)

    await graph.run(state=state, deps=deps)

    # Verify search was called with None filter
    assert len(search_calls) > 0, "Expected search to be called"
    for call in search_calls:
        assert call["filter"] is None, f"Expected filter None, got '{call['filter']}'"
