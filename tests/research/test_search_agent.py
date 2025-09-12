"""Tests for the search specialist agent."""

from unittest.mock import AsyncMock, create_autospec

import pytest
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from haiku.rag.client import HaikuRAG
from haiku.rag.research.dependencies import ResearchContext, ResearchDependencies
from haiku.rag.research.search_agent import SearchSpecialistAgent
from haiku.rag.store.models.chunk import Chunk


@pytest.fixture
def mock_client():
    """Create a mock HaikuRAG client."""
    client = create_autospec(HaikuRAG, instance=True)
    client.search = AsyncMock()
    client.expand_context = AsyncMock()
    return client


@pytest.fixture
def research_context():
    """Create a research context for testing."""
    return ResearchContext(
        original_question="What is climate change?",
    )


@pytest.fixture
def research_deps(mock_client, research_context):
    """Create research dependencies for testing."""
    return ResearchDependencies(client=mock_client, context=research_context)


def create_mock_chunk(chunk_id: str, content: str, score: float = 0.8):
    """Helper to create mock chunk objects."""
    return Chunk(
        id=chunk_id,
        document_id=f"doc_{chunk_id}",
        content=content,
        document_uri=f"doc_{chunk_id}.md",
        metadata={},
    ), score


def get_agent_tool(agent, tool_name: str):
    """Helper to get a tool from an agent by name."""
    tools = agent.agent._function_toolset.tools
    if tool_name in tools:
        return tools[tool_name].function
    return None


@pytest.mark.asyncio
async def test_search_agent_has_search_tool():
    """Test that the search agent registers a search tool."""
    test_model = TestModel()
    # Use a valid provider for initialization
    agent = SearchSpecialistAgent(provider="openai", model="gpt-4")

    # Run agent with TestModel to check tools
    with agent.agent.override(model=test_model):
        await agent.agent.run(
            "test",
            deps=ResearchDependencies(
                client=create_autospec(HaikuRAG, instance=True),
                context=ResearchContext(original_question="test"),
            ),
        )

    # Verify the search tool was registered
    assert test_model.last_model_request_parameters is not None
    tools = test_model.last_model_request_parameters.function_tools
    assert tools is not None
    assert len(tools) == 1
    assert tools[0].name == "search"


@pytest.mark.asyncio
async def test_search_single_query(mock_client, research_deps):
    """Test that search tool is called with single query."""
    # Setup mock responses
    mock_chunks = [
        create_mock_chunk("chunk1", "Climate change is a global phenomenon"),
        create_mock_chunk("chunk2", "Rising temperatures affect ecosystems"),
    ]

    mock_client.search.return_value = mock_chunks[:1]
    mock_client.expand_context.return_value = mock_chunks

    # Create agent
    agent = SearchSpecialistAgent(provider="openai", model="gpt-4")

    # Get the search tool
    search_tool = get_agent_tool(agent, "search")
    assert search_tool is not None

    # Test the tool
    ctx = RunContext(deps=research_deps, model=TestModel(), usage=RunUsage())
    results = await search_tool(ctx, queries="climate change")

    # Verify results
    assert len(results) == 2
    assert results[0].content == "Climate change is a global phenomenon"
    assert results[0].metadata["chunk_id"] == "chunk1"

    # Verify mock was called
    mock_client.search.assert_called_once_with("climate change", limit=5)
    mock_client.expand_context.assert_called_once()


@pytest.mark.asyncio
async def test_search_multiple_queries_deduplication(mock_client, research_deps):
    """Test deduplication when searching with multiple queries."""
    # Create chunks with duplicate IDs across queries
    chunks_q1 = [
        create_mock_chunk("chunk1", "Content 1", 0.9),
        create_mock_chunk("chunk2", "Content 2", 0.7),
    ]
    chunks_q2 = [
        create_mock_chunk("chunk1", "Content 1", 0.9),  # Duplicate
        create_mock_chunk("chunk3", "Content 3", 0.8),
    ]

    mock_client.search.side_effect = [[chunks_q1[0]], [chunks_q2[0]]]
    mock_client.expand_context.side_effect = [chunks_q1, chunks_q2]

    agent = SearchSpecialistAgent(provider="openai", model="gpt-4")

    # Get the search tool
    search_tool = get_agent_tool(agent, "search")
    assert search_tool is not None

    # Test the tool
    ctx = RunContext(deps=research_deps, model=TestModel(), usage=RunUsage())
    results = await search_tool(ctx, queries=["query1", "query2"])

    # Check deduplication - chunk1 should appear only once
    chunk_ids = [r.metadata["chunk_id"] for r in results]
    assert chunk_ids.count("chunk1") == 1
    assert "chunk2" in chunk_ids
    assert "chunk3" in chunk_ids

    # Verify sorting by score
    assert all(
        results[i].score >= results[i + 1].score for i in range(len(results) - 1)
    )


@pytest.mark.asyncio
async def test_search_updates_context(mock_client, research_deps):
    """Test that search results are stored in context."""
    mock_chunks = [create_mock_chunk("chunk1", "Test content")]

    mock_client.search.return_value = []
    mock_client.expand_context.return_value = mock_chunks

    agent = SearchSpecialistAgent(provider="openai", model="gpt-4")

    # Get the search tool
    search_tool = get_agent_tool(agent, "search")
    assert search_tool is not None

    # Test the tool
    ctx = RunContext(deps=research_deps, model=TestModel(), usage=RunUsage())
    await search_tool(ctx, queries="test query")

    # Verify context was updated
    assert len(research_deps.context.search_results) == 1
    assert research_deps.context.search_results[0]["query"] == "test query"
