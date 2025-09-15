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
    results = await search_tool(ctx, query="climate change")

    # Verify results - should be list of (Chunk, float) tuples
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0][0].content == "Climate change is a global phenomenon"
    assert results[0][0].id == "chunk1"
    assert results[0][1] == 0.8  # score

    # Verify mock was called
    mock_client.search.assert_called_once_with("climate change", limit=5)
    mock_client.expand_context.assert_called_once()


@pytest.mark.asyncio
async def test_search_with_limit(mock_client, research_deps):
    """Test that search respects the limit parameter."""
    # Create more chunks than the limit
    mock_chunks = [
        create_mock_chunk("chunk1", "Content 1", 0.9),
        create_mock_chunk("chunk2", "Content 2", 0.8),
        create_mock_chunk("chunk3", "Content 3", 0.7),
        create_mock_chunk("chunk4", "Content 4", 0.6),
        create_mock_chunk("chunk5", "Content 5", 0.5),
    ]

    mock_client.search.return_value = mock_chunks[:3]
    mock_client.expand_context.return_value = mock_chunks[:3]

    agent = SearchSpecialistAgent(provider="openai", model="gpt-4")

    # Get the search tool
    search_tool = get_agent_tool(agent, "search")
    assert search_tool is not None

    # Test the tool with limit
    ctx = RunContext(deps=research_deps, model=TestModel(), usage=RunUsage())
    results = await search_tool(ctx, query="test query", limit=3)

    # Verify results respect limit
    assert isinstance(results, list)
    assert len(results) == 3
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    assert all(isinstance(r[0], Chunk) and isinstance(r[1], float) for r in results)

    # Verify mock was called with correct limit
    mock_client.search.assert_called_once_with("test query", limit=3)


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
    await search_tool(ctx, query="test query")

    # Verify context was updated
    assert len(research_deps.context.search_results) == 1
    assert research_deps.context.search_results[0]["query"] == "test query"
