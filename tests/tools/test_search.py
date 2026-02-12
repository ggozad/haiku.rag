from types import SimpleNamespace

import pytest

from haiku.rag.tools import ToolContext
from haiku.rag.tools.search import SEARCH_NAMESPACE, SearchState, create_search_toolset


def make_ctx(client, context=None):
    """Create a lightweight RunContext-like object for direct tool function calls."""
    return SimpleNamespace(deps=SimpleNamespace(client=client, tool_context=context))


class TestSearchState:
    """Tests for SearchState model."""

    def test_search_state_defaults(self):
        """SearchState initializes with empty results."""
        state = SearchState()
        assert state.results == []

    def test_search_state_add_results(self):
        """Can add results to SearchState."""
        from haiku.rag.store.models import SearchResult

        state = SearchState()
        result = SearchResult(content="test content", score=0.9, chunk_id="chunk1")
        state.results.append(result)
        assert len(state.results) == 1
        assert state.results[0].chunk_id == "chunk1"

    def test_search_state_serialization(self):
        """SearchState serializes and deserializes correctly."""
        from haiku.rag.store.models import SearchResult

        state = SearchState()
        state.results.append(
            SearchResult(
                content="test",
                score=0.8,
                chunk_id="c1",
                document_title="Doc Title",
            )
        )

        # Serialize
        data = state.model_dump()
        assert "results" in data
        assert len(data["results"]) == 1

        # Deserialize
        restored = SearchState.model_validate(data)
        assert len(restored.results) == 1
        assert restored.results[0].chunk_id == "c1"


@pytest.mark.vcr()
class TestSearchToolset:
    """Tests for create_search_toolset."""

    def test_create_search_toolset_returns_function_toolset(self, search_config):
        """create_search_toolset returns a FunctionToolset."""
        from pydantic_ai import FunctionToolset

        toolset = create_search_toolset(search_config)
        assert isinstance(toolset, FunctionToolset)

    def test_search_toolset_has_search_tool(self, search_config):
        """The toolset includes a 'search' tool."""
        toolset = create_search_toolset(search_config)

        # toolset.tools is a dict with tool names as keys
        assert "search" in toolset.tools


@pytest.mark.vcr()
class TestSearchToolExecution:
    """Tests for search tool execution."""

    @pytest.mark.asyncio
    async def test_search_returns_formatted_results(self, search_client, search_config):
        """Search tool returns formatted results."""
        context = ToolContext()
        toolset = create_search_toolset(search_config)

        # Get the search function
        search_tool = toolset.tools["search"]
        ctx = make_ctx(search_client, context)
        result = await search_tool.function(ctx, "Python")

        assert "Python" in result or "programming" in result
        assert "No results found" not in result

    @pytest.mark.asyncio
    async def test_search_accumulates_in_state(self, search_client, search_config):
        """Search tool accumulates results in SearchState."""
        context = ToolContext()
        toolset = create_search_toolset(search_config)

        # Run search
        search_tool = toolset.tools["search"]
        ctx = make_ctx(search_client, context)
        await search_tool.function(ctx, "Python")

        # Check state was updated
        state = context.get(SEARCH_NAMESPACE)
        assert isinstance(state, SearchState)
        assert len(state.results) > 0
        assert any("Python" in r.content for r in state.results)

    @pytest.mark.asyncio
    async def test_search_with_no_results(self, temp_db_path, search_config):
        """Search tool returns appropriate message when no results."""
        from haiku.rag.client import HaikuRAG

        # Use empty database
        async with HaikuRAG(temp_db_path, create=True) as empty_client:
            context = ToolContext()
            toolset = create_search_toolset(search_config)

            search_tool = toolset.tools["search"]
            ctx = make_ctx(empty_client, context)
            result = await search_tool.function(ctx, "anything")

            assert result == "No results found."

    @pytest.mark.asyncio
    async def test_search_with_filter(self, search_client, search_config):
        """Search tool respects filter parameter."""
        context = ToolContext()
        toolset = create_search_toolset(search_config)

        search_tool = toolset.tools["search"]
        ctx = make_ctx(search_client, context)
        # Filter to only Python documents
        await search_tool.function(ctx, "programming", filter="title LIKE '%Python%'")

        # Should find Python but not JavaScript
        state = context.get(SEARCH_NAMESPACE)
        assert isinstance(state, SearchState)
        for r in state.results:
            assert "JavaScript" not in (r.document_title or "")

    @pytest.mark.asyncio
    async def test_search_without_context(self, search_client, search_config):
        """Search tool works without ToolContext."""
        toolset = create_search_toolset(search_config)

        search_tool = toolset.tools["search"]
        ctx = make_ctx(search_client, None)
        result = await search_tool.function(ctx, "Python")

        # Should still return results
        assert "Python" in result or "programming" in result

    @pytest.mark.asyncio
    async def test_search_multiple_accumulates(self, search_client, search_config):
        """Multiple searches accumulate results in state."""
        context = ToolContext()
        toolset = create_search_toolset(search_config)

        search_tool = toolset.tools["search"]
        ctx = make_ctx(search_client, context)
        await search_tool.function(ctx, "Python")
        state = context.get(SEARCH_NAMESPACE)
        assert isinstance(state, SearchState)
        first_count = len(state.results)

        await search_tool.function(ctx, "JavaScript")
        state = context.get(SEARCH_NAMESPACE)
        assert isinstance(state, SearchState)
        second_count = len(state.results)

        assert second_count > first_count

    @pytest.mark.asyncio
    async def test_search_with_base_filter(self, search_client, search_config):
        """Search toolset respects base_filter parameter."""
        context = ToolContext()
        # Create toolset with base_filter for Python documents only
        toolset = create_search_toolset(
            search_config,
            base_filter="title LIKE '%Python%'",
        )

        search_tool = toolset.tools["search"]
        ctx = make_ctx(search_client, context)
        await search_tool.function(ctx, "programming")

        # Should only find Python documents
        state = context.get(SEARCH_NAMESPACE)
        assert isinstance(state, SearchState)
        assert len(state.results) > 0
        for r in state.results:
            assert "JavaScript" not in (r.document_title or "")


@pytest.fixture
async def search_client(temp_db_path):
    """Create a HaikuRAG client with test data for search tests."""
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(temp_db_path, create=True) as rag:
        await rag.create_document(
            "Python is a programming language. It is widely used for web development.",
            uri="test://python",
            title="Python Guide",
        )
        await rag.create_document(
            "JavaScript runs in the browser. It powers interactive web pages.",
            uri="test://javascript",
            title="JavaScript Guide",
        )
        yield rag


@pytest.fixture
def search_config():
    """Default AppConfig for search tests."""
    from haiku.rag.config import Config

    return Config
