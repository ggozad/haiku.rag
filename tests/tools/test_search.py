import pytest

from haiku.rag.tools import ToolContext
from haiku.rag.tools.search import SEARCH_NAMESPACE, SearchState, create_search_toolset


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


class TestSearchToolset:
    """Tests for create_search_toolset."""

    def test_create_search_toolset_returns_function_toolset(
        self, search_client, search_config
    ):
        """create_search_toolset returns a FunctionToolset."""
        from pydantic_ai import FunctionToolset

        context = ToolContext()
        toolset = create_search_toolset(search_client, search_config, context)
        assert isinstance(toolset, FunctionToolset)

    def test_search_toolset_has_search_tool(self, search_client, search_config):
        """The toolset includes a 'search' tool."""
        context = ToolContext()
        toolset = create_search_toolset(search_client, search_config, context)

        # toolset.tools is a dict with tool names as keys
        assert "search" in toolset.tools

    def test_search_toolset_registers_state(self, search_client, search_config):
        """Toolset registers SearchState under SEARCH_NAMESPACE."""
        context = ToolContext()
        create_search_toolset(search_client, search_config, context)

        state = context.get(SEARCH_NAMESPACE)
        assert state is not None
        assert isinstance(state, SearchState)

    def test_search_toolset_uses_existing_state(self, search_client, search_config):
        """Toolset uses existing state if already registered."""
        from haiku.rag.store.models import SearchResult

        context = ToolContext()
        existing_state = SearchState()
        existing_state.results.append(
            SearchResult(content="pre-existing", score=0.5, chunk_id="pre1")
        )
        context.register(SEARCH_NAMESPACE, existing_state)

        create_search_toolset(search_client, search_config, context)

        state = context.get(SEARCH_NAMESPACE)
        assert isinstance(state, SearchState)
        assert len(state.results) == 1
        assert state.results[0].chunk_id == "pre1"


class TestSearchToolExecution:
    """Tests for search tool execution."""

    @pytest.mark.asyncio
    async def test_search_returns_formatted_results(self, search_client, search_config):
        """Search tool returns formatted results."""
        context = ToolContext()
        toolset = create_search_toolset(search_client, search_config, context)

        # Get the search function
        search_tool = toolset.tools["search"]
        result = await search_tool.function("Python")

        assert "Python" in result or "programming" in result
        assert "No results found" not in result

    @pytest.mark.asyncio
    async def test_search_accumulates_in_state(self, search_client, search_config):
        """Search tool accumulates results in SearchState."""
        context = ToolContext()
        toolset = create_search_toolset(search_client, search_config, context)

        # Run search
        search_tool = toolset.tools["search"]
        await search_tool.function("Python")

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
            toolset = create_search_toolset(empty_client, search_config, context)

            search_tool = toolset.tools["search"]
            result = await search_tool.function("anything")

            assert result == "No results found."

    @pytest.mark.asyncio
    async def test_search_with_filter(self, search_client, search_config):
        """Search tool respects filter parameter."""
        context = ToolContext()
        toolset = create_search_toolset(search_client, search_config, context)

        search_tool = toolset.tools["search"]
        # Filter to only Python documents
        await search_tool.function("programming", filter="title LIKE '%Python%'")

        # Should find Python but not JavaScript
        state = context.get(SEARCH_NAMESPACE)
        assert isinstance(state, SearchState)
        for r in state.results:
            assert "JavaScript" not in (r.document_title or "")

    @pytest.mark.asyncio
    async def test_search_without_context(self, search_client, search_config):
        """Search tool works without ToolContext."""
        toolset = create_search_toolset(search_client, search_config, context=None)

        search_tool = toolset.tools["search"]
        result = await search_tool.function("Python")

        # Should still return results
        assert "Python" in result or "programming" in result

    @pytest.mark.asyncio
    async def test_search_multiple_accumulates(self, search_client, search_config):
        """Multiple searches accumulate results in state."""
        context = ToolContext()
        toolset = create_search_toolset(search_client, search_config, context)

        search_tool = toolset.tools["search"]
        await search_tool.function("Python")
        state = context.get(SEARCH_NAMESPACE)
        assert isinstance(state, SearchState)
        first_count = len(state.results)

        await search_tool.function("JavaScript")
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
            search_client,
            search_config,
            context,
            base_filter="title LIKE '%Python%'",
        )

        search_tool = toolset.tools["search"]
        await search_tool.function("programming")

        # Should only find Python documents
        state = context.get(SEARCH_NAMESPACE)
        assert isinstance(state, SearchState)
        assert len(state.results) > 0
        for r in state.results:
            assert "JavaScript" not in (r.document_title or "")


@pytest.fixture
def search_client(temp_db_path):
    """Create a HaikuRAG client with test data for search tests."""
    import asyncio

    from haiku.rag.client import HaikuRAG

    async def setup():
        rag = HaikuRAG(temp_db_path, create=True)
        await rag.__aenter__()
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
        return rag

    return asyncio.get_event_loop().run_until_complete(setup())


@pytest.fixture
def search_config():
    """Default AppConfig for search tests."""
    from haiku.rag.config import Config

    return Config
