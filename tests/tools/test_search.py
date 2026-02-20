from pathlib import Path
from types import SimpleNamespace

import pytest

from haiku.rag.store.models import SearchResult
from haiku.rag.tools.search import create_search_toolset


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent.parent / "cassettes" / "test_search_tools")


def make_ctx(client):
    """Create a lightweight RunContext-like object for direct tool function calls."""
    return SimpleNamespace(deps=SimpleNamespace(client=client))


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
        toolset = create_search_toolset(search_config)

        search_tool = toolset.tools["search"]
        ctx = make_ctx(search_client)
        result = await search_tool.function(ctx, "Python")

        assert "Python" in result or "programming" in result
        assert "No results found" not in result

    @pytest.mark.asyncio
    async def test_search_with_no_results(self, temp_db_path, search_config):
        """Search tool returns appropriate message when no results."""
        from haiku.rag.client import HaikuRAG

        # Use empty database
        async with HaikuRAG(temp_db_path, create=True) as empty_client:
            toolset = create_search_toolset(search_config)

            search_tool = toolset.tools["search"]
            ctx = make_ctx(empty_client)
            result = await search_tool.function(ctx, "anything")

            assert result == "No results found."

    @pytest.mark.asyncio
    async def test_search_with_filter(self, search_client, search_config):
        """Search tool respects filter parameter."""
        accumulated: list[SearchResult] = []
        toolset = create_search_toolset(search_config, on_results=accumulated.extend)

        search_tool = toolset.tools["search"]
        ctx = make_ctx(search_client)
        await search_tool.function(ctx, "programming", filter="title LIKE '%Python%'")

        for r in accumulated:
            assert "JavaScript" not in (r.document_title or "")

    @pytest.mark.asyncio
    async def test_search_with_base_filter(self, search_client, search_config):
        """Search toolset respects base_filter parameter."""
        accumulated: list[SearchResult] = []
        toolset = create_search_toolset(
            search_config,
            base_filter="title LIKE '%Python%'",
            on_results=accumulated.extend,
        )

        search_tool = toolset.tools["search"]
        ctx = make_ctx(search_client)
        await search_tool.function(ctx, "programming")

        assert len(accumulated) > 0
        for r in accumulated:
            assert "JavaScript" not in (r.document_title or "")

    @pytest.mark.asyncio
    async def test_search_on_results_callback(self, search_client, search_config):
        """on_results callback receives search results."""
        accumulated: list[SearchResult] = []
        toolset = create_search_toolset(search_config, on_results=accumulated.extend)

        search_tool = toolset.tools["search"]
        ctx = make_ctx(search_client)
        await search_tool.function(ctx, "Python")

        assert len(accumulated) > 0
        assert any("Python" in r.content for r in accumulated)

    @pytest.mark.asyncio
    async def test_search_on_results_accumulates_across_calls(
        self, search_client, search_config
    ):
        """Multiple searches accumulate results via on_results callback."""
        accumulated: list[SearchResult] = []
        toolset = create_search_toolset(search_config, on_results=accumulated.extend)

        search_tool = toolset.tools["search"]
        ctx = make_ctx(search_client)
        await search_tool.function(ctx, "Python")
        first_count = len(accumulated)

        await search_tool.function(ctx, "JavaScript")
        assert len(accumulated) > first_count

    @pytest.mark.asyncio
    async def test_search_without_on_results(self, search_client, search_config):
        """Search works without on_results callback."""
        toolset = create_search_toolset(search_config)

        search_tool = toolset.tools["search"]
        ctx = make_ctx(search_client)
        result = await search_tool.function(ctx, "Python")

        assert "Python" in result or "programming" in result


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
