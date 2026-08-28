from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai import ToolFailed

from haiku.rag.store.models import SearchResult
from haiku.rag.tools.search import create_search_toolset


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent.parent / "cassettes" / "test_search_tools")


def make_ctx(client, run_id="test-run"):
    """Create a lightweight RunContext-like object for direct tool function calls."""
    return SimpleNamespace(deps=SimpleNamespace(client=client), run_id=run_id)


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


class TestNamingTheCollection:
    """The generic tool has no source selector, so what the client covers is
    what the search spans."""

    @staticmethod
    def _client(covers_multiple: bool, source: str | None, *, pictures: bool = False):
        from unittest.mock import AsyncMock

        results = [
            SearchResult(
                content="body",
                score=0.9,
                source=source,
                chunk_id="c1",
                document_id="d1",
                document_title="Report",
                image_data={"#/pictures/0": _png_b64()} if pictures else None,
            )
        ]
        return SimpleNamespace(
            covers_multiple=covers_multiple,
            search=AsyncMock(return_value=results),
            expand_context=AsyncMock(return_value=results),
        )

    @pytest.mark.asyncio
    async def test_a_client_covering_a_set_names_each_result(self, search_config):
        toolset = create_search_toolset(search_config)
        client = self._client(covers_multiple=True, source="alpha")

        text = await toolset.tools["search"].function(make_ctx(client), "cats")

        assert "Collection: alpha" in text

    @pytest.mark.asyncio
    async def test_one_named_collection_is_not_named(self, search_config):
        toolset = create_search_toolset(search_config)
        client = self._client(covers_multiple=False, source="alpha")

        text = await toolset.tools["search"].function(make_ctx(client), "cats")

        assert "Collection" not in text

    @staticmethod
    def _seeing_config(search_config):
        config = search_config.model_copy(deep=True)
        config.qa.model.vision = True
        return config

    @pytest.mark.asyncio
    async def test_a_client_covering_a_set_names_each_image(self, search_config):
        """Images travel beside the results and are labelled the same way."""
        from pydantic_ai.messages import ToolReturn

        toolset = create_search_toolset(self._seeing_config(search_config))
        client = self._client(covers_multiple=True, source="alpha", pictures=True)

        returned = await toolset.tools["search"].function(make_ctx(client), "cats")

        assert isinstance(returned, ToolReturn)
        labels = [item for item in returned.content if isinstance(item, str)]
        assert "Collection: alpha." in labels[0]

    @pytest.mark.asyncio
    async def test_one_named_collection_is_not_named_on_an_image(self, search_config):
        from pydantic_ai.messages import ToolReturn

        toolset = create_search_toolset(self._seeing_config(search_config))
        client = self._client(covers_multiple=False, source="alpha", pictures=True)

        returned = await toolset.tools["search"].function(make_ctx(client), "cats")

        assert isinstance(returned, ToolReturn)
        assert not [
            item
            for item in returned.content
            if isinstance(item, str) and "Collection" in item
        ]


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


@pytest.mark.vcr()
class TestSearchMaxSearches:
    """Tests for max_searches cap on search toolset."""

    @pytest.mark.asyncio
    async def test_searches_within_limit_return_results(
        self, search_client, search_config
    ):
        """Searches within max_searches return normal results."""
        toolset = create_search_toolset(search_config, max_searches=2)
        search_tool = toolset.tools["search"]
        ctx = make_ctx(search_client)

        assert await search_tool.function(ctx, "Python")
        assert await search_tool.function(ctx, "JavaScript")

    @pytest.mark.asyncio
    async def test_searches_beyond_limit_fail_the_tool(
        self, search_client, search_config
    ):
        """Searches beyond max_searches fail with the limit message."""
        toolset = create_search_toolset(search_config, max_searches=1)
        search_tool = toolset.tools["search"]
        ctx = make_ctx(search_client)

        assert await search_tool.function(ctx, "Python")

        with pytest.raises(ToolFailed, match="Search limit reached"):
            await search_tool.function(ctx, "JavaScript")

    @pytest.mark.asyncio
    async def test_counter_resets_across_runs(self, search_client, search_config):
        """Search counter resets when run_id changes (new agent run)."""
        toolset = create_search_toolset(search_config, max_searches=1)
        search_tool = toolset.tools["search"]

        ctx_run1 = make_ctx(search_client, run_id="run-1")
        assert await search_tool.function(ctx_run1, "Python")

        with pytest.raises(ToolFailed, match="Search limit reached"):
            await search_tool.function(ctx_run1, "JavaScript")

        ctx_run2 = make_ctx(search_client, run_id="run-2")
        assert await search_tool.function(ctx_run2, "Python")

    @pytest.mark.asyncio
    async def test_no_limit_by_default(self, search_client, search_config):
        """Without max_searches, searches are unlimited."""
        toolset = create_search_toolset(search_config)
        search_tool = toolset.tools["search"]
        ctx = make_ctx(search_client)

        for _ in range(5):
            assert await search_tool.function(ctx, "Python")


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
    from haiku.rag.config import get_config

    return get_config()


def _png_b64():
    import base64
    from io import BytesIO

    from PIL import Image as PILImage

    buf = BytesIO()
    PILImage.new("RGB", (4, 4), "red").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class TestBuildImageContentFromResults:
    """Picture bytes are attached once per (document, self_ref) pair, and labelled."""

    def test_results_without_image_data_contribute_nothing(self):
        from haiku.rag.tools.search import build_image_content_from_results

        results = [
            SearchResult(content="text only", score=0.5, chunk_id="c1", image_data=None)
        ]

        assert build_image_content_from_results(results) == []

    def test_duplicate_document_and_ref_is_attached_once(self):
        from pydantic_ai.messages import BinaryContent

        from haiku.rag.tools.search import build_image_content_from_results

        shared = {"#/pictures/0": _png_b64()}
        results = [
            SearchResult(
                content="a",
                score=0.9,
                chunk_id="c1",
                document_id="doc-1",
                image_data=shared,
            ),
            SearchResult(
                content="b",
                score=0.8,
                chunk_id="c2",
                document_id="doc-1",
                image_data=shared,
            ),
        ]

        content = build_image_content_from_results(results)

        images = [item for item in content if isinstance(item, BinaryContent)]
        assert len(images) == 1

    @staticmethod
    def _one_picture_in_two_collections():
        """A document copied into another collection keeps its ids and refs."""
        shared = {"#/pictures/0": _png_b64()}
        return [
            SearchResult(
                content="a",
                score=0.9,
                chunk_id="c1",
                document_id="doc-1",
                source="papers",
                image_data=shared,
            ),
            SearchResult(
                content="a",
                score=0.8,
                chunk_id="c1",
                document_id="doc-1",
                source="wiki",
                image_data=shared,
            ),
        ]

    def test_the_same_picture_in_two_collections_is_attached_from_each(self):
        from pydantic_ai.messages import BinaryContent

        from haiku.rag.tools.search import build_image_content_from_results

        content = build_image_content_from_results(
            self._one_picture_in_two_collections()
        )

        images = [item for item in content if isinstance(item, BinaryContent)]
        assert len(images) == 2

    def test_each_image_is_labelled_with_the_collection_it_came_from(self):
        """Nothing else tells the two apart: same chunk id, same document, same
        reference."""
        from haiku.rag.tools.search import build_image_content_from_results

        content = build_image_content_from_results(
            self._one_picture_in_two_collections(), include_collection=True
        )

        labels = [item for item in content if isinstance(item, str)]
        assert "Collection: papers." in labels[0]
        assert "Collection: wiki." in labels[1]

    def test_an_unasked_for_collection_is_not_named_on_an_image(self):
        from haiku.rag.tools.search import build_image_content_from_results

        content = build_image_content_from_results(
            self._one_picture_in_two_collections()
        )

        assert not [
            item for item in content if isinstance(item, str) and "Collection" in item
        ]

    def test_each_image_is_labelled_with_the_result_it_belongs_to(self):
        """Label every picture, not just the batch.

        ``ToolReturn.content`` reaches the model as a user-role message, and one
        leading note does not override that: with a single note on the wire,
        gemma4-26b still reasoned "the user also provided images in the prompt".
        A label adjacent to each picture also names the chunk to cite for it,
        which ``BinaryContent.identifier`` cannot do — it does not survive
        serialization to the vision API.
        """
        from pydantic_ai.messages import BinaryContent

        from haiku.rag.tools.search import build_image_content_from_results

        results = [
            SearchResult(
                content="a",
                score=0.9,
                chunk_id="c1",
                document_id="doc-1",
                image_data={"#/pictures/0": _png_b64()},
            ),
            SearchResult(
                content="b",
                score=0.8,
                chunk_id="c2",
                document_id="doc-2",
                image_data={"#/pictures/3": _png_b64()},
            ),
        ]

        content = build_image_content_from_results(results)

        # label, image, label, image — each picture preceded by its own line.
        assert [type(item) is str for item in content] == [True, False, True, False]
        assert isinstance(content[1], BinaryContent)
        assert isinstance(content[3], BinaryContent)

        first, second = content[0], content[2]
        assert isinstance(first, str) and isinstance(second, str)
        assert "c1" in first and "#/pictures/0" in first
        assert "c2" in second and "#/pictures/3" in second
        assert "1 of 2" in first and "2 of 2" in second
        for label in (first, second):
            assert "not provided by the user" in label.lower()
