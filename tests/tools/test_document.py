from pathlib import Path
from types import SimpleNamespace

import pytest

from haiku.rag.tools.document import (
    DocumentInfo,
    DocumentListResponse,
    create_document_toolset,
)


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent.parent / "cassettes" / "test_document_tools")


def make_ctx(client, context=None):
    """Create a lightweight RunContext-like object for direct tool function calls."""
    return SimpleNamespace(deps=SimpleNamespace(client=client, tool_context=context))


class TestDocumentModels:
    """Tests for document models."""

    def test_document_info(self):
        """DocumentInfo holds basic document metadata."""
        info = DocumentInfo(title="Test Doc", uri="test://doc", created="2024-01-01")
        assert info.title == "Test Doc"
        assert info.uri == "test://doc"
        assert info.created == "2024-01-01"

    def test_document_list_response(self):
        """DocumentListResponse holds paginated results."""
        response = DocumentListResponse(
            documents=[
                DocumentInfo(title="Doc 1", uri="test://1", created="2024-01-01"),
                DocumentInfo(title="Doc 2", uri="test://2", created="2024-01-02"),
            ],
            page=1,
            total_pages=3,
            total_documents=125,
        )
        assert len(response.documents) == 2
        assert response.page == 1
        assert response.total_pages == 3
        assert response.total_documents == 125


@pytest.mark.vcr()
class TestDocumentToolset:
    """Tests for create_document_toolset."""

    def test_create_document_toolset_returns_function_toolset(self, doc_config):
        """create_document_toolset returns a FunctionToolset."""
        from pydantic_ai import FunctionToolset

        toolset = create_document_toolset(doc_config)
        assert isinstance(toolset, FunctionToolset)

    def test_document_toolset_has_expected_tools(self, doc_config):
        """The toolset includes list_documents, get_document, summarize_document."""
        toolset = create_document_toolset(doc_config)

        assert "list_documents" in toolset.tools
        assert "get_document" in toolset.tools
        assert "summarize_document" in toolset.tools


@pytest.mark.vcr()
class TestDocumentToolExecution:
    """Tests for document tool execution."""

    @pytest.mark.asyncio
    async def test_list_documents_returns_paginated_results(
        self, doc_client, doc_config
    ):
        """list_documents returns DocumentListResponse."""
        toolset = create_document_toolset(doc_config)

        list_tool = toolset.tools["list_documents"]
        ctx = make_ctx(doc_client)
        result = await list_tool.function(ctx)

        assert isinstance(result, DocumentListResponse)
        assert result.total_documents == 2
        assert len(result.documents) == 2
        assert result.page == 1

    @pytest.mark.asyncio
    async def test_list_documents_pagination(self, doc_client, doc_config):
        """list_documents supports pagination."""
        toolset = create_document_toolset(doc_config)

        list_tool = toolset.tools["list_documents"]
        ctx = make_ctx(doc_client)
        result = await list_tool.function(ctx, page=2)

        # With only 2 documents and page_size=50, page 2 should be empty
        assert result.page == 2
        assert len(result.documents) == 0

    @pytest.mark.asyncio
    async def test_get_document_by_title(self, doc_client, doc_config):
        """get_document finds document by title."""
        toolset = create_document_toolset(doc_config)

        get_tool = toolset.tools["get_document"]
        ctx = make_ctx(doc_client)
        result = await get_tool.function(ctx, "Python Guide")

        assert "Python Guide" in result
        assert "Python is a programming language" in result

    @pytest.mark.asyncio
    async def test_get_document_by_uri(self, doc_client, doc_config):
        """get_document finds document by URI."""
        toolset = create_document_toolset(doc_config)

        get_tool = toolset.tools["get_document"]
        ctx = make_ctx(doc_client)
        result = await get_tool.function(ctx, "test://python")

        assert "Python Guide" in result

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, doc_client, doc_config):
        """get_document returns appropriate message when not found."""
        toolset = create_document_toolset(doc_config)

        get_tool = toolset.tools["get_document"]
        ctx = make_ctx(doc_client)
        result = await get_tool.function(ctx, "nonexistent")

        assert "Document not found" in result

    @pytest.mark.asyncio
    async def test_list_documents_with_base_filter(self, doc_client, doc_config):
        """list_documents respects base_filter."""
        toolset = create_document_toolset(
            doc_config, base_filter="title LIKE '%Python%'"
        )

        list_tool = toolset.tools["list_documents"]
        ctx = make_ctx(doc_client)
        result = await list_tool.function(ctx)

        assert result.total_documents == 1
        assert result.documents[0].title == "Python Guide"


@pytest.mark.vcr()
class TestFindDocument:
    """Tests for find_document helper function."""

    @pytest.mark.asyncio
    async def test_find_document_partial_uri(self, doc_client):
        """find_document resolves partial URI match."""
        from haiku.rag.tools.document import find_document

        doc = await find_document(doc_client, "python")
        assert doc is not None
        assert doc.uri == "test://python"

    @pytest.mark.asyncio
    async def test_find_document_partial_title(self, doc_client):
        """find_document resolves partial title match."""
        from haiku.rag.tools.document import find_document

        doc = await find_document(doc_client, "JavaScript")
        assert doc is not None
        assert doc.title == "JavaScript Guide"


@pytest.mark.vcr()
class TestSummarizeDocumentTool:
    """Tests for summarize_document tool."""

    @pytest.mark.asyncio
    async def test_summarize_document_not_found(self, doc_client, doc_config):
        """summarize_document returns not-found message for nonexistent document."""
        toolset = create_document_toolset(doc_config)

        summarize_tool = toolset.tools["summarize_document"]
        ctx = make_ctx(doc_client)
        result = await summarize_tool.function(ctx, "nonexistent document")

        assert "Document not found" in result


@pytest.fixture
async def doc_client(temp_db_path):
    """Create a HaikuRAG client with test documents."""
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
def doc_config():
    """Default AppConfig for document tests."""
    from haiku.rag.config import Config

    return Config
