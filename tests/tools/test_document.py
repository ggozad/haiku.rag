import pytest

from haiku.rag.tools import ToolContext
from haiku.rag.tools.document import (
    DOCUMENT_NAMESPACE,
    DocumentInfo,
    DocumentListResponse,
    DocumentState,
    create_document_toolset,
)


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

    def test_document_state_defaults(self):
        """DocumentState initializes with empty accessed list."""
        state = DocumentState()
        assert state.accessed_documents == []


class TestDocumentToolset:
    """Tests for create_document_toolset."""

    def test_create_document_toolset_returns_function_toolset(
        self, doc_client, doc_config
    ):
        """create_document_toolset returns a FunctionToolset."""
        from pydantic_ai import FunctionToolset

        toolset = create_document_toolset(doc_client, doc_config)
        assert isinstance(toolset, FunctionToolset)

    def test_document_toolset_has_expected_tools(self, doc_client, doc_config):
        """The toolset includes list_documents, get_document, summarize_document."""
        toolset = create_document_toolset(doc_client, doc_config)

        assert "list_documents" in toolset.tools
        assert "get_document" in toolset.tools
        assert "summarize_document" in toolset.tools

    def test_document_toolset_registers_state(self, doc_client, doc_config):
        """Toolset registers DocumentState under DOCUMENT_NAMESPACE."""
        context = ToolContext()
        create_document_toolset(doc_client, doc_config, context=context)

        state = context.get(DOCUMENT_NAMESPACE)
        assert state is not None
        assert isinstance(state, DocumentState)


class TestDocumentToolExecution:
    """Tests for document tool execution."""

    @pytest.mark.asyncio
    async def test_list_documents_returns_paginated_results(
        self, doc_client, doc_config
    ):
        """list_documents returns DocumentListResponse."""
        toolset = create_document_toolset(doc_client, doc_config)

        list_tool = toolset.tools["list_documents"]
        result = await list_tool.function()

        assert isinstance(result, DocumentListResponse)
        assert result.total_documents == 2
        assert len(result.documents) == 2
        assert result.page == 1

    @pytest.mark.asyncio
    async def test_list_documents_pagination(self, doc_client, doc_config):
        """list_documents supports pagination."""
        toolset = create_document_toolset(doc_client, doc_config)

        list_tool = toolset.tools["list_documents"]
        result = await list_tool.function(page=2)

        # With only 2 documents and page_size=50, page 2 should be empty
        assert result.page == 2
        assert len(result.documents) == 0

    @pytest.mark.asyncio
    async def test_get_document_by_title(self, doc_client, doc_config):
        """get_document finds document by title."""
        toolset = create_document_toolset(doc_client, doc_config)

        get_tool = toolset.tools["get_document"]
        result = await get_tool.function("Python Guide")

        assert "Python Guide" in result
        assert "Python is a programming language" in result

    @pytest.mark.asyncio
    async def test_get_document_by_uri(self, doc_client, doc_config):
        """get_document finds document by URI."""
        toolset = create_document_toolset(doc_client, doc_config)

        get_tool = toolset.tools["get_document"]
        result = await get_tool.function("test://python")

        assert "Python Guide" in result

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, doc_client, doc_config):
        """get_document returns appropriate message when not found."""
        toolset = create_document_toolset(doc_client, doc_config)

        get_tool = toolset.tools["get_document"]
        result = await get_tool.function("nonexistent")

        assert "Document not found" in result

    @pytest.mark.asyncio
    async def test_get_document_tracks_in_state(self, doc_client, doc_config):
        """get_document tracks accessed documents in state."""
        context = ToolContext()
        toolset = create_document_toolset(doc_client, doc_config, context=context)

        get_tool = toolset.tools["get_document"]
        await get_tool.function("Python Guide")

        state = context.get(DOCUMENT_NAMESPACE)
        assert isinstance(state, DocumentState)
        assert len(state.accessed_documents) == 1
        assert state.accessed_documents[0].title == "Python Guide"

    @pytest.mark.asyncio
    async def test_list_documents_with_base_filter(self, doc_client, doc_config):
        """list_documents respects base_filter."""
        toolset = create_document_toolset(
            doc_client, doc_config, base_filter="title LIKE '%Python%'"
        )

        list_tool = toolset.tools["list_documents"]
        result = await list_tool.function()

        assert result.total_documents == 1
        assert result.documents[0].title == "Python Guide"


@pytest.fixture
def doc_client(temp_db_path):
    """Create a HaikuRAG client with test documents."""
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
def doc_config():
    """Default AppConfig for document tests."""
    from haiku.rag.config import Config

    return Config
