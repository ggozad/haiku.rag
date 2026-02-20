import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.mcp import create_mcp_server
from haiku.rag.store.models import Document, SearchResult
from haiku.rag.tools.document import DocumentInfo


@pytest.fixture(autouse=True)
def mock_embedder(monkeypatch):
    """Monkeypatch the embedder to return deterministic vectors."""
    import random

    from haiku.rag.embeddings import EmbedderWrapper

    async def fake_embed_query(self, text):
        random.seed(hash(text) % (2**32))
        return [random.random() for _ in range(2560)]

    async def fake_embed_documents(self, texts):
        result = []
        for t in texts:
            random.seed(hash(t) % (2**32))
            result.append([random.random() for _ in range(2560)])
        return result

    monkeypatch.setattr(EmbedderWrapper, "embed_query", fake_embed_query)
    monkeypatch.setattr(EmbedderWrapper, "embed_documents", fake_embed_documents)


@pytest.fixture
async def mcp_db(temp_db_path):
    """Create a test database with sample documents."""
    async with HaikuRAG(temp_db_path, create=True) as rag:
        await rag.create_document(
            "Artificial intelligence is transforming industries worldwide.",
            title="AI Overview",
            uri="test://ai-overview",
        )
        await rag.create_document(
            "Machine learning is a subset of artificial intelligence.",
            title="ML Basics",
            uri="test://ml-basics",
        )
    return temp_db_path


def _get_tool(mcp, name):
    """Get a tool function from an MCP server by name."""
    for tool in mcp._tool_manager._tools.values():
        if tool.fn.__name__ == name:
            return tool.fn
    raise ValueError(f"Tool {name!r} not found in MCP server")


class TestMCPReadTools:
    @pytest.mark.asyncio
    async def test_search_documents(self, mcp_db):
        mcp = create_mcp_server(mcp_db, read_only=True)
        search = _get_tool(mcp, "search_documents")

        results = await search(query="artificial intelligence")
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_documents_with_limit(self, mcp_db):
        mcp = create_mcp_server(mcp_db, read_only=True)
        search = _get_tool(mcp, "search_documents")

        results = await search(query="artificial intelligence", limit=1)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_document(self, mcp_db):
        mcp = create_mcp_server(mcp_db, read_only=True)
        get_doc = _get_tool(mcp, "get_document")

        # First get the ID via list
        list_docs = _get_tool(mcp, "list_documents")
        docs = await list_docs()
        doc_id = docs[0].id

        result = await get_doc(document_id=doc_id)
        assert isinstance(result, Document)
        assert result.content != ""
        assert result.title is not None

    @pytest.mark.asyncio
    async def test_get_document_excludes_docling_fields(self, mcp_db):
        mcp = create_mcp_server(mcp_db, read_only=True)
        get_doc = _get_tool(mcp, "get_document")

        list_docs = _get_tool(mcp, "list_documents")
        docs = await list_docs()
        doc_id = docs[0].id

        result = await get_doc(document_id=doc_id)
        serialized = result.model_dump(mode="json")
        assert "docling_document" not in serialized
        assert "docling_version" not in serialized

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, mcp_db):
        mcp = create_mcp_server(mcp_db, read_only=True)
        get_doc = _get_tool(mcp, "get_document")

        result = await get_doc(document_id="nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_documents(self, mcp_db):
        mcp = create_mcp_server(mcp_db, read_only=True)
        list_docs = _get_tool(mcp, "list_documents")

        results = await list_docs()
        assert len(results) == 2
        assert all(isinstance(r, DocumentInfo) for r in results)

    @pytest.mark.asyncio
    async def test_list_documents_with_limit(self, mcp_db):
        mcp = create_mcp_server(mcp_db, read_only=True)
        list_docs = _get_tool(mcp, "list_documents")

        results = await list_docs(limit=1)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_documents_with_filter(self, mcp_db):
        mcp = create_mcp_server(mcp_db, read_only=True)
        list_docs = _get_tool(mcp, "list_documents")

        results = await list_docs(filter="title = 'AI Overview'")
        assert len(results) == 1
        assert results[0].title == "AI Overview"


class TestMCPWriteTools:
    @pytest.mark.asyncio
    async def test_write_tools_registered_when_not_read_only(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True):
            pass
        mcp = create_mcp_server(temp_db_path, read_only=False)
        tool_names = list(mcp._tool_manager._tools.keys())
        assert "add_document_from_text" in tool_names
        assert "add_document_from_file" in tool_names
        assert "add_document_from_url" in tool_names
        assert "delete_document" in tool_names

    @pytest.mark.asyncio
    async def test_write_tools_not_registered_when_read_only(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True):
            pass
        mcp = create_mcp_server(temp_db_path, read_only=True)
        tool_names = list(mcp._tool_manager._tools.keys())
        assert "add_document_from_text" not in tool_names
        assert "delete_document" not in tool_names

    @pytest.mark.asyncio
    async def test_add_document_from_text(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True):
            pass
        mcp = create_mcp_server(temp_db_path, read_only=False)
        add_text = _get_tool(mcp, "add_document_from_text")

        doc_id = await add_text(content="Test content for MCP", title="MCP Test Doc")
        assert doc_id is not None

        get_doc = _get_tool(mcp, "get_document")
        doc = await get_doc(document_id=doc_id)
        assert doc.title == "MCP Test Doc"
        assert doc.content == "Test content for MCP"

    @pytest.mark.asyncio
    async def test_delete_document(self, mcp_db):
        mcp = create_mcp_server(mcp_db, read_only=False)
        list_docs = _get_tool(mcp, "list_documents")
        delete_doc = _get_tool(mcp, "delete_document")

        docs = await list_docs()
        assert len(docs) == 2

        result = await delete_doc(document_id=docs[0].id)
        assert result is True

        docs_after = await list_docs()
        assert len(docs_after) == 1

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, mcp_db):
        mcp = create_mcp_server(mcp_db, read_only=False)
        delete_doc = _get_tool(mcp, "delete_document")

        result = await delete_doc(document_id="nonexistent-id")
        assert result is False
