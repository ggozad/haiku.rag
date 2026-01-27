import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from haiku.rag.agents.research.models import ResearchReport
from haiku.rag.mcp import create_mcp_server
from haiku.rag.store.models.document import Document


@pytest.mark.asyncio
async def test_mcp_add_document_from_file():
    """Test add_document_from_file tool is properly wired."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.lancedb"
        mcp = create_mcp_server(db_path)

        mock_doc = Document(content="test", uri="file:///test.txt")
        mock_doc.id = "doc123"

        with patch("haiku.rag.mcp.HaikuRAG") as mock_rag_class:
            mock_rag = AsyncMock()
            mock_rag.create_document_from_source = AsyncMock(return_value=mock_doc)
            mock_rag_class.return_value.__aenter__ = AsyncMock(return_value=mock_rag)
            mock_rag_class.return_value.__aexit__ = AsyncMock(return_value=None)

            tools = await mcp.get_tools()
            add_file_tool = next(
                t for t in tools.values() if t.name == "add_document_from_file"
            )

            result = await add_file_tool.fn(file_path="/test.txt")
            assert result == "doc123"
            mock_rag.create_document_from_source.assert_called_once()


@pytest.mark.asyncio
async def test_mcp_ask_question():
    """Test ask_question tool is properly wired."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.lancedb"
        mcp = create_mcp_server(db_path)

        with patch("haiku.rag.mcp.HaikuRAG") as mock_rag_class:
            mock_rag = AsyncMock()
            mock_rag.ask = AsyncMock(return_value=("This is the answer", []))
            mock_rag_class.return_value.__aenter__ = AsyncMock(return_value=mock_rag)
            mock_rag_class.return_value.__aexit__ = AsyncMock(return_value=None)

            tools = await mcp.get_tools()
            ask_tool = next(t for t in tools.values() if t.name == "ask_question")

            result = await ask_tool.fn(question="What is this?", cite=False, deep=False)

            assert result == "This is the answer"
            mock_rag.ask.assert_called_once_with("What is this?")


@pytest.mark.asyncio
async def test_mcp_add_document_from_url():
    """Test add_document_from_url tool is properly wired."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.lancedb"
        mcp = create_mcp_server(db_path)

        mock_doc = Document(content="test", uri="https://example.com")
        mock_doc.id = "doc456"

        with patch("haiku.rag.mcp.HaikuRAG") as mock_rag_class:
            mock_rag = AsyncMock()
            mock_rag.create_document_from_source = AsyncMock(return_value=mock_doc)
            mock_rag_class.return_value.__aenter__ = AsyncMock(return_value=mock_rag)
            mock_rag_class.return_value.__aexit__ = AsyncMock(return_value=None)

            tools = await mcp.get_tools()
            add_url_tool = next(
                t for t in tools.values() if t.name == "add_document_from_url"
            )

            result = await add_url_tool.fn(url="https://example.com")
            assert result == "doc456"
            mock_rag.create_document_from_source.assert_called_once()


@pytest.mark.asyncio
async def test_mcp_add_document_from_text():
    """Test add_document_from_text tool is properly wired."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.lancedb"
        mcp = create_mcp_server(db_path)

        mock_doc = Document(content="test content", uri="text://test")
        mock_doc.id = "doc789"

        with patch("haiku.rag.mcp.HaikuRAG") as mock_rag_class:
            mock_rag = AsyncMock()
            mock_rag.create_document = AsyncMock(return_value=mock_doc)
            mock_rag_class.return_value.__aenter__ = AsyncMock(return_value=mock_rag)
            mock_rag_class.return_value.__aexit__ = AsyncMock(return_value=None)

            tools = await mcp.get_tools()
            add_text_tool = next(
                t for t in tools.values() if t.name == "add_document_from_text"
            )

            result = await add_text_tool.fn(content="test content", uri="text://test")

            assert result == "doc789"
            mock_rag.create_document.assert_called_once()


@pytest.mark.asyncio
async def test_mcp_search_documents():
    """Test search_documents tool is properly wired."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.lancedb"
        mcp = create_mcp_server(db_path)

        from haiku.rag.store.models import SearchResult

        mock_results = [
            SearchResult(content="Result 1", score=0.9, document_id="doc1"),
            SearchResult(content="Result 2", score=0.8, document_id="doc2"),
        ]

        with patch("haiku.rag.mcp.HaikuRAG") as mock_rag_class:
            mock_rag = AsyncMock()
            mock_rag.search = AsyncMock(return_value=mock_results)
            mock_rag_class.return_value.__aenter__ = AsyncMock(return_value=mock_rag)
            mock_rag_class.return_value.__aexit__ = AsyncMock(return_value=None)

            tools = await mcp.get_tools()
            search_tool = next(
                t for t in tools.values() if t.name == "search_documents"
            )

            result = await search_tool.fn(query="test query", limit=5)
            assert len(result) == 2
            assert result[0].document_id == "doc1"
            assert result[0].content == "Result 1"
            assert result[0].score == 0.9
            assert result[1].document_id == "doc2"
            mock_rag.search.assert_called_once_with("test query", limit=5)


@pytest.mark.asyncio
async def test_mcp_get_document():
    """Test get_document tool is properly wired."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.lancedb"
        mcp = create_mcp_server(db_path)

        from datetime import UTC, datetime

        mock_doc = Document(content="test", uri="file:///test.txt", title="Test Doc")
        mock_doc.id = "doc123"
        mock_doc.created_at = datetime(2024, 1, 1, tzinfo=UTC)
        mock_doc.updated_at = datetime(2024, 1, 2, tzinfo=UTC)

        with patch("haiku.rag.mcp.HaikuRAG") as mock_rag_class:
            mock_rag = AsyncMock()
            mock_rag.get_document_by_id = AsyncMock(return_value=mock_doc)
            mock_rag_class.return_value.__aenter__ = AsyncMock(return_value=mock_rag)
            mock_rag_class.return_value.__aexit__ = AsyncMock(return_value=None)

            tools = await mcp.get_tools()
            get_tool = next(t for t in tools.values() if t.name == "get_document")

            result = await get_tool.fn(document_id="doc123")
            assert result is not None
            assert result.id == "doc123"
            assert result.content == "test"
            assert result.title == "Test Doc"
            mock_rag.get_document_by_id.assert_called_once_with("doc123")


@pytest.mark.asyncio
async def test_mcp_list_documents():
    """Test list_documents tool is properly wired."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.lancedb"
        mcp = create_mcp_server(db_path)

        from datetime import UTC, datetime

        mock_doc1 = Document(content="test1", uri="file:///test1.txt")
        mock_doc1.id = "doc1"
        mock_doc1.created_at = datetime(2024, 1, 1, tzinfo=UTC)
        mock_doc1.updated_at = datetime(2024, 1, 1, tzinfo=UTC)

        mock_doc2 = Document(content="test2", uri="file:///test2.txt")
        mock_doc2.id = "doc2"
        mock_doc2.created_at = datetime(2024, 1, 2, tzinfo=UTC)
        mock_doc2.updated_at = datetime(2024, 1, 2, tzinfo=UTC)

        with patch("haiku.rag.mcp.HaikuRAG") as mock_rag_class:
            mock_rag = AsyncMock()
            mock_rag.list_documents = AsyncMock(return_value=[mock_doc1, mock_doc2])
            mock_rag_class.return_value.__aenter__ = AsyncMock(return_value=mock_rag)
            mock_rag_class.return_value.__aexit__ = AsyncMock(return_value=None)

            tools = await mcp.get_tools()
            list_tool = next(t for t in tools.values() if t.name == "list_documents")

            result = await list_tool.fn(limit=10, offset=0)
            assert len(result) == 2
            assert result[0].id == "doc1"
            assert result[1].id == "doc2"
            mock_rag.list_documents.assert_called_once_with(10, 0, None)


@pytest.mark.asyncio
async def test_mcp_delete_document():
    """Test delete_document tool is properly wired."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.lancedb"
        mcp = create_mcp_server(db_path)

        with patch("haiku.rag.mcp.HaikuRAG") as mock_rag_class:
            mock_rag = AsyncMock()
            mock_rag.delete_document = AsyncMock(return_value=True)
            mock_rag_class.return_value.__aenter__ = AsyncMock(return_value=mock_rag)
            mock_rag_class.return_value.__aexit__ = AsyncMock(return_value=None)

            tools = await mcp.get_tools()
            delete_tool = next(t for t in tools.values() if t.name == "delete_document")

            result = await delete_tool.fn(document_id="doc123")
            assert result is True
            mock_rag.delete_document.assert_called_once_with("doc123")


@pytest.mark.asyncio
async def test_mcp_ask_question_deep():
    """Test ask_question tool with deep=True uses research graph."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.lancedb"
        mcp = create_mcp_server(db_path)

        with (
            patch("haiku.rag.mcp.HaikuRAG") as mock_rag_class,
            patch(
                "haiku.rag.agents.research.graph.build_research_graph"
            ) as mock_graph_builder,
        ):
            mock_rag = AsyncMock()
            mock_rag_class.return_value.__aenter__ = AsyncMock(return_value=mock_rag)
            mock_rag_class.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_graph = AsyncMock()
            mock_result = AsyncMock()
            mock_result.executive_summary = "Deep answer from research"
            mock_graph.run = AsyncMock(return_value=mock_result)
            mock_graph_builder.return_value = mock_graph

            tools = await mcp.get_tools()
            ask_tool = next(t for t in tools.values() if t.name == "ask_question")

            # cite=False to avoid citation formatting in output
            result = await ask_tool.fn(question="Deep question?", cite=False, deep=True)

            assert result == "Deep answer from research"
            mock_graph.run.assert_called_once()


@pytest.mark.asyncio
async def test_mcp_research_question():
    """Test research_question tool is properly wired."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.lancedb"
        mcp = create_mcp_server(db_path)

        mock_report = ResearchReport(
            title="Research Title",
            executive_summary="Summary",
            main_findings=["Finding 1"],
            conclusions=["Conclusion 1"],
            recommendations=["Recommendation 1"],
        )

        with (
            patch("haiku.rag.mcp.HaikuRAG") as mock_rag_class,
            patch(
                "haiku.rag.agents.research.graph.build_research_graph"
            ) as mock_graph_builder,
        ):
            mock_rag = AsyncMock()
            mock_rag_class.return_value.__aenter__ = AsyncMock(return_value=mock_rag)
            mock_rag_class.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_graph = AsyncMock()
            mock_graph.run = AsyncMock(return_value=mock_report)
            mock_graph_builder.return_value = mock_graph

            tools = await mcp.get_tools()
            research_tool = next(
                t for t in tools.values() if t.name == "research_question"
            )

            result = await research_tool.fn(
                question="Research question?",
            )

            assert result is not None
            assert result.title == "Research Title"
            assert result.executive_summary == "Summary"
            mock_graph.run.assert_called_once()
