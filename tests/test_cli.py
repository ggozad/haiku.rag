from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from haiku.rag.cli import cli

runner = CliRunner()


def test_list_documents():
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.list_documents = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        mock_app_instance.list_documents.assert_called_once()


def test_add_document_text():
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.add_document_from_text = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["add", "test document"])

        assert result.exit_code == 0
        mock_app_instance.add_document_from_text.assert_called_once_with(
            text="test document"
        )


def test_add_document_src():
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.add_document_from_source = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["add-src", "test.txt"])

        assert result.exit_code == 0
        mock_app_instance.add_document_from_source.assert_called_once()


def test_get_document():
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.get_document = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["get", "1"])

        assert result.exit_code == 0
        mock_app_instance.get_document.assert_called_once_with(doc_id=1)


def test_delete_document():
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.delete_document = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["delete", "1"])

        assert result.exit_code == 0
        mock_app_instance.delete_document.assert_called_once_with(doc_id=1)


def test_search():
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.search = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["search", "query"])

        assert result.exit_code == 0
        mock_app_instance.search.assert_called_once_with(query="query", limit=5, k=60)


def test_serve():
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.serve = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["serve"])

        assert result.exit_code == 0
        mock_app_instance.serve.assert_called_once_with(transport=None)


def test_serve_stdio():
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.serve = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["serve", "--stdio"])

        assert result.exit_code == 0
        mock_app_instance.serve.assert_called_once_with(transport="stdio")


def test_serve_sse():
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.serve = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["serve", "--sse"])

        assert result.exit_code == 0
        mock_app_instance.serve.assert_called_once_with(transport="sse")


def test_serve_stdio_and_sse():
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.serve = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["serve", "--stdio", "--sse"])

        assert result.exit_code == 1
        assert "Error: Cannot use both --stdio and --http options" in result.stdout


def test_ask():
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.ask = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["ask", "What is Python?"])

        assert result.exit_code == 0
        mock_app_instance.ask.assert_called_once_with(
            question="What is Python?", cite=False
        )


def test_ask_with_cite():
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.ask = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["ask", "What is Python?", "--cite"])

        assert result.exit_code == 0
        mock_app_instance.ask.assert_called_once_with(
            question="What is Python?", cite=True
        )
