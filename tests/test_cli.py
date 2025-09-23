from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from haiku.rag.cli import cli

runner = CliRunner()


def test_list_documents():
    with patch("haiku.rag.app.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.list_documents = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        mock_app_instance.list_documents.assert_called_once()


def test_add_document_text():
    with patch("haiku.rag.app.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.add_document_from_text = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["add", "test document"])

        assert result.exit_code == 0
        mock_app_instance.add_document_from_text.assert_called_once()
        _, kwargs = mock_app_instance.add_document_from_text.call_args
        assert kwargs.get("text") == "test document"
        assert kwargs.get("metadata") is None


def test_add_document_src():
    with patch("haiku.rag.app.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.add_document_from_source = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["add-src", "test.txt"])

        assert result.exit_code == 0
        mock_app_instance.add_document_from_source.assert_called_once()


def test_add_document_src_with_title():
    with patch("haiku.rag.app.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.add_document_from_source = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["add-src", "test.txt", "--title", "Nice Name"])

        assert result.exit_code == 0
        mock_app_instance.add_document_from_source.assert_called_once()
        # Verify title is forwarded (inspect call kwargs)
        _, kwargs = mock_app_instance.add_document_from_source.call_args
    assert kwargs.get("title") == "Nice Name"
    assert kwargs.get("source") == "test.txt"


def test_add_document_text_with_meta():
    with patch("haiku.rag.app.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.add_document_from_text = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(
            cli,
            [
                "add",
                "some text",
                "--meta",
                "author=alice",
                "--meta",
                "topic=notes",
            ],
        )

        assert result.exit_code == 0
        mock_app_instance.add_document_from_text.assert_called_once()
        _, kwargs = mock_app_instance.add_document_from_text.call_args
        assert kwargs.get("text") == "some text"
        assert kwargs.get("metadata") == {"author": "alice", "topic": "notes"}


def test_add_document_src_with_meta():
    with patch("haiku.rag.app.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.add_document_from_source = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(
            cli,
            [
                "add-src",
                "test.txt",
                "--meta",
                "source=manual",
                "--meta",
                "lang=en",
            ],
        )

        assert result.exit_code == 0
        mock_app_instance.add_document_from_source.assert_called_once()
        _, kwargs = mock_app_instance.add_document_from_source.call_args
        assert kwargs.get("source") == "test.txt"
        assert kwargs.get("metadata") == {"source": "manual", "lang": "en"}


def test_add_document_text_with_numeric_meta():
    with patch("haiku.rag.app.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.add_document_from_text = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(
            cli,
            [
                "add",
                "some text",
                "--meta",
                "version=3",
                "--meta",
                "published=true",
            ],
        )

        assert result.exit_code == 0
        mock_app_instance.add_document_from_text.assert_called_once()
        _, kwargs = mock_app_instance.add_document_from_text.call_args
        assert kwargs.get("text") == "some text"
        assert kwargs.get("metadata") == {"version": 3, "published": True}


def test_get_document():
    with patch("haiku.rag.app.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.get_document = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["get", "1"])

        assert result.exit_code == 0
        mock_app_instance.get_document.assert_called_once_with(doc_id="1")


def test_delete_document():
    with patch("haiku.rag.app.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.delete_document = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["delete", "1"])

        assert result.exit_code == 0
        mock_app_instance.delete_document.assert_called_once_with(doc_id="1")


def test_search():
    with patch("haiku.rag.app.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.search = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["search", "query"])

        assert result.exit_code == 0
        mock_app_instance.search.assert_called_once_with(query="query", limit=5)


def test_serve():
    with patch("haiku.rag.app.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.serve = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["serve"])

        assert result.exit_code == 0
        mock_app_instance.serve.assert_called_once_with(transport=None)


def test_serve_stdio():
    with patch("haiku.rag.app.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.serve = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["serve", "--stdio"])

        assert result.exit_code == 0
        mock_app_instance.serve.assert_called_once_with(transport="stdio")


def test_ask():
    with patch("haiku.rag.app.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.ask = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["ask", "What is Python?"])

        assert result.exit_code == 0
        mock_app_instance.ask.assert_called_once_with(
            question="What is Python?", cite=False
        )


def test_ask_with_cite():
    with patch("haiku.rag.app.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.ask = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["ask", "What is Python?", "--cite"])

        assert result.exit_code == 0
    mock_app_instance.ask.assert_called_once_with(question="What is Python?", cite=True)


def test_info():
    with patch("haiku.rag.app.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.info = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        mock_app_instance.info.assert_called_once()
