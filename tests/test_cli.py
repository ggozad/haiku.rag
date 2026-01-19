from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from haiku.rag.cli import _cli as cli
from haiku.rag.cli import cli as cli_wrapper
from haiku.rag.store.exceptions import MigrationRequiredError

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
        mock_app_instance.add_document_from_text.assert_called_once()
        _, kwargs = mock_app_instance.add_document_from_text.call_args
        assert kwargs.get("text") == "test document"
        assert kwargs.get("metadata") is None


def test_add_document_src():
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.add_document_from_source = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["add-src", "test.txt"])

        assert result.exit_code == 0
        mock_app_instance.add_document_from_source.assert_called_once()


def test_add_document_src_with_title():
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
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
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
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
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
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
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
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
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.get_document = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["get", "1"])

        assert result.exit_code == 0
        mock_app_instance.get_document.assert_called_once_with(doc_id="1")


def test_delete_document():
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.delete_document = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["delete", "1"])

        assert result.exit_code == 0
        mock_app_instance.delete_document.assert_called_once_with(doc_id="1")


def test_search():
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.search = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["search", "query"])

        assert result.exit_code == 0
        mock_app_instance.search.assert_called_once_with(
            query="query", limit=None, filter=None
        )


def test_serve_no_flags():
    """Test serve command fails without flags."""
    result = runner.invoke(cli, ["serve"])
    assert result.exit_code == 1
    assert "At least one service flag" in result.output


def test_serve_mcp_only():
    """Test serve command with MCP only."""
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.serve = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["serve", "--mcp"])

        assert result.exit_code == 0
        mock_app_instance.serve.assert_called_once()
        _, kwargs = mock_app_instance.serve.call_args
        assert kwargs["enable_monitor"] is False
        assert kwargs["enable_mcp"] is True
        assert kwargs["mcp_transport"] is None
        assert kwargs["mcp_port"] == 8001


def test_serve_mcp_stdio():
    """Test serve command with MCP stdio transport."""
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.serve = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["serve", "--mcp", "--stdio"])

        assert result.exit_code == 0
        mock_app_instance.serve.assert_called_once()
        _, kwargs = mock_app_instance.serve.call_args
        assert kwargs["mcp_transport"] == "stdio"


def test_serve_monitor_only():
    """Test serve command with monitor only."""
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.serve = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["serve", "--monitor"])

        assert result.exit_code == 0
        mock_app_instance.serve.assert_called_once()
        _, kwargs = mock_app_instance.serve.call_args
        assert kwargs["enable_monitor"] is True
        assert kwargs["enable_mcp"] is False


def test_serve_all_services():
    """Test serve command with all services."""
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.serve = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["serve", "--monitor", "--mcp"])

        assert result.exit_code == 0
        mock_app_instance.serve.assert_called_once()
        _, kwargs = mock_app_instance.serve.call_args
        assert kwargs["enable_monitor"] is True
        assert kwargs["enable_mcp"] is True


def test_serve_custom_ports():
    """Test serve command with custom MCP port."""
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.serve = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["serve", "--mcp", "--mcp-port", "9000"])

        assert result.exit_code == 0
        mock_app_instance.serve.assert_called_once()
        _, kwargs = mock_app_instance.serve.call_args
        assert kwargs["mcp_port"] == 9000


def test_serve_stdio_without_mcp():
    """Test serve command fails when --stdio is used without --mcp."""
    result = runner.invoke(cli, ["serve", "--stdio", "--monitor"])
    assert result.exit_code == 1
    assert "--stdio requires --mcp" in result.output


def test_ask():
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.ask = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["ask", "What is Python?"])

        assert result.exit_code == 0
        mock_app_instance.ask.assert_called_once_with(
            question="What is Python?",
            cite=False,
            deep=False,
            filter=None,
            background_context=None,
        )


def test_ask_with_cite():
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.ask = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["ask", "What is Python?", "--cite"])

        assert result.exit_code == 0
        mock_app_instance.ask.assert_called_once_with(
            question="What is Python?",
            cite=True,
            deep=False,
            filter=None,
            background_context=None,
        )


def test_ask_with_deep():
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.ask = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["ask", "What is Python?", "--deep"])

        assert result.exit_code == 0
        mock_app_instance.ask.assert_called_once_with(
            question="What is Python?",
            cite=False,
            deep=True,
            filter=None,
            background_context=None,
        )


def test_ask_with_deep_and_cite():
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.ask = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["ask", "What is Python?", "--deep", "--cite"])

        assert result.exit_code == 0
        mock_app_instance.ask.assert_called_once_with(
            question="What is Python?",
            cite=True,
            deep=True,
            filter=None,
            background_context=None,
        )


def test_init():
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.init = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["init"])

        assert result.exit_code == 0
        mock_app_instance.init.assert_called_once()


def test_info():
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.info = AsyncMock()
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        mock_app_instance.info.assert_called_once()


def test_add_document_src_directory(tmp_path):
    """Test adding documents from a directory recursively."""
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.add_document_from_source = AsyncMock()
        mock_app.return_value = mock_app_instance

        test_dir = tmp_path / "test_docs"
        test_dir.mkdir()
        (test_dir / "doc1.txt").write_text("doc1")
        (test_dir / "doc2.md").write_text("doc2")
        subdir = test_dir / "subdir"
        subdir.mkdir()
        (subdir / "doc3.pdf").write_text("doc3")

        result = runner.invoke(cli, ["add-src", str(test_dir)])

        assert result.exit_code == 0
        mock_app_instance.add_document_from_source.assert_called_once()
        call_args = mock_app_instance.add_document_from_source.call_args
        assert call_args[1]["source"] == str(test_dir)


def test_migrate_with_applied_migrations():
    """Test migrate command when migrations are applied."""
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.migrate.return_value = [
            "Add full-text search index",
            "Add metadata column",
        ]
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["migrate"])

        assert result.exit_code == 0
        mock_app_instance.migrate.assert_called_once()
        assert "Applied 2 migration(s)" in result.output
        assert "Add full-text search index" in result.output
        assert "Add metadata column" in result.output
        assert "Migration completed successfully" in result.output


def test_migrate_no_pending_migrations():
    """Test migrate command when no migrations are pending."""
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.migrate.return_value = []
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["migrate"])

        assert result.exit_code == 0
        mock_app_instance.migrate.assert_called_once()
        assert "No migrations pending" in result.output
        assert "Database is up to date" in result.output


def test_migrate_failure():
    """Test migrate command when migration fails."""
    with patch("haiku.rag.cli.HaikuRAGApp") as mock_app:
        mock_app_instance = MagicMock()
        mock_app_instance.migrate.side_effect = Exception("Migration failed")
        mock_app.return_value = mock_app_instance

        result = runner.invoke(cli, ["migrate"])

        assert result.exit_code == 1
        assert "Migration failed" in result.output


def test_cli_wrapper_catches_migration_required_error():
    """Test that cli() wrapper catches MigrationRequiredError and exits with code 1."""
    with patch("haiku.rag.cli._cli") as mock_cli:
        mock_cli.side_effect = MigrationRequiredError(
            "Database requires migration. Run 'haiku-rag migrate' to upgrade."
        )

        with patch("sys.exit") as mock_exit:
            cli_wrapper()
            mock_exit.assert_called_once_with(1)
