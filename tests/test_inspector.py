from unittest.mock import patch

from typer.testing import CliRunner

from haiku.rag.cli import cli

runner = CliRunner()


def test_inspect_command():
    """Test inspect command launches inspector TUI."""
    with patch("haiku.rag.inspector.run_inspector") as mock_inspector:
        mock_inspector.return_value = None

        result = runner.invoke(cli, ["inspect"])

        assert result.exit_code == 0
        mock_inspector.assert_called_once()
