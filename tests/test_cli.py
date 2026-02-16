from unittest.mock import patch

import pytest
from click.exceptions import BadParameter
from typer.testing import CliRunner

from haiku.rag.cli import _cli as cli
from haiku.rag.cli import _parse_meta_options
from haiku.rag.cli import cli as cli_wrapper
from haiku.rag.store.exceptions import MigrationRequiredError

runner = CliRunner()


class TestParseMetaOptions:
    def test_empty_input(self):
        assert _parse_meta_options(None) == {}
        assert _parse_meta_options([]) == {}

    def test_simple_key_value(self):
        result = _parse_meta_options(["author=alice", "topic=notes"])
        assert result == {"author": "alice", "topic": "notes"}

    def test_missing_equals_raises(self):
        with pytest.raises(BadParameter):
            _parse_meta_options(["no_equals_here"])

    def test_empty_key_raises(self):
        with pytest.raises(BadParameter):
            _parse_meta_options(["=value"])

    def test_json_number(self):
        result = _parse_meta_options(["version=3"])
        assert result == {"version": 3}
        assert isinstance(result["version"], int)

    def test_json_float(self):
        result = _parse_meta_options(["score=3.14"])
        assert result == {"score": 3.14}
        assert isinstance(result["score"], float)

    def test_json_bool(self):
        result = _parse_meta_options(["published=true", "draft=false"])
        assert result == {"published": True, "draft": False}

    def test_json_null(self):
        result = _parse_meta_options(["empty=null"])
        assert result == {"empty": None}

    def test_json_array(self):
        result = _parse_meta_options(['tags=["a","b","c"]'])
        assert result == {"tags": ["a", "b", "c"]}

    def test_json_object(self):
        result = _parse_meta_options(['nested={"x": 1}'])
        assert result == {"nested": {"x": 1}}

    def test_plain_string_not_json(self):
        result = _parse_meta_options(["name=hello world"])
        assert result == {"name": "hello world"}
        assert isinstance(result["name"], str)

    def test_value_with_equals_sign(self):
        result = _parse_meta_options(["equation=a=b+c"])
        assert result == {"equation": "a=b+c"}


class TestServeValidation:
    def test_no_flags_fails(self):
        result = runner.invoke(cli, ["serve"])
        assert result.exit_code == 1
        assert "At least one service flag" in result.output

    def test_stdio_without_mcp_fails(self):
        result = runner.invoke(cli, ["serve", "--stdio", "--monitor"])
        assert result.exit_code == 1
        assert "--stdio requires --mcp" in result.output


class TestRebuildValidation:
    def test_embed_only_and_rechunk_mutually_exclusive(self):
        result = runner.invoke(
            cli, ["rebuild", "--embed-only", "--rechunk", "--db", "/tmp/fake.lancedb"]
        )
        assert result.exit_code == 1
        assert "mutually exclusive" in result.output


class TestCliMigrationError:
    def test_catches_migration_required_error(self):
        with patch("haiku.rag.cli._cli") as mock_cli:
            mock_cli.side_effect = MigrationRequiredError(
                "Database requires migration. Run 'haiku-rag migrate' to upgrade."
            )

            with pytest.raises(SystemExit) as exc_info:
                cli_wrapper()
            assert exc_info.value.code == 1
