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


class TestTagCommands:
    def test_tag_round_trip(self, temp_db_path):
        db = str(temp_db_path)

        result = runner.invoke(cli, ["init", "--db", db])
        assert result.exit_code == 0

        result = runner.invoke(cli, ["tag", "create", "release-1", "--db", db])
        assert result.exit_code == 0
        assert "release-1" in result.output

        result = runner.invoke(cli, ["tag", "list", "--db", db])
        assert result.exit_code == 0
        assert "release-1" in result.output
        assert "partial" not in result.output

        result = runner.invoke(cli, ["history", "--db", db, "-t", "documents"])
        assert result.exit_code == 0
        assert "release-1" in result.output

        result = runner.invoke(cli, ["tag", "create", "release-1", "--db", db])
        assert result.exit_code == 1
        assert "already exists" in result.output

        result = runner.invoke(cli, ["tag", "delete", "release-1", "--db", db])
        assert result.exit_code == 0

        result = runner.invoke(cli, ["tag", "list", "--db", db])
        assert result.exit_code == 0
        assert "No tags" in result.output

        result = runner.invoke(cli, ["tag", "delete", "release-1", "--db", db])
        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_tag_create_rejected_when_migrations_pending(self, temp_db_path):
        """A writable tag operation must hit the migration gate and must not
        mutate a legacy database (e.g. by creating missing tables)."""
        import asyncio

        import lancedb

        from haiku.rag.store.engine import Store

        async def _prepare_legacy_db():
            async with Store(temp_db_path, create=True) as store:
                await store.set_haiku_version("0.19.0")
            db = await lancedb.connect_async(temp_db_path.absolute())
            await db.drop_table("document_meta")
            db.close()

        asyncio.run(_prepare_legacy_db())

        result = runner.invoke(
            cli, ["tag", "create", "release-1", "--db", str(temp_db_path)]
        )
        assert result.exit_code == 1
        assert isinstance(result.exception, MigrationRequiredError)

        async def _table_names() -> list[str]:
            db = await lancedb.connect_async(temp_db_path.absolute())
            tables = (await db.list_tables()).tables
            db.close()
            return tables

        assert "document_meta" not in asyncio.run(_table_names())

    def test_tag_commands_missing_database_exit_nonzero(self, tmp_path):
        missing = str(tmp_path / "does_not_exist.lancedb")
        for args in (
            ["tag", "create", "r1", "--db", missing],
            ["tag", "delete", "r1", "--db", missing],
            ["tag", "list", "--db", missing],
        ):
            result = runner.invoke(cli, args)
            assert result.exit_code == 1, args
            assert "does not exist" in result.output, args

    def test_tag_create_invalid_name_fails_cleanly(self, temp_db_path):
        """lance restricts ref names to alphanumeric, '.', '-', '_'; the CLI
        surfaces that as a clean error instead of a traceback."""
        db = str(temp_db_path)
        result = runner.invoke(cli, ["init", "--db", db])
        assert result.exit_code == 0

        result = runner.invoke(cli, ["tag", "create", "[red]release[/red]", "--db", db])
        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "Ref characters" in result.output


class TestTagRestore:
    def test_restore_requires_confirmation_and_decline_changes_nothing(
        self, temp_db_path
    ):
        db = str(temp_db_path)
        assert runner.invoke(cli, ["init", "--db", db]).exit_code == 0
        assert runner.invoke(cli, ["tag", "create", "r1", "--db", db]).exit_code == 0

        result = runner.invoke(cli, ["tag", "restore", "r1", "--db", db], input="n\n")
        assert result.exit_code == 1
        assert "live database state" in result.output
        assert "Stop all ingestion" in result.output
        assert "not transactionally atomic" in result.output
        assert "safety tag" in result.output

        result = runner.invoke(cli, ["tag", "list", "--db", db])
        assert "before-restore" not in result.output

    def test_restore_non_interactive_without_yes_fails(self, temp_db_path):
        db = str(temp_db_path)
        assert runner.invoke(cli, ["init", "--db", db]).exit_code == 0
        assert runner.invoke(cli, ["tag", "create", "r1", "--db", db]).exit_code == 0

        result = runner.invoke(cli, ["tag", "restore", "r1", "--db", db])
        assert result.exit_code == 1

        result = runner.invoke(cli, ["tag", "list", "--db", db])
        assert "before-restore" not in result.output

    def test_restore_with_yes(self, temp_db_path):
        db = str(temp_db_path)
        assert runner.invoke(cli, ["init", "--db", db]).exit_code == 0
        assert runner.invoke(cli, ["tag", "create", "r1", "--db", db]).exit_code == 0

        result = runner.invoke(cli, ["tag", "restore", "r1", "--yes", "--db", db])
        assert result.exit_code == 0
        assert "Restored database to tag 'r1'" in result.output
        assert "before-restore-" in result.output
        assert "now live" in result.output
        assert "migrate" in result.output

        result = runner.invoke(cli, ["tag", "list", "--db", db])
        assert "before-restore-" in result.output

    def test_restore_missing_tag_errors(self, temp_db_path):
        db = str(temp_db_path)
        assert runner.invoke(cli, ["init", "--db", db]).exit_code == 0

        result = runner.invoke(cli, ["tag", "restore", "nope", "--yes", "--db", db])
        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_restore_partial_tag_errors(self, temp_db_path):
        import asyncio

        from haiku.rag.store.engine import Store

        async def _partial_tag():
            async with Store(temp_db_path, create=True) as store:
                version = await store.chunks_table.version()
                await store.chunks_table.tags.create("stale", version)

        asyncio.run(_partial_tag())

        result = runner.invoke(
            cli, ["tag", "restore", "stale", "--yes", "--db", str(temp_db_path)]
        )
        assert result.exit_code == 1
        assert "partial" in result.output
        assert "documents" in result.output

    def test_restore_missing_database_exits_nonzero(self, tmp_path):
        missing = tmp_path / "does_not_exist.lancedb"
        result = runner.invoke(
            cli, ["tag", "restore", "r1", "--yes", "--db", str(missing)]
        )
        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_tag_help_includes_restore(self):
        result = runner.invoke(cli, ["tag", "--help"])
        assert result.exit_code == 0
        assert "restore" in result.output

        result = runner.invoke(cli, ["--help"])
        assert "--before" not in result.output
        assert "--at" not in result.output
