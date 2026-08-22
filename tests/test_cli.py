import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.exceptions import BadParameter
from typer.testing import CliRunner

from haiku.rag.cli import _cli as cli
from haiku.rag.cli import (
    _parse_meta_options,
    require_one_database,
    resolve_db_path,
    select_database,
)
from haiku.rag.cli import cli as cli_wrapper
from haiku.rag.config import get_config, set_config
from haiku.rag.config.models import AppConfig, LanceDBConfig
from haiku.rag.store.exceptions import (
    AmbiguousDatabaseError,
    MigrationRequiredError,
)

runner = CliRunner()


def test_importing_cli_does_not_load_heavy_dependencies():
    """Importing the CLI must not pull the heavy runtime dependencies; they cost
    seconds of startup. Runs in a subprocess because another test in the same
    session may already have imported them."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import haiku.rag.cli, sys; "
            "loaded = {'lancedb', 'pyarrow', 'pydantic_ai'} & sys.modules.keys(); "
            "assert not loaded, loaded",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


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


class TestOneDatabaseCommands:
    """`lancedb.databases` names a set; most commands work on one database."""

    @staticmethod
    def _config(**databases):
        return AppConfig(lancedb=LanceDBConfig(databases=databases))

    def test_a_configured_set_refuses_a_one_database_command(self):
        with pytest.raises(AmbiguousDatabaseError, match="alpha, beta"):
            require_one_database(
                self._config(alpha="/db/a.lancedb", beta="/db/b.lancedb"),
                None,
                federated=False,
            )

    def test_the_refusal_names_the_databases_and_not_their_locations(self):
        """A location in an error message travels into logs and terminals; the
        names exist so it does not have to."""
        with pytest.raises(AmbiguousDatabaseError) as raised:
            require_one_database(
                self._config(alpha="s3://bucket/prefix/a.lancedb"),
                None,
                federated=False,
            )

        assert "alpha" in str(raised.value)
        assert "s3://bucket/prefix/a.lancedb" not in str(raised.value)
        assert "bucket" not in str(raised.value)

    def test_a_command_covering_the_set_is_allowed(self):
        require_one_database(
            self._config(alpha="/db/a.lancedb", beta="/db/b.lancedb"),
            None,
            federated=True,
        )

    def test_naming_a_path_is_allowed(self):
        require_one_database(
            self._config(alpha="/db/a.lancedb"),
            Path("/db/other.lancedb"),
            federated=False,
        )

    def test_no_configured_databases_is_allowed(self):
        require_one_database(AppConfig(), None, federated=False)

    def test_the_refusal_exits_with_an_error(self):
        with patch("haiku.rag.cli._cli") as mock_cli:
            mock_cli.side_effect = AmbiguousDatabaseError("names alpha, beta")

            with pytest.raises(SystemExit) as exc_info:
                cli_wrapper()
            assert exc_info.value.code == 1


class TestSelectingADatabaseByName:
    """`--database NAME` is the only way to reach a configured database whose
    location is a URI, since `--db` takes a path."""

    @staticmethod
    def _install(monkeypatch, **databases):
        import haiku.rag.config as config_module

        monkeypatch.setattr(config_module, "_config", None)
        set_config(AppConfig(lancedb=LanceDBConfig(databases=databases)))

    def test_a_uri_location_becomes_the_configured_uri(self, monkeypatch):
        self._install(monkeypatch, medic="s3://bucket/prefix/medic.lancedb")

        db_path = select_database("medic")

        assert db_path is None
        config = get_config()
        assert config.lancedb.uri == "s3://bucket/prefix/medic.lancedb"
        assert config.lancedb.databases == {}

    def test_a_local_location_becomes_the_database_path(self, monkeypatch):
        self._install(monkeypatch, st="/data/st.lancedb")

        db_path = select_database("st")

        assert db_path == Path("/data/st.lancedb")
        assert get_config().lancedb.uri == ""

    def test_an_unknown_name_names_the_configured_ones(self, monkeypatch):
        self._install(monkeypatch, alpha="/data/a.lancedb", beta="/data/b.lancedb")

        with pytest.raises(AmbiguousDatabaseError, match="alpha, beta"):
            select_database("gamma")

    def test_an_unknown_name_does_not_leak_locations(self, monkeypatch):
        self._install(monkeypatch, medic="s3://bucket/prefix/medic.lancedb")

        with pytest.raises(AmbiguousDatabaseError) as raised:
            select_database("gamma")

        assert "bucket" not in str(raised.value)

    def test_selecting_nothing_reports_an_empty_mapping(self, monkeypatch):
        import haiku.rag.config as config_module

        monkeypatch.setattr(config_module, "_config", None)
        set_config(AppConfig())

        with pytest.raises(AmbiguousDatabaseError, match="nothing"):
            select_database("medic")

    def test_the_callback_selects_before_a_command_runs(self, tmp_path, monkeypatch):
        """`--database` is resolved once the config is loaded, so every command
        and both TUIs see the selected database."""
        import haiku.rag.config as config_module

        monkeypatch.setattr(config_module, "_config", None)
        config_file = tmp_path / "haiku.rag.yaml"
        config_file.write_text("lancedb:\n  databases:\n    alpha: /data/a.lancedb\n")

        result = runner.invoke(
            cli, ["--config", str(config_file), "--database", "nope", "list"]
        )

        assert result.exit_code != 0
        assert isinstance(result.exception, AmbiguousDatabaseError)
        assert "nope" in str(result.exception)

    def test_a_selection_does_not_outlive_its_invocation(self, tmp_path, monkeypatch):
        """The selector is module state, so a second invocation without
        `--database` must not inherit the first one's database."""
        import haiku.rag.cli as cli_module
        import haiku.rag.config as config_module

        monkeypatch.setattr(config_module, "_config", None)
        monkeypatch.setattr(cli_module, "_database", None)
        monkeypatch.setattr(cli_module, "_database_path", None)
        config_file = tmp_path / "haiku.rag.yaml"
        selected = tmp_path / "alpha.lancedb"
        config_file.write_text(f"lancedb:\n  databases:\n    alpha: {selected}\n")

        runner.invoke(
            cli, ["--config", str(config_file), "--database", "alpha", "info"]
        )
        assert cli_module._database_path == selected

        runner.invoke(cli, ["--config", str(config_file), "settings"])

        assert cli_module._database_path is None
        assert cli_module._database is None

    def test_a_selection_does_not_outlive_its_invocation_in_process(
        self, tmp_path, monkeypatch
    ):
        """Selecting rewrites the configuration, so a second invocation has to
        start from a freshly loaded one rather than inherit the rewrite."""
        import haiku.rag.cli as cli_module
        import haiku.rag.config as config_module

        monkeypatch.setattr(config_module, "_config", None)
        monkeypatch.setattr(cli_module, "_database", None)
        monkeypatch.setattr(cli_module, "_database_path", None)
        config_file = tmp_path / "haiku.rag.yaml"
        config_file.write_text(
            f"lancedb:\n  databases:\n    alpha: {tmp_path / 'alpha.lancedb'}\n"
            f"    beta: {tmp_path / 'beta.lancedb'}\n"
        )

        runner.invoke(
            cli, ["--config", str(config_file), "--database", "alpha", "settings"]
        )
        assert get_config().lancedb.uri == ""
        assert get_config().lancedb.databases == {}

        runner.invoke(cli, ["--config", str(config_file), "settings"])

        assert set(get_config().lancedb.databases) == {"alpha", "beta"}

    def test_a_selection_does_not_outlive_an_invocation_without_a_config_file(
        self, tmp_path, monkeypatch
    ):
        """No config file is still a load: the previous invocation's selected URI
        must not be what the next one talks to."""
        import haiku.rag.cli as cli_module
        import haiku.rag.config as config_module

        monkeypatch.setattr(config_module, "_config", None)
        monkeypatch.setattr(cli_module, "_database", None)
        monkeypatch.setattr(cli_module, "_database_path", None)
        monkeypatch.delenv("HAIKU_RAG_CONFIG_PATH", raising=False)
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / "selected.yaml"
        config_file.write_text(
            "lancedb:\n  databases:\n    medic: s3://bucket/medic.lancedb\n"
        )

        runner.invoke(
            cli, ["--config", str(config_file), "--database", "medic", "settings"]
        )
        assert get_config().lancedb.uri == "s3://bucket/medic.lancedb"

        runner.invoke(cli, ["settings"])

        assert get_config().lancedb.uri == ""


class TestResolvingTheDatabasePath:
    def test_a_path_wins_when_nothing_is_selected(self, monkeypatch):
        monkeypatch.setattr("haiku.rag.cli._database", None)
        monkeypatch.setattr("haiku.rag.cli._database_path", None)

        assert resolve_db_path(Path("/data/one.lancedb")) == Path("/data/one.lancedb")

    def test_the_selected_database_is_used_without_a_path(self, monkeypatch):
        monkeypatch.setattr("haiku.rag.cli._database", "st")
        monkeypatch.setattr("haiku.rag.cli._database_path", Path("/data/st.lancedb"))

        assert resolve_db_path(None) == Path("/data/st.lancedb")

    def test_naming_a_database_twice_is_refused(self, monkeypatch):
        monkeypatch.setattr("haiku.rag.cli._database", "st")
        monkeypatch.setattr("haiku.rag.cli._database_path", Path("/data/st.lancedb"))

        with pytest.raises(AmbiguousDatabaseError, match="not both"):
            resolve_db_path(Path("/data/other.lancedb"))


class TestCliConfigMismatchError:
    def test_a_config_mismatch_exits_with_its_remedy(self):
        """The message says which database and what to run, so it is worth more
        than a traceback."""
        from haiku.rag.store.exceptions import ConfigMismatchError

        with patch("haiku.rag.cli._cli") as mock_cli:
            mock_cli.side_effect = ConfigMismatchError(
                "database 'nemotron': vector dimension 2048 -> 2560"
            )

            with pytest.raises(SystemExit) as exc_info:
                cli_wrapper()
            assert exc_info.value.code == 1


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

        # Without --yes the missing database is reported before the
        # confirmation prompt, not after the user confirms.
        result = runner.invoke(cli, ["tag", "restore", "r1", "--db", str(missing)])
        assert result.exit_code == 1
        assert "does not exist" in result.output
        assert "Continue?" not in result.output

    def test_tag_help_includes_restore(self):
        result = runner.invoke(cli, ["tag", "--help"])
        assert result.exit_code == 0
        assert "restore" in result.output

        result = runner.invoke(cli, ["--help"])
        assert "--before" not in result.output
        assert "--at" not in result.output


class TestAskAnalyzeImageOption:
    def test_ask_forwards_image_paths(self):
        from unittest.mock import AsyncMock

        from haiku.rag.app import HaikuRAGApp

        with patch.object(HaikuRAGApp, "ask", new_callable=AsyncMock) as mock_ask:
            result = runner.invoke(
                cli,
                ["ask", "q", "--image", "/tmp/a.png", "--image", "/tmp/b.jpg"],
            )
        assert result.exit_code == 0
        from pathlib import Path

        assert mock_ask.call_args.kwargs["images"] == [
            Path("/tmp/a.png"),
            Path("/tmp/b.jpg"),
        ]

    def test_analyze_forwards_image_paths(self):
        from unittest.mock import AsyncMock

        from haiku.rag.app import HaikuRAGApp

        with patch.object(HaikuRAGApp, "analyze", new_callable=AsyncMock) as mock:
            result = runner.invoke(cli, ["analyze", "q", "--image", "/tmp/a.png"])
        assert result.exit_code == 0
        from pathlib import Path

        assert mock.call_args.kwargs["images"] == [Path("/tmp/a.png")]

    @pytest.mark.asyncio
    async def test_app_ask_reads_image_bytes(self, temp_db_path, tmp_path):
        from io import BytesIO
        from unittest.mock import AsyncMock

        from PIL import Image as PILImage

        from haiku.rag.app import HaikuRAGApp
        from haiku.rag.client import HaikuRAG

        buffer = BytesIO()
        PILImage.new("RGB", (4, 4)).save(buffer, format="PNG")
        img_path = tmp_path / "img.png"
        img_path.write_bytes(buffer.getvalue())

        async with HaikuRAG(temp_db_path, create=True):
            pass

        with patch.object(
            HaikuRAG, "ask", new_callable=AsyncMock, return_value=("answer", [])
        ) as mock_ask:
            app = HaikuRAGApp(db_path=temp_db_path)
            await app.ask("q", images=[img_path])

        assert mock_ask.call_args.kwargs["images"] == [buffer.getvalue()]


class TestChatCoversTheSet:
    """Chat is a read verb: it answers with the same capabilities `ask` uses, so
    it covers the configured set rather than demanding one database."""

    @staticmethod
    def _config_file(tmp_path):
        config_file = tmp_path / "haiku.rag.yaml"
        config_file.write_text(
            f"lancedb:\n  databases:\n    arxiv: {tmp_path / 'a.lancedb'}\n"
            f"    wiki: {tmp_path / 'w.lancedb'}\n"
        )
        return config_file

    def test_a_configured_set_is_covered_rather_than_refused(
        self, tmp_path, monkeypatch
    ):
        import haiku.rag.cli as cli_module
        import haiku.rag.config as config_module

        monkeypatch.setattr(config_module, "_config", None)
        monkeypatch.setattr(cli_module, "_database", None)
        monkeypatch.setattr(cli_module, "_database_path", None)

        with patch("haiku.rag.chat.run_chat") as run_chat:
            result = runner.invoke(
                cli, ["--config", str(self._config_file(tmp_path)), "chat"]
            )

        assert result.exit_code == 0, result.output
        # None is what makes the client resolve the set for itself.
        assert run_chat.call_args.args[0] is None

    def test_naming_one_database_opens_that_one(self, tmp_path, monkeypatch):
        import haiku.rag.cli as cli_module
        import haiku.rag.config as config_module

        monkeypatch.setattr(config_module, "_config", None)
        monkeypatch.setattr(cli_module, "_database", None)
        monkeypatch.setattr(cli_module, "_database_path", None)

        with patch("haiku.rag.chat.run_chat") as run_chat:
            result = runner.invoke(
                cli,
                [
                    "--config",
                    str(self._config_file(tmp_path)),
                    "--database",
                    "wiki",
                    "chat",
                ],
            )

        assert result.exit_code == 0, result.output
        assert run_chat.call_args.args[0] == tmp_path / "w.lancedb"

    def test_a_single_database_setup_is_unchanged(self, tmp_path, monkeypatch):
        """Without a configured set, chat opens the path it always did."""
        import haiku.rag.cli as cli_module
        import haiku.rag.config as config_module

        monkeypatch.setattr(config_module, "_config", None)
        monkeypatch.setattr(cli_module, "_database", None)
        monkeypatch.setattr(cli_module, "_database_path", None)

        with patch("haiku.rag.chat.run_chat") as run_chat:
            result = runner.invoke(cli, ["chat", "--db", str(tmp_path / "one.lancedb")])

        assert result.exit_code == 0, result.output
        assert run_chat.call_args.args[0] == tmp_path / "one.lancedb"


class TestRenderingTheDatabase:
    """Across databases a result has to say which one it came from. One database
    needs no such label, so single-database output is unchanged."""

    @staticmethod
    def _app(tmp_path, **databases):
        from haiku.rag.app import HaikuRAGApp

        return HaikuRAGApp(
            db_path=tmp_path / "unused",
            config=AppConfig(lancedb=LanceDBConfig(databases=databases)),
        )

    @staticmethod
    def _rendered(app, result) -> str:
        from rich.console import Console

        app.console = Console(record=True, width=200)
        app._rich_print_search_result(result)
        return app.console.export_text()

    @staticmethod
    def _result():
        from haiku.rag.store.models import SearchResult

        return SearchResult(
            content="a body",
            score=0.9,
            source="alpha",
            chunk_id="c1",
            document_id="d1",
            document_uri="test://alpha/one",
        )

    def test_a_set_labels_each_result(self, tmp_path):
        app = self._app(
            tmp_path,
            alpha=str(tmp_path / "alpha.lancedb"),
            beta=str(tmp_path / "beta.lancedb"),
        )

        assert "database: alpha" in self._rendered(app, self._result())

    def test_one_database_is_not_labelled(self, tmp_path):
        app = self._app(tmp_path, alpha=str(tmp_path / "alpha.lancedb"))

        assert "database:" not in self._rendered(app, self._result())


@pytest.fixture
def app_stub(monkeypatch, tmp_path):
    """Stand in for HaikuRAGApp so a command's wiring can be checked without a
    database or a model. These tests pin argument parsing and dispatch, not what
    the application layer renders."""
    # AsyncMock so every command's `asyncio.run(app.x(...))` gets a coroutine.
    stub = AsyncMock()
    monkeypatch.setattr(
        "haiku.rag.cli.create_app",
        lambda db=None, *, federated=False: stub,
    )
    return stub


DB_ARGS = ["--db", "/tmp/test.lancedb"]


@pytest.mark.parametrize(
    "argv, method, expected",
    [
        (["list"], "list_documents", {"filter": None}),
        (
            ["list", "--filter", "uri LIKE 'x%'"],
            "list_documents",
            {"filter": "uri LIKE 'x%'"},
        ),
        (
            ["add", "some text", "--title", "T"],
            "add_document_from_text",
            {"text": "some text", "title": "T", "metadata": None},
        ),
        (
            ["add-src", "/tmp/doc.md"],
            "add_document_from_source",
            {"source": "/tmp/doc.md", "title": None, "metadata": None},
        ),
        (["get", "doc-1"], "get_document", {"doc_id": "doc-1"}),
        (["delete", "doc-1"], "delete_document", {"doc_id": "doc-1"}),
        (
            ["visualize", "chunk-1"],
            "visualize_chunk",
            {"chunk_id": "chunk-1", "expand": True},
        ),
        (
            ["visualize", "chunk-1", "--no-expand"],
            "visualize_chunk",
            {"chunk_id": "chunk-1", "expand": False},
        ),
        (["vacuum"], "vacuum", {}),
        (["create-index"], "create_index", {}),
        (["init"], "init", {}),
        (["info"], "info", {}),
        # limit/search_type default to None: the app layer resolves the config
        # default, so the CLI must not invent one.
        (["history"], "history", {"table": None, "limit": None}),
        (
            ["history", "--table", "chunks", "--limit", "5"],
            "history",
            {"table": "chunks", "limit": 5},
        ),
    ],
)
def test_command_dispatches_to_the_app(app_stub, argv, method, expected):
    result = runner.invoke(cli, argv + DB_ARGS)

    assert result.exit_code == 0, result.output
    getattr(app_stub, method).assert_called_once_with(**expected)


@pytest.mark.parametrize(
    "argv, expected",
    [
        (
            ["search", "q"],
            {
                "query": "q",
                "limit": None,
                "filter": None,
                "search_type": None,
                "image": None,
            },
        ),
        (
            ["search", "q", "--limit", "3", "--search-type", "vector"],
            {
                "query": "q",
                "limit": 3,
                "filter": None,
                "search_type": "vector",
                "image": None,
            },
        ),
    ],
)
def test_search_dispatch(app_stub, argv, expected):
    result = runner.invoke(cli, argv + DB_ARGS)

    assert result.exit_code == 0, result.output
    app_stub.search.assert_called_once_with(**expected)


@pytest.mark.parametrize("command, method", [("ask", "ask"), ("analyze", "analyze")])
def test_question_commands_dispatch(app_stub, command, method):
    result = runner.invoke(cli, [command, "why?"] + DB_ARGS)

    assert result.exit_code == 0, result.output
    getattr(app_stub, method).assert_called_once_with(
        question="why?", filter=None, images=None
    )


@pytest.mark.parametrize(
    "flag, mode_name",
    [
        (None, "FULL"),
        ("--embed-only", "EMBED_ONLY"),
        ("--rechunk", "RECHUNK"),
        ("--title-only", "TITLE_ONLY"),
        ("--descriptions", "DESCRIPTIONS"),
        ("--set-embedder", "SET_EMBEDDER"),
    ],
)
def test_rebuild_flag_selects_the_mode(app_stub, flag, mode_name):
    """Each flag picks one rebuild mode, and no flag means a full rebuild."""
    result = runner.invoke(cli, ["rebuild"] + ([flag] if flag else []) + DB_ARGS)

    assert result.exit_code == 0, result.output
    (_, kwargs) = app_stub.rebuild.call_args
    assert kwargs["mode"].name == mode_name


def test_migrate_reports_applied_migrations(app_stub):
    app_stub.migrate.return_value = ["v0_40_0: add document_items"]

    result = runner.invoke(cli, ["migrate"] + DB_ARGS)

    assert result.exit_code == 0, result.output
    assert "Applied 1 migration(s)" in result.output
    assert "add document_items" in result.output


def test_migrate_reports_an_up_to_date_database(app_stub):
    app_stub.migrate.return_value = []

    result = runner.invoke(cli, ["migrate"] + DB_ARGS)

    assert result.exit_code == 0, result.output
    assert "No migrations pending" in result.output


def test_migrate_exits_nonzero_on_failure(app_stub):
    app_stub.migrate.side_effect = RuntimeError("schema is from the future")

    result = runner.invoke(cli, ["migrate"] + DB_ARGS)

    assert result.exit_code == 1
    assert "Migration failed: schema is from the future" in result.output


def test_mcp_stdio_selects_the_transport(app_stub):
    result = runner.invoke(cli, ["mcp", "--stdio"] + DB_ARGS)

    assert result.exit_code == 0, result.output
    app_stub.run_mcp.assert_called_once()
    kwargs = app_stub.run_mcp.call_args.kwargs
    assert kwargs["transport"] == "stdio"


def test_mcp_without_stdio_leaves_the_transport_unset(app_stub):
    result = runner.invoke(cli, ["mcp"] + DB_ARGS)

    assert result.exit_code == 0, result.output
    assert app_stub.run_mcp.call_args.kwargs["transport"] is None


def test_version_flag_prints_the_version():
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0, result.output
    assert "haiku.rag version" in result.output


def test_outdated_install_warns(app_stub, monkeypatch):
    """The startup check warns but does not block the command."""

    async def outdated():
        return False, "0.1.0", "9.9.9"

    monkeypatch.setattr("haiku.rag.cli.is_up_to_date", outdated)

    result = runner.invoke(cli, ["list"] + DB_ARGS)

    assert result.exit_code == 0, result.output
    assert "haiku.rag is outdated" in result.output
    assert "Current: 0.1.0, Latest: 9.9.9" in result.output
    app_stub.list_documents.assert_called_once()


def test_up_to_date_install_says_nothing(app_stub, monkeypatch):
    async def current():
        return True, "9.9.9", "9.9.9"

    monkeypatch.setattr("haiku.rag.cli.is_up_to_date", current)

    result = runner.invoke(cli, ["list"] + DB_ARGS)

    assert result.exit_code == 0, result.output
    assert "outdated" not in result.output


def test_a_failing_version_check_does_not_block_the_cli(app_stub, monkeypatch):
    """PyPI being unreachable must not stop a command from running."""

    async def boom():
        raise RuntimeError("no network")

    monkeypatch.setattr("haiku.rag.cli.is_up_to_date", boom)

    result = runner.invoke(cli, ["list"] + DB_ARGS)

    assert result.exit_code == 0, result.output
    app_stub.list_documents.assert_called_once()


def test_settings_command_shows_the_configuration(monkeypatch):
    shown = []

    class StubApp:
        def __init__(self, **kwargs):
            shown.append(kwargs)

        def show_settings(self):
            shown.append("shown")

    monkeypatch.setattr("haiku.rag.app.HaikuRAGApp", StubApp)

    result = runner.invoke(cli, ["settings"])

    assert result.exit_code == 0, result.output
    assert "shown" in shown
    assert shown[0]["read_only"] is True


def test_download_models_reports_failure(monkeypatch):
    class StubApp:
        def __init__(self, **kwargs):
            pass

        async def download_models(self):
            raise RuntimeError("hub unreachable")

    monkeypatch.setattr("haiku.rag.app.HaikuRAGApp", StubApp)

    result = runner.invoke(cli, ["download-models"])

    assert result.exit_code == 1
    assert "Error downloading models: hub unreachable" in result.output


def test_download_models_succeeds(monkeypatch):
    calls = []

    class StubApp:
        def __init__(self, **kwargs):
            pass

        async def download_models(self):
            calls.append("downloaded")

    monkeypatch.setattr("haiku.rag.app.HaikuRAGApp", StubApp)

    result = runner.invoke(cli, ["download-models"])

    assert result.exit_code == 0, result.output
    assert calls == ["downloaded"]


def test_chat_reports_a_missing_tui_extra(monkeypatch):
    """run_chat imports the Textual app itself, so a missing tui extra surfaces
    from the call, not from importing haiku.rag.chat. The CLI must report it and
    exit nonzero rather than traceback."""
    import sys

    monkeypatch.setitem(sys.modules, "haiku.rag.chat.app", None)

    result = runner.invoke(cli, ["chat"])

    assert result.exit_code == 1
    assert "textual is not installed" in result.output
    assert "haiku.rag-slim[tui]" in result.output


def test_inspect_reports_a_missing_tui_extra(monkeypatch):
    """run_inspector raises at import instead, so the guard sits on the import."""
    import sys

    monkeypatch.delitem(sys.modules, "haiku.rag.inspector", raising=False)
    monkeypatch.setitem(sys.modules, "haiku.rag.inspector.app", None)

    result = runner.invoke(cli, ["inspect"])

    assert result.exit_code == 1
    assert "textual is not installed" in result.output
    assert "haiku.rag-slim[tui]" in result.output
