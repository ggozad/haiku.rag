import subprocess
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from evaluations.arm import load_arm, same_commit
from evaluations.preflight import Check, run_preflight
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _sha(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A committed git checkout carrying a .env with a telemetry token."""
    repo = tmp_path / "wt"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "code.py").write_text("x = 1\n")
    _git(repo, "add", "code.py")
    _git(
        repo,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "-m",
        "init",
    )
    (repo / ".env").write_text("LOGFIRE_TOKEN=pylf_v1_eu_test\n")
    return repo


@pytest.fixture
def arm_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "arms"
    directory.mkdir()
    (directory / "frames.yaml").write_text("search:\n  limit: 5\n")
    return directory


def _fields(checkout: Path, **overrides) -> dict:
    fields = {
        "name": "frames-main",
        "dataset": "frames",
        "worktree": str(checkout),
        "sha": _sha(checkout)[:12],
        "config": "frames.yaml",
        "db": "frames.lancedb",
        "limit": 10,
        "flags": ["--skip-db", "--skip-retrieval"],
    }
    fields.update(overrides)
    return fields


def _write_arm(path: Path, fields: dict) -> Path:
    path.write_text(yaml.safe_dump(fields))
    return path


async def _database(path: Path, config: AppConfig) -> None:
    async with HaikuRAG(path, config=config, create=True):
        pass


def _by_name(checks: list[Check]) -> dict[str, Check]:
    return {check.name: check for check in checks}


class TestArmSpec:
    def test_a_missing_sha_matches_no_commit(self) -> None:
        assert same_commit("abc123def456", "abc123def4567890")
        assert not same_commit("", "abc123def456")
        assert not same_commit("abc123def456", "")

    def test_flags_may_not_repeat_a_pinned_option(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        for flag in (
            ["--config", "other.yaml"],
            ["--name", "other"],
            ["--db", "other.lancedb"],
            ["--limit", "5"],
            ["--limit=5"],
            ["--filter-ids", "other.txt"],
        ):
            with pytest.raises(ValidationError, match="flags"):
                load_arm(_write_arm(arm_dir / "a.yaml", _fields(checkout, flags=flag)))

    def test_flags_that_set_nothing_the_arm_pins_are_kept(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        flags = ["--skip-db", "--no-telemetry", "--capability-model", "ollama:gpt-oss"]
        arm = load_arm(_write_arm(arm_dir / "a.yaml", _fields(checkout, flags=flags)))
        assert arm.command()[-len(flags) :] == flags

    def test_relative_paths_resolve_against_the_arm_file(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        arm = load_arm(
            _write_arm(arm_dir / "a.yaml", _fields(checkout, filter_ids="ids.txt"))
        )
        assert arm.worktree == checkout
        assert arm.config == arm_dir / "frames.yaml"
        assert arm.db == arm_dir / "frames.lancedb"
        assert arm.filter_ids == arm_dir / "ids.txt"

    def test_home_expands(self, checkout: Path, arm_dir: Path) -> None:
        arm = load_arm(
            _write_arm(arm_dir / "a.yaml", _fields(checkout, worktree="~/somewhere"))
        )
        assert arm.worktree == Path.home() / "somewhere"

    def test_unknown_keys_are_rejected(self, checkout: Path, arm_dir: Path) -> None:
        with pytest.raises(ValidationError):
            load_arm(
                _write_arm(
                    arm_dir / "a.yaml", _fields(checkout, target="analysis-capability")
                )
            )

    def test_a_comparator_needs_a_decision_rule(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        with pytest.raises(ValidationError, match="decision_rule"):
            load_arm(
                _write_arm(arm_dir / "a.yaml", _fields(checkout, comparator="b.yaml"))
            )

    def test_differences_need_a_comparator(self, checkout: Path, arm_dir: Path) -> None:
        with pytest.raises(ValidationError, match="comparator"):
            load_arm(
                _write_arm(arm_dir / "a.yaml", _fields(checkout, differences=["sha"]))
            )

    def test_command_is_the_run_invocation(self, checkout: Path, arm_dir: Path) -> None:
        arm = load_arm(
            _write_arm(
                arm_dir / "a.yaml",
                _fields(checkout, filter_ids="ids.txt", flags=["--skip-db"]),
            )
        )
        assert arm.command() == [
            "evaluations",
            "run",
            "frames",
            "--config",
            str(arm_dir / "frames.yaml"),
            "--name",
            "frames-main",
            "--db",
            str(arm_dir / "frames.lancedb"),
            "--limit",
            "10",
            "--filter-ids",
            str(arm_dir / "ids.txt"),
            "--skip-db",
        ]


class TestPreflight:
    async def test_a_sound_arm_passes_every_check(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        await _database(
            arm_dir / "frames.lancedb",
            AppConfig.model_validate({"search": {"limit": 5}}),
        )
        (arm_dir / "ids.txt").write_text("1\n2\n")
        arm = _write_arm(arm_dir / "a.yaml", _fields(checkout, filter_ids="ids.txt"))

        checks = _by_name((await run_preflight(arm)).checks)

        failed = [check for check in checks.values() if not check.ok]
        assert failed == []
        assert set(checks) == {
            "arm file",
            "dataset",
            "worktree",
            "telemetry",
            "config",
            "database",
            "filter ids",
        }
        assert checks["worktree"].detail.startswith(_sha(checkout))
        assert "2 ids" in checks["filter ids"].detail

    async def test_an_arm_file_that_does_not_validate_is_the_only_check(
        self, arm_dir: Path
    ) -> None:
        arm = arm_dir / "a.yaml"
        arm.write_text("name: x\n")
        checks = (await run_preflight(arm)).checks
        assert [check.name for check in checks] == ["arm file"]
        assert checks[0].ok is False

    async def test_an_unknown_dataset_fails(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        arm = _write_arm(arm_dir / "a.yaml", _fields(checkout, dataset="nope"))
        checks = _by_name((await run_preflight(arm)).checks)
        assert checks["dataset"].ok is False
        assert "nope" in checks["dataset"].detail

    async def test_a_sha_mismatch_fails_the_worktree_check(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        arm = _write_arm(arm_dir / "a.yaml", _fields(checkout, sha="0" * 12))
        checks = _by_name((await run_preflight(arm)).checks)
        assert checks["worktree"].ok is False
        assert _sha(checkout)[:12] in checks["worktree"].detail

    async def test_uncommitted_changes_fail_the_worktree_check(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        (checkout / "code.py").write_text("x = 2\n")
        arm = _write_arm(arm_dir / "a.yaml", _fields(checkout))
        checks = _by_name((await run_preflight(arm)).checks)
        assert checks["worktree"].ok is False
        assert "uncommitted" in checks["worktree"].detail

    async def test_a_missing_worktree_fails(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        arm = _write_arm(
            arm_dir / "a.yaml", _fields(checkout, worktree=str(arm_dir / "gone"))
        )
        checks = _by_name((await run_preflight(arm)).checks)
        assert checks["worktree"].ok is False

    async def test_a_missing_or_empty_token_fails_telemetry(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        arm = _write_arm(arm_dir / "a.yaml", _fields(checkout))
        (checkout / ".env").write_text("LOGFIRE_TOKEN=\n")
        assert _by_name((await run_preflight(arm)).checks)["telemetry"].ok is False
        (checkout / ".env").unlink()
        assert _by_name((await run_preflight(arm)).checks)["telemetry"].ok is False

    async def test_a_config_that_does_not_validate_fails(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        (arm_dir / "frames.yaml").write_text("search:\n  limit: many\n")
        arm = _write_arm(arm_dir / "a.yaml", _fields(checkout))
        checks = _by_name((await run_preflight(arm)).checks)
        assert checks["config"].ok is False
        assert checks["database"].ok is False

    async def test_a_missing_config_fails(self, checkout: Path, arm_dir: Path) -> None:
        arm = _write_arm(arm_dir / "a.yaml", _fields(checkout, config="nope.yaml"))
        assert _by_name((await run_preflight(arm)).checks)["config"].ok is False

    async def test_a_missing_database_fails(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        arm = _write_arm(arm_dir / "a.yaml", _fields(checkout))
        checks = _by_name((await run_preflight(arm)).checks)
        assert checks["database"].ok is False
        assert str(arm_dir / "frames.lancedb") in checks["database"].detail

    async def test_the_database_embedder_must_match_the_config(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        await _database(arm_dir / "frames.lancedb", AppConfig())
        (arm_dir / "frames.yaml").write_text(
            "embeddings:\n  model:\n    vector_dim: 8\n"
        )
        arm = _write_arm(arm_dir / "a.yaml", _fields(checkout))
        checks = _by_name((await run_preflight(arm)).checks)
        assert checks["database"].ok is False
        assert "8" in checks["database"].detail
        assert str(AppConfig().embeddings.model.vector_dim) in checks["database"].detail

    async def test_a_configured_set_needs_no_database_path(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        for name in ("a", "b"):
            await _database(arm_dir / f"{name}.lancedb", AppConfig())
        (arm_dir / "frames.yaml").write_text(
            f"lancedb:\n  databases:\n"
            f"    a: {arm_dir / 'a.lancedb'}\n    b: {arm_dir / 'b.lancedb'}\n"
        )
        arm = _write_arm(arm_dir / "a.yaml", _fields(checkout, db=None))
        checks = _by_name((await run_preflight(arm)).checks)
        assert checks["database"].ok is True, checks["database"].detail
        assert "a" in checks["database"].detail and "b" in checks["database"].detail

    async def test_every_database_of_a_configured_set_is_opened(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        await _database(arm_dir / "a.lancedb", AppConfig())
        (arm_dir / "frames.yaml").write_text(
            f"lancedb:\n  databases:\n"
            f"    a: {arm_dir / 'a.lancedb'}\n    b: {arm_dir / 'gone.lancedb'}\n"
        )
        arm = _write_arm(arm_dir / "a.yaml", _fields(checkout, db=None))
        checks = _by_name((await run_preflight(arm)).checks)
        assert checks["database"].ok is False
        assert "gone.lancedb" in checks["database"].detail

    async def test_a_configured_database_must_store_the_config_embedder(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        await _database(arm_dir / "a.lancedb", AppConfig())
        (arm_dir / "frames.yaml").write_text(
            f"embeddings:\n  model:\n    vector_dim: 8\n"
            f"lancedb:\n  databases:\n    a: {arm_dir / 'a.lancedb'}\n"
        )
        arm = _write_arm(arm_dir / "a.yaml", _fields(checkout, db=None))
        checks = _by_name((await run_preflight(arm)).checks)
        assert checks["database"].ok is False
        assert "8" in checks["database"].detail

    async def test_a_remote_database_is_named_and_not_opened(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        await _database(arm_dir / "a.lancedb", AppConfig())
        (arm_dir / "frames.yaml").write_text(
            f"lancedb:\n  databases:\n"
            f"    a: {arm_dir / 'a.lancedb'}\n    b: s3://bucket/b.lancedb\n"
        )
        arm = _write_arm(arm_dir / "a.yaml", _fields(checkout, db=None))
        check = _by_name((await run_preflight(arm)).checks)["database"]
        assert check.ok is True, check.detail
        assert "b" in check.detail

    async def test_a_relative_location_resolves_against_the_worktree(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        await _database(checkout / "rel.lancedb", AppConfig())
        (arm_dir / "frames.yaml").write_text(
            "lancedb:\n  databases:\n    a: rel.lancedb\n"
        )
        arm = _write_arm(arm_dir / "a.yaml", _fields(checkout, db=None))
        check = _by_name((await run_preflight(arm)).checks)["database"]
        assert check.ok is True, check.detail

    async def test_an_offline_arm_that_writes_no_result_file_fails(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        skipping_qa = _write_arm(
            arm_dir / "a.yaml",
            _fields(checkout, flags=["--skip-db", "--skip-qa", "--no-telemetry"]),
        )
        check = _by_name((await run_preflight(skipping_qa)).checks)["telemetry"]
        assert check.ok is False
        assert "--skip-qa" in check.detail

        live = _write_arm(
            arm_dir / "b.yaml",
            _fields(
                checkout,
                dataset="mtrag_clapnq_live",
                flags=["--skip-db", "--no-telemetry"],
            ),
        )
        check = _by_name((await run_preflight(live)).checks)["telemetry"]
        assert check.ok is False
        assert "mtrag_clapnq_live" in check.detail

    async def test_a_configured_set_needs_skip_db(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        (arm_dir / "frames.yaml").write_text(
            "lancedb:\n  databases:\n    a: /a.lancedb\n"
        )
        arm = _write_arm(
            arm_dir / "a.yaml",
            _fields(checkout, db=None, flags=["--skip-retrieval"]),
        )
        checks = _by_name((await run_preflight(arm)).checks)
        assert checks["database"].ok is False
        assert "--skip-db" in checks["database"].detail

    async def test_a_configured_set_refuses_a_database_path(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        (arm_dir / "frames.yaml").write_text(
            "lancedb:\n  databases:\n    a: /a.lancedb\n"
        )
        await _database(arm_dir / "frames.lancedb", AppConfig())
        arm = _write_arm(arm_dir / "a.yaml", _fields(checkout))
        checks = _by_name((await run_preflight(arm)).checks)
        assert checks["database"].ok is False
        assert "lancedb.databases" in checks["database"].detail

    async def test_an_arm_that_runs_without_telemetry_needs_no_token(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        (checkout / ".env").unlink()
        arm = _write_arm(
            arm_dir / "a.yaml",
            _fields(checkout, flags=["--skip-db", "--no-telemetry"]),
        )
        check = _by_name((await run_preflight(arm)).checks)["telemetry"]
        assert check.ok is True
        assert "--no-telemetry" in check.detail

    async def test_a_missing_filter_file_fails(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        arm = _write_arm(arm_dir / "a.yaml", _fields(checkout, smoke_ids="nope.txt"))
        checks = _by_name((await run_preflight(arm)).checks)
        assert checks["filter ids"].ok is False
        assert "nope.txt" in checks["filter ids"].detail

    async def test_an_empty_filter_file_fails(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        (arm_dir / "ids.txt").write_text("\n")
        arm = _write_arm(arm_dir / "a.yaml", _fields(checkout, filter_ids="ids.txt"))
        assert _by_name((await run_preflight(arm)).checks)["filter ids"].ok is False


class TestPreflightComparator:
    def _pair(self, checkout: Path, arm_dir: Path, **treated: object) -> Path:
        (arm_dir / "main.yaml").write_text(
            "search:\n  limit: 5\nqa:\n  max_searches: 3\n"
        )
        _write_arm(
            arm_dir / "main.arm.yaml",
            _fields(
                checkout,
                name="frames-main",
                config="main.yaml",
                flags=["--skip-db", "--target", "analysis-capability"],
            ),
        )
        fields = _fields(
            checkout,
            name="frames-branch",
            comparator="main.arm.yaml",
            decision_rule="McNemar exact, two-sided, fail below p 0.05",
            flags=["--skip-db"],
        )
        fields.update(treated)
        return _write_arm(arm_dir / "branch.arm.yaml", fields)

    async def test_unnamed_differences_stop_the_launch(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        arm = self._pair(checkout, arm_dir)
        check = _by_name((await run_preflight(arm)).checks)["comparator"]
        assert check.ok is False
        assert "qa.max_searches" in check.detail
        assert "--target" in check.detail

    async def test_named_differences_pass(self, checkout: Path, arm_dir: Path) -> None:
        arm = self._pair(checkout, arm_dir, differences=["qa.max_searches", "--target"])
        check = _by_name((await run_preflight(arm)).checks)["comparator"]
        assert check.ok is True
        assert "qa.max_searches" in check.detail

    async def test_a_named_difference_that_does_not_differ_fails(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        arm = self._pair(
            checkout,
            arm_dir,
            differences=["qa.max_searches", "--target", "qa.model.name"],
        )
        check = _by_name((await run_preflight(arm)).checks)["comparator"]
        assert check.ok is False
        assert "qa.model.name" in check.detail

    async def test_a_different_sha_must_be_named(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        arm = self._pair(
            checkout,
            arm_dir,
            sha="a" * 12,
            differences=["qa.max_searches", "--target"],
        )
        check = _by_name((await run_preflight(arm)).checks)["comparator"]
        assert check.ok is False
        assert "sha" in check.detail

    async def test_a_flag_value_difference_is_named_by_its_option(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        arm = self._pair(
            checkout,
            arm_dir,
            flags=["--skip-db", "--target", "rag-capability"],
            differences=["qa.max_searches"],
        )
        check = _by_name((await run_preflight(arm)).checks)["comparator"]
        assert check.ok is False
        assert "--target" in check.detail

    async def test_an_attached_flag_value_is_named_by_its_option(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        arm = self._pair(
            checkout,
            arm_dir,
            flags=["--skip-db", "--target=rag-capability"],
            differences=["qa.max_searches", "--target"],
        )
        check = _by_name((await run_preflight(arm)).checks)["comparator"]
        assert check.ok is True, check.detail

    async def test_a_missing_comparator_file_fails(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        arm = _write_arm(
            arm_dir / "a.yaml",
            _fields(checkout, comparator="gone.yaml", decision_rule="sign test"),
        )
        check = _by_name((await run_preflight(arm)).checks)["comparator"]
        assert check.ok is False
        assert "gone.yaml" in check.detail


class TestPreflightCommand:
    def test_exit_status_and_output_follow_the_checks(
        self, checkout: Path, arm_dir: Path
    ) -> None:
        import asyncio

        from typer.testing import CliRunner

        from evaluations.benchmark import app

        arm = _write_arm(arm_dir / "a.yaml", _fields(checkout))
        failing = CliRunner().invoke(app, ["preflight", str(arm)])
        assert failing.exit_code == 1
        assert "FAIL database" in failing.output
        assert _sha(checkout) in failing.output

        asyncio.run(
            _database(
                arm_dir / "frames.lancedb",
                AppConfig.model_validate({"search": {"limit": 5}}),
            )
        )
        passing = CliRunner().invoke(app, ["preflight", str(arm)])
        assert passing.exit_code == 0, passing.output
        assert "FAIL" not in passing.output
        assert "ok   database" in passing.output
