import asyncio
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from evaluations.arm import load_arm
from evaluations.queue import Queue, provision_worktree, tmux_command
from evaluations.registry import ArmRecord, Registry
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig

# Stands in for `evaluations run`: prints its name, hangs or fails on demand.
FAKE_RUN = textwrap.dedent(
    """
    import json, os, pathlib, sys, time
    name = sys.argv[sys.argv.index("--name") + 1]
    print("fake run", name, flush=True)
    if name.endswith("-hang"):
        time.sleep(30)
    if name.endswith("-fail"):
        sys.exit(3)
    results = os.environ.get("HAIKU_RAG_EVAL_RESULTS")
    if results:
        trace = ("1" if name.endswith("-smoke") else "2") * 32
        directory = pathlib.Path(results)
        directory.mkdir(parents=True, exist_ok=True)
        rows = [
            {"case_name": f"{i}_{i}", "key": str(i), "passed": True, "cited": True,
             "cited_map": 0.5, "aborted": False, "trace_id": trace}
            for i in range(3)
        ]
        (directory / f"{name}.{trace[:12]}.jsonl").write_text(
            "".join(json.dumps(row) + "\\n" for row in rows)
        )
    if "-flaky" in name:
        sys.exit(3)
    """
)
PYPROJECT = (
    '[project]\nname = "probe"\nversion = "0.0.0"\n\n[tool.uv]\npackage = false\n'
)
SMOKE_TRACE = "1" * 32
RUN_TRACE = "2" * 32


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _commit_all(repo: Path) -> None:
    _git(repo, "add", "-A")
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


@pytest.fixture
def workspace(tmp_path: Path) -> SimpleNamespace:
    """A checkout with .env, a config, a database, a smoke file and a fake run."""
    repo = tmp_path / "wt"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "code.py").write_text("x = 1\n")
    (repo / "pyproject.toml").write_text(PYPROJECT)
    (repo / ".gitignore").write_text(".env\n.venv\n")
    _commit_all(repo)
    (repo / ".env").write_text("LOGFIRE_TOKEN=pylf_v1_eu_test\n")
    arms = tmp_path / "arms"
    arms.mkdir()
    (arms / "frames.yaml").write_text("search:\n  limit: 5\n")
    (arms / "smoke.txt").write_text("1\n2\n3\n")

    async def create() -> None:
        async with HaikuRAG(
            arms / "frames.lancedb",
            config=AppConfig.model_validate({"search": {"limit": 5}}),
            create=True,
        ):
            pass

    asyncio.run(create())
    fake = tmp_path / "fake_run.py"
    fake.write_text(FAKE_RUN)
    return SimpleNamespace(
        repo=repo,
        arms=arms,
        fake=fake,
        tmp=tmp_path,
        sha=_git(repo, "rev-parse", "HEAD")[:12],
    )


def _arm(ws: SimpleNamespace, name: str, **overrides: Any) -> Path:
    fields: dict[str, Any] = {
        "name": name,
        "dataset": "frames",
        "worktree": str(ws.repo),
        "sha": ws.sha,
        "config": "frames.yaml",
        "db": "frames.lancedb",
        "limit": 10,
        "flags": ["--skip-db", "--skip-retrieval"],
        "smoke_ids": "smoke.txt",
        "deadline_hours": 1,
    }
    fields.update(overrides)
    fields = {key: value for key, value in fields.items() if value is not None}
    path = ws.arms / f"{name}.yaml"
    path.write_text(yaml.safe_dump(fields))
    return path


def _cases(n: int) -> list[dict]:
    return [
        {
            "case_name": f"{i}_{i}",
            "pair_key": str(i),
            "answer_equivalent": "true" if i % 2 else "false",
            "number_match": None,
            "cited_map": 0.5,
            "cited_uris": '["u1"]',
            "is_exception": False,
        }
        for i in range(n)
    ]


def _fake_query(traces: dict[str, str], cases: dict[str, list[dict]]):
    """`traces` maps a run name to its trace, `cases` a trace to its case rows."""
    span = {
        "start_timestamp": "2026-01-01T10:00:00Z",
        "end_timestamp": "2026-01-01T11:00:00Z",
    }

    def query(sql: str, *, min_timestamp: str) -> list[dict]:
        if "attributes->>'name' = '" in sql:
            name = sql.split("attributes->>'name' = '")[1].split("'")[0]
            trace = traces.get(name)
            return [] if trace is None else [{"trace_id": trace, **span}]
        trace = sql.split("trace_id = '")[1].split("'")[0]
        if "evaluate {name}" in sql:
            return [{"trace_id": trace, **span}]
        rows = cases.get(trace, [])
        if "count(*)" in sql:
            return [{"n": len(rows)}]
        return rows

    return query


def _queue(
    ws: SimpleNamespace,
    traces: dict[str, str],
    cases: dict[str, list[dict]],
    **overrides: Any,
) -> Queue:
    options: dict[str, Any] = {
        "registry": Registry(ws.tmp / "registry.sqlite"),
        "logs": ws.tmp / "logs",
        "query": _fake_query(traces, cases),
        "prefix": [sys.executable, str(ws.fake)],
        "settle_seconds": 0.3,
        "poll_seconds": 0.05,
    }
    options.update(overrides)
    return Queue(**options)


def _both(name: str) -> dict[str, str]:
    return {f"{name}-smoke": SMOKE_TRACE, name: RUN_TRACE}


class TestSmokeCommand:
    def test_smoke_replaces_the_case_selection_and_suffixes_the_name(
        self, workspace: SimpleNamespace
    ) -> None:
        arm = load_arm(_arm(workspace, "arm-ok"))
        command = arm.command(smoke=True)
        assert command[:3] == ["evaluations", "run", "frames"]
        assert "--limit" not in command
        assert command[command.index("--name") + 1] == "arm-ok-smoke"
        assert command[command.index("--filter-ids") + 1] == str(
            workspace.arms / "smoke.txt"
        )
        assert command[-2:] == ["--skip-db", "--skip-retrieval"]


class TestQueue:
    def test_runs_the_smoke_then_the_arm_and_completes_the_row(
        self, workspace: SimpleNamespace
    ) -> None:
        arm = _arm(workspace, "arm-ok")
        queue = _queue(
            workspace, _both("arm-ok"), {SMOKE_TRACE: _cases(3), RUN_TRACE: _cases(10)}
        )

        outcomes = queue.run([arm])

        assert [(o.name, o.status) for o in outcomes] == [("arm-ok", "valid")]
        record = queue.registry.get("arm-ok")
        assert record is not None
        assert record.status == "valid"
        assert record.trace_id == RUN_TRACE
        assert record.cases == 10
        assert record.concurrency is not None and record.concurrency >= 1
        assert record.git_sha is not None and record.git_sha.startswith(workspace.sha)
        assert (
            "fake run arm-ok-smoke"
            in (workspace.tmp / "logs" / "arm-ok-smoke.log").read_text()
        )
        assert "fake run arm-ok" in (workspace.tmp / "logs" / "arm-ok.log").read_text()
        queue_log = (workspace.tmp / "logs" / "queue.log").read_text()
        assert "registered arm-ok" in queue_log
        assert "smoke 0_0" in queue_log

    def test_result_files_replace_the_span_checks(
        self, workspace: SimpleNamespace
    ) -> None:
        arm = _arm(workspace, "arm-ok")
        queue = _queue(
            workspace, {}, {RUN_TRACE: []}, results_dir=workspace.tmp / "results"
        )

        outcomes = queue.run([arm])

        assert [(o.name, o.status) for o in outcomes] == [("arm-ok", "valid")]
        record = queue.registry.get("arm-ok")
        assert record is not None
        assert record.trace_id == RUN_TRACE
        assert record.cases == 3
        assert record.wall_seconds == pytest.approx(3600.0)
        assert (
            workspace.tmp / "results" / f"arm-ok-smoke.{SMOKE_TRACE[:12]}.jsonl"
        ).exists()
        assert "smoke 0_0" in (workspace.tmp / "logs" / "queue.log").read_text()

    def test_the_smoke_is_optional(self, workspace: SimpleNamespace) -> None:
        arm = _arm(workspace, "arm-ok", smoke_ids=None)
        queue = _queue(workspace, {"arm-ok": RUN_TRACE}, {RUN_TRACE: _cases(2)})
        outcomes = queue.run([arm])
        assert outcomes[0].status == "valid"
        assert not (workspace.tmp / "logs" / "arm-ok-smoke.log").exists()

    def test_a_failed_preflight_skips_the_arm_and_continues(
        self, workspace: SimpleNamespace
    ) -> None:
        bad = _arm(workspace, "arm-bad", sha="0" * 12)
        good = _arm(workspace, "arm-ok")
        queue = _queue(
            workspace, _both("arm-ok"), {SMOKE_TRACE: _cases(3), RUN_TRACE: _cases(10)}
        )

        outcomes = queue.run([bad, good])

        assert [(o.name, o.status) for o in outcomes] == [
            ("arm-bad", "skipped"),
            ("arm-ok", "valid"),
        ]
        assert "worktree" in outcomes[0].detail
        assert queue.registry.get("arm-bad") is None

    def test_a_smoke_without_a_case_span_stops_the_arm(
        self, workspace: SimpleNamespace
    ) -> None:
        arm = _arm(workspace, "arm-ok")
        queue = _queue(workspace, {"arm-ok": RUN_TRACE}, {RUN_TRACE: _cases(10)})
        outcomes = queue.run([arm])
        assert outcomes[0].status == "skipped"
        assert "smoke" in outcomes[0].detail
        assert queue.registry.get("arm-ok") is None
        assert not (workspace.tmp / "logs" / "arm-ok.log").exists()

    def test_a_smoke_that_exits_non_zero_stops_the_arm(
        self, workspace: SimpleNamespace
    ) -> None:
        arm = _arm(workspace, "arm-flaky")
        queue = _queue(
            workspace, {}, {RUN_TRACE: []}, results_dir=workspace.tmp / "results"
        )

        outcomes = queue.run([arm])

        assert outcomes[0].status == "skipped"
        assert "smoke" in outcomes[0].detail and "exit code 3" in outcomes[0].detail
        assert queue.registry.get("arm-flaky") is None
        assert not (workspace.tmp / "logs" / "arm-flaky.log").exists()

    def test_an_unexpected_failure_skips_the_arm_and_the_queue_continues(
        self, workspace: SimpleNamespace
    ) -> None:
        missing = _arm(
            workspace,
            "arm-missing",
            worktree=str(workspace.tmp / "absent"),
            sha="0" * 40,
            smoke_ids=None,
        )
        good = _arm(workspace, "arm-ok", smoke_ids=None)
        queue = _queue(
            workspace,
            {"arm-ok": RUN_TRACE},
            {RUN_TRACE: _cases(2)},
            repo=workspace.repo,
        )

        outcomes = queue.run([missing, good])

        assert [(o.name, o.status) for o in outcomes] == [
            ("arm-missing", "failed"),
            ("arm-ok", "valid"),
        ]
        assert "CalledProcessError" in outcomes[0].detail
        assert queue.registry.get("arm-missing") is None
        assert (
            "arm-missing: failed" in (workspace.tmp / "logs" / "queue.log").read_text()
        )

    def test_a_deadline_kills_the_run_and_voids_the_row(
        self, workspace: SimpleNamespace
    ) -> None:
        arm = _arm(workspace, "arm-hang", deadline_hours=1 / 3600)
        queue = _queue(
            workspace, _both("arm-hang"), {SMOKE_TRACE: _cases(3), RUN_TRACE: []}
        )

        started = time.monotonic()
        outcomes = queue.run([arm])

        assert time.monotonic() - started < 20
        assert outcomes[0].status == "void"
        assert "deadline" in outcomes[0].detail
        record = queue.registry.get("arm-hang")
        assert record is not None
        assert record.status == "void"
        assert record.void_reason is not None and "deadline" in record.void_reason

    def test_a_non_zero_exit_voids_with_the_code_and_keeps_metrics(
        self, workspace: SimpleNamespace
    ) -> None:
        arm = _arm(workspace, "arm-fail")
        queue = _queue(
            workspace, _both("arm-fail"), {SMOKE_TRACE: _cases(3), RUN_TRACE: _cases(5)}
        )
        outcomes = queue.run([arm])
        assert outcomes[0].status == "void"
        assert "exit code 3" in outcomes[0].detail
        record = queue.registry.get("arm-fail")
        assert record is not None
        assert record.void_reason == "exit code 3"
        assert record.trace_id == RUN_TRACE and record.cases == 5

    def test_no_trace_after_settling_is_void_no_telemetry(
        self, workspace: SimpleNamespace
    ) -> None:
        arm = _arm(workspace, "arm-ok")
        queue = _queue(
            workspace, {"arm-ok-smoke": SMOKE_TRACE}, {SMOKE_TRACE: _cases(3)}
        )
        outcomes = queue.run([arm])
        assert outcomes[0].status == "void"
        record = queue.registry.get("arm-ok")
        assert record is not None and record.void_reason == "no telemetry"

    def test_a_registered_name_is_refused(self, workspace: SimpleNamespace) -> None:
        arm = _arm(workspace, "arm-ok")
        queue = _queue(
            workspace, _both("arm-ok"), {SMOKE_TRACE: _cases(3), RUN_TRACE: _cases(1)}
        )
        queue.registry.register_launch(
            ArmRecord(
                name="arm-ok",
                dataset="frames",
                kind="qa",
                status="launched",
                started_at="2026-01-01T00:00:00+00:00",
                source="harness",
            )
        )
        outcomes = queue.run([arm])
        assert outcomes[0].status == "skipped"
        assert "registered" in outcomes[0].detail
        record = queue.registry.get("arm-ok")
        assert record is not None and record.status == "launched"

    def test_provisions_a_missing_worktree_from_the_repo(
        self, workspace: SimpleNamespace
    ) -> None:
        target = workspace.tmp / "provisioned" / "arm-ok"
        arm = _arm(workspace, "arm-ok", worktree=str(target), smoke_ids=None)
        queue = _queue(
            workspace,
            {"arm-ok": RUN_TRACE},
            {RUN_TRACE: _cases(1)},
            repo=workspace.repo,
            env_source=workspace.repo / ".env",
        )

        outcomes = queue.run([arm])

        assert outcomes[0].status == "valid", outcomes[0].detail
        assert _git(target, "rev-parse", "HEAD").startswith(workspace.sha)
        assert (target / ".env").read_text() == (workspace.repo / ".env").read_text()
        assert (target / ".venv").exists()


class TestProvisionWorktree:
    def test_checks_out_the_sha_syncs_and_copies_env(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-q")
        (repo / "pyproject.toml").write_text(PYPROJECT)
        _commit_all(repo)
        sha = _git(repo, "rev-parse", "HEAD")
        env = tmp_path / "env"
        env.write_text("LOGFIRE_TOKEN=x\n")
        target = tmp_path / "wt" / "probe"

        provision_worktree(repo, sha[:12], target, env)

        assert _git(target, "rev-parse", "HEAD") == sha
        assert (target / ".env").read_text() == "LOGFIRE_TOKEN=x\n"
        assert (target / ".venv").exists()


class TestTmuxCommand:
    def test_builds_a_detached_session_in_the_current_directory(self) -> None:
        assert tmux_command(
            "night", ["evaluations", "queue", "a b.yaml"], Path("/w")
        ) == [
            "tmux",
            "new-session",
            "-d",
            "-s",
            "night",
            "-c",
            "/w",
            "evaluations queue 'a b.yaml'",
        ]


class TestQueueCommand:
    def test_exit_status_follows_the_outcomes(
        self, workspace: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import evaluations.benchmark as benchmark

        good = _arm(workspace, "arm-ok", smoke_ids=None)
        bad = _arm(workspace, "arm-bad", sha="0" * 12, smoke_ids=None)
        monkeypatch.setattr(
            benchmark,
            "query_logfire",
            _fake_query({"arm-ok": RUN_TRACE}, {RUN_TRACE: _cases(2)}),
        )
        common = [
            "--registry",
            str(workspace.tmp / "registry.sqlite"),
            "--logs",
            str(workspace.tmp / "logs"),
            "--prefix",
            sys.executable,
            "--prefix",
            str(workspace.fake),
            "--settle-minutes",
            "0.005",
        ]

        passing = CliRunner().invoke(benchmark.app, ["queue", str(good), *common])
        assert passing.exit_code == 0, passing.output
        assert "arm-ok" in passing.output and "valid" in passing.output

        failing = CliRunner().invoke(benchmark.app, ["queue", str(bad), *common])
        assert failing.exit_code == 1
        assert "skipped" in failing.output
