"""Run arms in order: preflight, smoke, register, run under a deadline, complete."""

import asyncio
import os
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from evaluations.arm import ArmSpec, load_arm
from evaluations.completion import (
    complete_arm,
    find_trace,
    running_evaluations,
    window_start,
)
from evaluations.datasets import DATASETS
from evaluations.preflight import run_preflight
from evaluations.registry import Registry, launch_record
from evaluations.traces import QueryFn, case_outcomes
from haiku.rag.utils import get_default_data_dir


def default_logs_path() -> Path:
    return get_default_data_dir() / "evaluations" / "logs"


def provision_worktree(
    repo: Path, sha: str, path: Path, env_source: Path | None
) -> None:
    """Check `sha` out at `path` as a worktree of `repo`, install it, copy `.env`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "--detach", str(path), sha],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["uv", "sync"], cwd=path, check=True, capture_output=True, text=True)
    if env_source is not None:
        shutil.copy(env_source, path / ".env")


def tmux_command(session: str, argv: list[str], cwd: Path) -> list[str]:
    """A detached tmux session running `argv` from `cwd`."""
    return [
        "tmux",
        "new-session",
        "-d",
        "-s",
        session,
        "-c",
        str(cwd),
        shlex.join(argv),
    ]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class ArmOutcome:
    name: str
    status: str
    detail: str


@dataclass
class Queue:
    """Runs arm files in order. Each arm is preflighted, smoked when it names
    `smoke_ids`, registered, run from its worktree with output appended to
    `logs/<name>.log`, killed at its deadline, and completed from its trace.
    A failed step skips the arm and the queue continues."""

    registry: Registry
    logs: Path
    query: QueryFn
    prefix: list[str] = field(default_factory=lambda: ["uv", "run"])
    repo: Path | None = None
    env_source: Path | None = None
    settle_seconds: float = 900.0
    poll_seconds: float = 30.0

    def run(self, arm_paths: list[Path]) -> list[ArmOutcome]:
        self.logs.mkdir(parents=True, exist_ok=True)
        outcomes: list[ArmOutcome] = []
        for path in arm_paths:
            outcome = self._run_one(path)
            self._log(f"{outcome.name}: {outcome.status}: {outcome.detail}")
            outcomes.append(outcome)
        return outcomes

    def _log(self, message: str) -> None:
        line = f"{_now()} {message}"
        print(line, flush=True)
        with (self.logs / "queue.log").open("a") as log:
            log.write(line + "\n")

    def _run_one(self, arm_path: Path) -> ArmOutcome:
        try:
            arm = load_arm(arm_path)
        except Exception as error:  # any loader failure is this arm's failure
            return ArmOutcome(arm_path.stem, "skipped", f"arm file: {error}")
        if self.registry.get(arm.name) is not None:
            return ArmOutcome(
                arm.name, "skipped", "already registered; a rerun needs a new name"
            )
        if not arm.worktree.exists() and self.repo is not None:
            provision_worktree(self.repo, arm.sha, arm.worktree, self.env_source)
            self._log(f"{arm.name}: provisioned {arm.worktree} at {arm.sha}")
        preflight = asyncio.run(run_preflight(arm_path))
        for check in preflight.checks:
            label = "ok  " if check.ok else "FAIL"
            self._log(f"{arm.name}: {label} {check.name}: {check.detail}")
        if not preflight.ok:
            failed = ", ".join(check.name for check in preflight.checks if not check.ok)
            return ArmOutcome(arm.name, "skipped", f"preflight failed: {failed}")
        assert preflight.config is not None
        key = DATASETS[arm.dataset].pair_key

        if arm.smoke_ids is not None:
            smoke_name = f"{arm.name}-smoke"
            since = window_start(_now())
            self._execute(arm, arm.command(smoke=True), smoke_name, arm.deadline_hours)
            trace = self._await_trace(smoke_name, since)
            outcomes = (
                []
                if trace is None
                else case_outcomes(trace, key, query=self.query, min_timestamp=since)
            )
            if not outcomes:
                return ArmOutcome(
                    arm.name, "skipped", "smoke produced no case span in Logfire"
                )
            for outcome in outcomes:
                self._log(
                    f"smoke {outcome.case_name}: passed={outcome.passed} "
                    f"cited={outcome.cited} aborted={outcome.aborted}"
                )

        record = launch_record(
            arm,
            preflight.config,
            preflight.fingerprint,
            started_at=_now(),
            git_sha=preflight.git_sha,
        )
        record.concurrency = running_evaluations() + 1
        self.registry.register_launch(record)
        self._log(f"registered {arm.name} (concurrency {record.concurrency})")

        code = self._execute(arm, arm.command(), arm.name, arm.deadline_hours)
        void_reason = None
        if code is None:
            void_reason = f"deadline of {arm.deadline_hours} h exceeded"
        elif code != 0:
            void_reason = f"exit code {code}"
        try:
            trace = self._await_trace(arm.name, window_start(record.started_at))
            if trace is None:
                reason = (
                    "no telemetry"
                    if void_reason is None
                    else f"no telemetry; {void_reason}"
                )
                self.registry.mark_void(arm.name, reason)
            else:
                complete_arm(
                    self.registry,
                    arm.name,
                    query=self.query,
                    trace_id=trace,
                    void_reason=void_reason,
                )
        except ValueError as error:
            self.registry.mark_void(arm.name, str(error))
        final = self.registry.get(arm.name)
        assert final is not None
        detail = final.void_reason or (
            f"{final.cases} cases, accuracy {final.accuracy}, trace {final.trace_id}"
        )
        return ArmOutcome(arm.name, final.status, detail)

    def _execute(
        self, arm: ArmSpec, argv: list[str], log_name: str, deadline_hours: float | None
    ) -> int | None:
        """Run `argv` from the arm's worktree with output appended to
        `logs/<log_name>.log`. The exit code, or None when the deadline killed it."""
        self._log(f"{log_name}: {' '.join(argv)}")
        deadline = (
            None if deadline_hours is None else time.monotonic() + deadline_hours * 3600
        )
        with (self.logs / f"{log_name}.log").open("ab") as log:
            process = subprocess.Popen(
                [*self.prefix, *argv],
                cwd=arm.worktree,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                start_new_session=True,
            )
            while True:
                try:
                    return process.wait(timeout=self.poll_seconds)
                except subprocess.TimeoutExpired:
                    pass
                if deadline is not None and time.monotonic() >= deadline:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    return None

    def _await_trace(self, name: str, since: str) -> str | None:
        """The run's trace, polled until spans arrive or `settle_seconds` pass."""
        end = time.monotonic() + self.settle_seconds
        while True:
            trace = find_trace(name, since, query=self.query)
            remaining = end - time.monotonic()
            if trace is not None or remaining <= 0:
                return trace
            time.sleep(min(self.poll_seconds, remaining))
