"""Run arms in order: preflight, smoke, register, run, complete."""

import asyncio
import os
import shlex
import shutil
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
from evaluations.results import (
    RESULTS_ENV,
    find_results,
    partial_path,
    read_results,
)
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


def _failure_reason(code: int) -> str | None:
    """Why a run counts as failed, or None when it exited cleanly."""
    return None if code == 0 else f"exit code {code}"


@dataclass
class ArmOutcome:
    name: str
    status: str
    detail: str


@dataclass
class Queue:
    """Runs arm files in order. Each arm is preflighted, smoked when it names
    `smoke_ids`, registered, run from its worktree with output appended to
    `logs/<name>.log`, and completed from its trace.
    A failed step ends that arm and the queue continues; an arm that fails
    after it was registered keeps its launched row, to complete by hand.
    A run that finishes no case for `stall_seconds` is called out in the log
    and left alone: a stall is for a person to read, not for the queue to act
    on."""

    registry: Registry
    logs: Path
    query: QueryFn
    prefix: list[str] = field(default_factory=lambda: ["uv", "run"])
    repo: Path | None = None
    env_source: Path | None = None
    settle_seconds: float = 900.0
    poll_seconds: float = 30.0
    stall_seconds: float = 600.0
    results_dir: Path | None = None

    def run(self, arm_paths: list[Path]) -> list[ArmOutcome]:
        self.logs.mkdir(parents=True, exist_ok=True)
        outcomes: list[ArmOutcome] = []
        for path in arm_paths:
            try:
                outcome = self._run_one(path)
            except Exception as error:  # one arm's failure ends that arm only
                outcome = ArmOutcome(
                    path.stem, "failed", f"{type(error).__name__}: {error}"
                )
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
        preflight = asyncio.run(run_preflight(arm_path, self.registry))
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
            code = self._execute(arm, arm.command(smoke=True), smoke_name)
            failed = _failure_reason(code)
            if failed is not None:
                return ArmOutcome(arm.name, "skipped", f"smoke failed: {failed}")
            outcomes = self._file_outcomes(smoke_name)
            if outcomes is None:
                trace = self._await_trace(smoke_name, since)
                outcomes = (
                    []
                    if trace is None
                    else case_outcomes(
                        trace, key, query=self.query, min_timestamp=since
                    )
                )
            if not outcomes:
                return ArmOutcome(
                    arm.name,
                    "skipped",
                    "smoke produced no result file and no case span",
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

        code = self._execute(arm, arm.command(), arm.name)
        void_reason = _failure_reason(code)
        try:
            if self._file_outcomes(arm.name) is not None:
                complete_arm(
                    self.registry,
                    arm.name,
                    query=self.query,
                    void_reason=void_reason,
                    results_dir=self.results_dir,
                )
            else:
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

    def _file_outcomes(self, name: str):
        """The run's result file rows, or None when no file exists."""
        if self.results_dir is None:
            return None
        path = find_results(self.results_dir, name)
        return None if path is None else read_results(path)[1]

    def _execute(self, arm: ArmSpec, argv: list[str], log_name: str) -> int:
        """Run `argv` from the arm's worktree with output appended to
        `logs/<log_name>.log`, and return its exit code. Nothing kills the run:
        a stall is called out for a person to act on."""
        self._log(f"{log_name}: {' '.join(argv)}")
        with (self.logs / f"{log_name}.log").open("ab") as log:
            process = subprocess.Popen(
                [*self.prefix, *argv],
                cwd=arm.worktree,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    **(
                        {RESULTS_ENV: str(self.results_dir)} if self.results_dir else {}
                    ),
                },
                start_new_session=True,
            )
            cases_bytes = 0
            moved_at = time.monotonic()
            while True:
                try:
                    return process.wait(timeout=self.poll_seconds)
                except subprocess.TimeoutExpired:
                    pass
                written = self._cases_written(log_name)
                if written != cases_bytes:
                    cases_bytes, moved_at = written, time.monotonic()
                elif written and time.monotonic() - moved_at >= self.stall_seconds:
                    self._log(
                        f"{log_name}: no case finished in "
                        f"{self.stall_seconds / 60:.0f} min"
                    )
                    moved_at = time.monotonic()

    def _cases_written(self, name: str) -> int:
        """Bytes the run has appended to its result file. Zero before its first
        case, which is not a stall: a build or an ingest writes no case."""
        if self.results_dir is None:
            return 0
        try:
            return partial_path(self.results_dir, name).stat().st_size
        except OSError:
            return 0

    def _await_trace(self, name: str, since: str) -> str | None:
        """The run's trace, polled until spans arrive or `settle_seconds` pass."""
        end = time.monotonic() + self.settle_seconds
        while True:
            trace = find_trace(name, since, query=self.query)
            remaining = end - time.monotonic()
            if trace is not None or remaining <= 0:
                return trace
            time.sleep(min(self.poll_seconds, remaining))
