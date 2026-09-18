import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evaluations.completion import (
    complete_arm,
    count_evaluation_runs,
    find_trace,
    window_start,
)
from evaluations.registry import ArmRecord, Registry

TRACE = "3" * 32


def _record(name: str = "orb-branch", **overrides: Any) -> ArmRecord:
    fields: dict[str, Any] = {
        "name": name,
        "dataset": "orb_text",
        "kind": "qa",
        "status": "launched",
        "started_at": "2026-01-01T10:00:00+00:00",
        "source": "harness",
    }
    fields.update(overrides)
    return ArmRecord(**fields)


def _case(
    i: int, passed: bool | None, cited_map: float | None = 0.5, aborted: bool = False
) -> dict:
    return {
        "case_name": f"{i}_{i}",
        "pair_key": str(i),
        "answer_equivalent": None if passed is None else str(passed).lower(),
        "number_match": None,
        "cited_map": cited_map,
        "cited_uris": '["u1"]' if cited_map else "[]",
        "is_exception": aborted,
    }


def _fake_query(experiments: list[dict], cases: list[dict]):
    calls: list[str] = []

    def query(sql: str, *, min_timestamp: str) -> list[dict]:
        calls.append(sql)
        if "evaluate {name}" in sql:
            return experiments
        if "count(*)" in sql:
            return [{"n": len(cases)}]
        return cases

    return query, calls


_EXPERIMENT = {
    "trace_id": TRACE,
    "start_timestamp": "2026-01-01T10:05:00Z",
    "end_timestamp": "2026-01-01T12:05:00Z",
}


class TestWindowStart:
    def test_starts_an_hour_before_launch_in_utc(self) -> None:
        assert window_start("2026-01-01T10:00:00+00:00") == "2026-01-01T09:00:00Z"
        assert window_start("2026-01-01T12:00:00+02:00") == "2026-01-01T09:00:00Z"


class TestFindTrace:
    def test_one_match_returns_the_trace(self) -> None:
        query, calls = _fake_query([_EXPERIMENT], [])
        assert find_trace("orb-branch", "2026-01-01T09:00:00Z", query=query) == TRACE
        assert "attributes->>'name' = 'orb-branch'" in calls[0]
        assert "start_timestamp >= '2026-01-01T09:00:00Z'" in calls[0]
        assert "span_name = 'evaluate {name}'" in calls[0]

    def test_no_match_is_none(self) -> None:
        query, _ = _fake_query([], [])
        assert find_trace("orb-branch", "t", query=query) is None

    def test_more_than_one_match_raises(self) -> None:
        query, _ = _fake_query([_EXPERIMENT, {**_EXPERIMENT, "trace_id": "4" * 32}], [])
        with pytest.raises(ValueError, match="2 traces"):
            find_trace("orb-branch", "t", query=query)

    def test_rejects_a_name_that_cannot_be_quoted(self) -> None:
        query, _ = _fake_query([], [])
        with pytest.raises(ValueError, match="name"):
            find_trace("it's", "t", query=query)


class TestCompleteArm:
    def test_fills_the_row_from_the_trace(self, tmp_path: Path) -> None:
        registry = Registry(tmp_path / "registry.sqlite")
        registry.register_launch(_record())
        cases = [
            _case(1, True, 1.0),
            _case(2, False, 0.0),
            _case(3, True, 0.5),
            _case(4, None, None, aborted=True),
        ]
        query, _ = _fake_query([_EXPERIMENT], cases)

        summary = complete_arm(registry, "orb-branch", query=query)

        record = registry.get("orb-branch")
        assert record is not None
        assert record.status == "valid"
        assert record.trace_id == TRACE
        assert record.cases == 4
        assert record.accuracy == pytest.approx(2 / 3)
        assert record.cite_rate_all_cases == pytest.approx(0.5)
        assert record.cited_map == pytest.approx(0.5)
        assert record.aborts == 1
        assert record.wall_seconds == pytest.approx(7200.0)
        assert record.ended_at == "2026-01-01T12:05:00Z"
        assert summary is not None and summary.cases == 4

    def test_a_given_trace_id_skips_the_name_lookup(self, tmp_path: Path) -> None:
        registry = Registry(tmp_path / "registry.sqlite")
        registry.register_launch(_record())
        query, calls = _fake_query([_EXPERIMENT], [_case(1, True)])

        complete_arm(registry, "orb-branch", trace_id=TRACE, query=query)

        assert not any("attributes->>'name'" in call for call in calls)
        record = registry.get("orb-branch")
        assert record is not None and record.trace_id == TRACE

    def test_no_trace_marks_the_arm_void(self, tmp_path: Path) -> None:
        registry = Registry(tmp_path / "registry.sqlite")
        registry.register_launch(_record())
        query, _ = _fake_query([], [])

        assert complete_arm(registry, "orb-branch", query=query) is None

        record = registry.get("orb-branch")
        assert record is not None
        assert record.status == "void"
        assert record.void_reason == "no telemetry"

    def test_a_result_file_completes_while_logfire_is_unreachable(
        self, tmp_path: Path
    ) -> None:
        registry = Registry(tmp_path / "registry.sqlite")
        registry.register_launch(_record())
        results = tmp_path / "results"
        results.mkdir()
        rows = [
            {
                "case_name": f"{i}_{i}",
                "key": str(i),
                "passed": i % 2 == 0,
                "cited": True,
                "cited_map": 0.5,
                "aborted": False,
                "trace_id": TRACE,
            }
            for i in range(2)
        ]
        (results / f"orb-branch.{TRACE[:12]}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

        def unreachable(sql: str, *, min_timestamp: str) -> list[dict]:
            raise OSError("logfire is unreachable")

        summary = complete_arm(
            registry, "orb-branch", query=unreachable, results_dir=results
        )

        assert summary is not None and summary.cases == 2
        record = registry.get("orb-branch")
        assert record is not None
        assert record.status == "valid"
        assert record.trace_id == TRACE
        assert record.cases == 2
        assert record.wall_seconds is None

    def test_an_ambiguous_name_leaves_the_row_alone(self, tmp_path: Path) -> None:
        registry = Registry(tmp_path / "registry.sqlite")
        registry.register_launch(_record())
        query, _ = _fake_query([_EXPERIMENT, {**_EXPERIMENT, "trace_id": "4" * 32}], [])
        with pytest.raises(ValueError, match="2 traces"):
            complete_arm(registry, "orb-branch", query=query)
        record = registry.get("orb-branch")
        assert record is not None and record.status == "launched"

    def test_an_unknown_arm_raises(self, tmp_path: Path) -> None:
        registry = Registry(tmp_path / "registry.sqlite")
        query, _ = _fake_query([], [])
        with pytest.raises(ValueError, match="ghost"):
            complete_arm(registry, "ghost", query=query)

    def test_a_failed_run_records_its_metrics_and_stays_void(
        self, tmp_path: Path
    ) -> None:
        registry = Registry(tmp_path / "registry.sqlite")
        registry.register_launch(_record())
        query, _ = _fake_query([_EXPERIMENT], [_case(1, True)])

        complete_arm(registry, "orb-branch", query=query, void_reason="exit code 137")

        record = registry.get("orb-branch")
        assert record is not None
        assert record.status == "void"
        assert record.void_reason == "exit code 137"
        assert record.trace_id == TRACE and record.cases == 1


class TestCountEvaluationRuns:
    def test_counts_run_commands_and_ignores_the_rest(self) -> None:
        lines = [
            "/usr/bin/python3 /home/x/.venv/bin/evaluations run frames --limit 150",
            "/bin/sh -c uv run evaluations run orb_text --skip-db",
            "grep evaluations run",
            "/usr/bin/python3 -m evaluations.benchmark arms list",
            "tmux new-session -d -s q",
        ]
        assert count_evaluation_runs(lines) == 2


class TestCompleteCommand:
    def test_completes_a_registered_arm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import evaluations.benchmark as benchmark

        registry_path = tmp_path / "registry.sqlite"
        Registry(registry_path).register_launch(_record())
        query, _ = _fake_query([_EXPERIMENT], [_case(1, True), _case(2, False)])
        monkeypatch.setattr(benchmark, "query_logfire", query)

        result = CliRunner().invoke(
            benchmark.app,
            ["arms", "complete", "orb-branch", "--registry", str(registry_path)],
        )

        assert result.exit_code == 0, result.output
        assert TRACE in result.output
        record = Registry(registry_path).get("orb-branch")
        assert record is not None and record.status == "valid" and record.cases == 2

    def test_reports_a_void_completion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import evaluations.benchmark as benchmark

        registry_path = tmp_path / "registry.sqlite"
        Registry(registry_path).register_launch(_record())
        query, _ = _fake_query([], [])
        monkeypatch.setattr(benchmark, "query_logfire", query)

        result = CliRunner().invoke(
            benchmark.app,
            ["arms", "complete", "orb-branch", "--registry", str(registry_path)],
        )

        assert result.exit_code == 1
        assert "no telemetry" in result.output
