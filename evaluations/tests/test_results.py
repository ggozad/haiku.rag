import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_evals import Case, Dataset, set_eval_attribute
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from typer.testing import CliRunner

from evaluations.capability_runner import CapabilityRunResult
from evaluations.completion import complete_arm
from evaluations.config import DatasetSpec
from evaluations.evaluators import NumberMatchEvaluator
from evaluations.qa import run_qa_benchmark
from evaluations.registry import ArmRecord, Registry
from evaluations.results import find_results, read_results, write_results
from evaluations.traces import CaseOutcome
from haiku.rag.config.models import AppConfig

TRACE = "3" * 32


@dataclass
class Judge(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> dict[str, bool | float]:
        return {
            "answer_equivalent": ctx.output == ctx.expected_output,
            "cited_map": 0.5,
        }


@dataclass
class Numbers(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> dict[str, float]:
        return {"number_match": 1.0 if ctx.output == ctx.expected_output else 0.0}


async def _task(question: str) -> str:
    if question == "boom":
        raise RuntimeError("dead")
    if question == "cite":
        set_eval_attribute("cited_uris", ["u1"])
    return "other" if question == "wrong" else "answer"


async def _report(evaluator: Evaluator = Judge()):
    dataset = Dataset(
        name="run-x",
        cases=[
            Case(
                name="1_a",
                inputs="cite",
                expected_output="answer",
                metadata={"query_id": "a"},
            ),
            Case(
                name="2_b",
                inputs="wrong",
                expected_output="answer",
                metadata={"query_id": "b"},
            ),
            Case(
                name="3_c",
                inputs="boom",
                expected_output="answer",
                metadata={"query_id": "c"},
            ),
            Case(name="4_d", inputs="plain", expected_output="answer"),
        ],
        evaluators=[evaluator],
    )
    return await dataset.evaluate(
        _task, name="run-x", max_concurrency=1, progress=False
    )


class TestWriteResults:
    async def test_writes_one_row_per_case_and_failure(self, tmp_path: Path) -> None:
        report = await _report()

        path = write_results(
            report, name="run-x", pair_key="query_id", directory=tmp_path
        )

        assert path.parent == tmp_path
        assert path.name.startswith("run-x.") and path.suffix == ".jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [row["case_name"] for row in rows] == ["1_a", "2_b", "3_c", "4_d"]
        by_name = {row["case_name"]: row for row in rows}
        first = by_name["1_a"]
        assert {k: first[k] for k in CaseOutcome.__dataclass_fields__} == {
            "case_name": "1_a",
            "key": "a",
            "passed": True,
            "cited": True,
            "cited_map": 0.5,
            "aborted": False,
        }
        assert first["trace_id"] == report.trace_id
        assert first["answer"] == "answer"
        assert first["reason"] is None
        assert first["attributes"] == {"cited_uris": ["u1"]}
        assert isinstance(first["task_duration"], float)
        assert by_name["2_b"]["passed"] is False
        assert by_name["2_b"]["cited"] is False
        assert by_name["2_b"]["answer"] == "other"
        failed = by_name["3_c"]
        assert failed["aborted"] is True
        assert failed["passed"] is None
        assert failed["cited_map"] is None
        assert "dead" in failed["reason"]
        assert failed["attributes"] == {} and failed["task_duration"] is None
        assert by_name["4_d"]["key"] is None

    async def test_rows_are_written_as_cases_finish(self, tmp_path: Path) -> None:
        """A killed run keeps what it finished, and a watcher can see it move."""
        from evaluations.results import case_writer, partial_path

        dataset = Dataset(
            name="run-x",
            cases=[
                Case(name="1_a", inputs="cite", expected_output="answer"),
                Case(name="2_b", inputs="boom", expected_output="answer"),
            ],
            evaluators=[Judge()],
        )
        partial = partial_path(tmp_path, "run-x")

        report = await dataset.evaluate(
            _task,
            name="run-x",
            max_concurrency=1,
            progress=False,
            lifecycle=case_writer(tmp_path, name="run-x", pair_key="query_id"),
        )

        rows = [json.loads(line) for line in partial.read_text().splitlines()]
        assert {row["case_name"] for row in rows} == {"1_a", "2_b"}
        assert [row["aborted"] for row in rows if row["case_name"] == "2_b"] == [True]
        assert read_results(partial)[1][0].case_name in {"1_a", "2_b"}

        final = write_results(
            report, name="run-x", pair_key="query_id", directory=tmp_path
        )
        assert not partial.exists()
        assert find_results(tmp_path, "run-x") == final

    async def test_a_partial_file_is_the_result_when_nothing_finished_the_run(
        self, tmp_path: Path
    ) -> None:
        partial = tmp_path / "run-x.partial.jsonl"
        partial.write_text(
            json.dumps(
                {
                    "case_name": "1_a",
                    "key": "a",
                    "passed": True,
                    "cited": True,
                    "cited_map": 0.5,
                    "aborted": False,
                    "trace_id": TRACE,
                }
            )
            + "\n"
        )
        assert find_results(tmp_path, "run-x") == partial
        assert read_results(partial)[1][0].case_name == "1_a"

    async def test_a_name_that_is_not_a_file_name_is_refused(
        self, tmp_path: Path
    ) -> None:
        report = await _report()
        for name in ("../escape", "a/b", "/abs", ".hidden", ""):
            with pytest.raises(ValueError, match="run name"):
                write_results(
                    report, name=name, pair_key="query_id", directory=tmp_path
                )
        assert list(tmp_path.iterdir()) == []

    async def test_number_match_scores_pass_at_one(self, tmp_path: Path) -> None:
        report = await _report(Numbers())
        path = write_results(
            report, name="run-x", pair_key="query_id", directory=tmp_path
        )
        rows = {
            json.loads(line)["case_name"]: json.loads(line)
            for line in path.read_text().splitlines()
        }
        assert rows["1_a"]["passed"] is True
        assert rows["2_b"]["passed"] is False

    async def test_the_file_name_carries_the_trace_or_says_it_has_none(
        self, tmp_path: Path
    ) -> None:
        report = await _report()
        path = write_results(
            report, name="run-x", pair_key="query_id", directory=tmp_path
        )
        tag = path.name[len("run-x.") : -len(".jsonl")]
        if report.trace_id is None:
            assert tag.startswith("notrace-")
        else:
            assert tag == report.trace_id[:12]


class TestReadAndFind:
    async def test_round_trips_to_outcomes(self, tmp_path: Path) -> None:
        report = await _report()
        path = write_results(
            report, name="run-x", pair_key="query_id", directory=tmp_path
        )

        trace_id, outcomes = read_results(path)

        assert trace_id == report.trace_id
        assert outcomes[0] == CaseOutcome(
            case_name="1_a",
            key="a",
            passed=True,
            cited=True,
            cited_map=0.5,
            aborted=False,
        )
        assert outcomes[2].aborted is True
        assert outcomes[3].key is None

    def test_find_by_run_name(self, tmp_path: Path) -> None:
        assert find_results(tmp_path, "run-x") is None
        (tmp_path / "run-x.abc.jsonl").write_text("")
        assert find_results(tmp_path, "run-x") == tmp_path / "run-x.abc.jsonl"
        assert find_results(tmp_path, "run") is None
        (tmp_path / "run-x.def.jsonl").write_text("")
        with pytest.raises(ValueError, match="2 result files"):
            find_results(tmp_path, "run-x")


def _record(name: str, **overrides: Any) -> ArmRecord:
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


def _write_rows(
    directory: Path, name: str, trace_id: str | None, passes: list[bool]
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    tag = trace_id[:12] if trace_id else "notrace-20260101T100000Z"
    path = directory / f"{name}.{tag}.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "case_name": f"{i}_{i}",
                    "key": str(i),
                    "passed": passed,
                    "cited": True,
                    "cited_map": 0.5,
                    "aborted": False,
                    "trace_id": trace_id,
                }
            )
            + "\n"
            for i, passed in enumerate(passes)
        )
    )
    return path


class TestCompletionPrefersFiles:
    def test_metrics_and_trace_come_from_the_file(self, tmp_path: Path) -> None:
        registry = Registry(tmp_path / "registry.sqlite")
        registry.register_launch(_record("orb-branch"))
        _write_rows(tmp_path / "results", "orb-branch", TRACE, [True, False, True])
        calls: list[str] = []

        def query(sql: str, *, min_timestamp: str) -> list[dict]:
            calls.append(sql)
            if "evaluate {name}" in sql:
                return [
                    {
                        "trace_id": TRACE,
                        "start_timestamp": "2026-01-01T10:00:00Z",
                        "end_timestamp": "2026-01-01T11:00:00Z",
                    }
                ]
            raise AssertionError("case spans must come from the file")

        complete_arm(
            registry, "orb-branch", query=query, results_dir=tmp_path / "results"
        )

        record = registry.get("orb-branch")
        assert record is not None
        assert record.status == "valid"
        assert record.trace_id == TRACE
        assert record.cases == 3
        assert record.accuracy == pytest.approx(2 / 3)
        assert record.wall_seconds == pytest.approx(3600.0)
        assert calls and all("evaluate {name}" in call for call in calls)

    def test_a_file_without_a_trace_completes_without_logfire(
        self, tmp_path: Path
    ) -> None:
        registry = Registry(tmp_path / "registry.sqlite")
        registry.register_launch(_record("orb-branch"))
        _write_rows(tmp_path / "results", "orb-branch", None, [True, True])

        def never(sql: str, *, min_timestamp: str) -> list[dict]:
            raise AssertionError("no Logfire query without a trace")

        complete_arm(
            registry, "orb-branch", query=never, results_dir=tmp_path / "results"
        )

        record = registry.get("orb-branch")
        assert record is not None
        assert record.status == "valid"
        assert record.trace_id is None
        assert record.cases == 2 and record.accuracy == 1.0


class TestPairCommandReadsFiles:
    def test_pairs_from_result_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import evaluations.benchmark as benchmark

        registry_path = tmp_path / "registry.sqlite"
        registry = Registry(registry_path)
        registry.register_launch(
            _record(
                "main",
                status="valid",
                trace_id="1" * 32,
                git_sha="a" * 40,
                config_hash="c" * 64,
            )
        )
        registry.register_launch(
            _record(
                "branch",
                status="valid",
                trace_id="2" * 32,
                git_sha="b" * 40,
                config_hash="c" * 64,
                comparator="main",
                differences='["sha"]',
            )
        )
        _write_rows(tmp_path / "results", "main", "1" * 32, [True, False, True, False])
        _write_rows(tmp_path / "results", "branch", "2" * 32, [True, True, True, False])

        def never(sql: str, *, min_timestamp: str) -> list[dict]:
            raise AssertionError("result files must be preferred over Logfire")

        monkeypatch.setattr(benchmark, "query_logfire", never)
        result = CliRunner().invoke(
            benchmark.app,
            [
                "arms",
                "pair",
                "main",
                "branch",
                "--registry",
                str(registry_path),
                "--results",
                str(tmp_path / "results"),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "paired on query_id: 4 cases" in result.output


def _spec() -> DatasetSpec:
    return DatasetSpec(
        key="test",
        db_filename="test.lancedb",
        document_loader=lambda: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        document_mapper=lambda doc: None,
        qa_loader=lambda: [{"question": "What is X?", "answer": "42"}],  # type: ignore[arg-type,return-value]  # ty: ignore[invalid-argument-type]
        qa_case_builder=lambda idx, doc: Case(
            name=f"case-{idx}",
            inputs=doc["question"],
            expected_output=doc["answer"],
            metadata={"question_id": str(idx)},
        ),
        qa_evaluator=NumberMatchEvaluator(),
    )


class TestRunWritesResults:
    @pytest.mark.asyncio
    async def test_qa_benchmark_writes_per_case_results(self, tmp_path: Path) -> None:
        with patch(
            "evaluations.qa.run_capability_question",
            new_callable=AsyncMock,
            return_value=CapabilityRunResult(answer="ANSWER: 42"),
        ):
            await run_qa_benchmark(
                _spec(),
                AppConfig(),
                db_path=tmp_path / "test.lancedb",
                results_dir=tmp_path / "results",
            )

        files = list((tmp_path / "results").glob("test_qa_evaluation.*.jsonl"))
        assert len(files) == 1
        _, outcomes = read_results(files[0])
        assert [(o.case_name, o.key, o.passed) for o in outcomes] == [
            ("case-1", "1", True)
        ]

    @pytest.mark.asyncio
    async def test_without_a_directory_nothing_is_written(self, tmp_path: Path) -> None:
        with patch(
            "evaluations.qa.run_capability_question",
            new_callable=AsyncMock,
            return_value=CapabilityRunResult(answer="ANSWER: 42"),
        ):
            await run_qa_benchmark(
                _spec(), AppConfig(), db_path=tmp_path / "test.lancedb"
            )
        assert not (tmp_path / "results").exists()

    @pytest.mark.asyncio
    async def test_evaluate_dataset_threads_the_directory(self, tmp_path: Path) -> None:
        from evaluations.benchmark import evaluate_dataset

        with patch(
            "evaluations.benchmark.run_qa_benchmark", new_callable=AsyncMock
        ) as qa:
            await evaluate_dataset(
                spec=_spec(),
                config=AppConfig(),
                skip_db=True,
                skip_retrieval=True,
                skip_qa=False,
                limit=None,
                name=None,
                db_path=None,
                results_dir=tmp_path / "results",
            )
        assert qa.call_args[1]["results_dir"] == tmp_path / "results"
