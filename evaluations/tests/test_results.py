import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_evals import Case, Dataset, set_eval_attribute
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from evaluations.capability_runner import CapabilityRunResult
from evaluations.config import DatasetSpec
from evaluations.evaluators import NumberMatchEvaluator
from evaluations.qa import run_qa_benchmark
from evaluations.results import (
    CaseOutcome,
    case_writer,
    partial_path,
    read_results,
    write_results,
)
from haiku.rag.config.models import AppConfig


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


def _dataset(evaluator: Evaluator = Judge()) -> Dataset:
    return Dataset(
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


async def _report(evaluator: Evaluator = Judge()):
    return await _dataset(evaluator).evaluate(
        _task, name="run-x", max_concurrency=1, progress=False
    )


def _rows(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {row["case_name"]: row for row in rows}


class TestWriteResults:
    async def test_writes_one_row_per_case_and_failure(self, tmp_path: Path) -> None:
        report = await _report()

        path = write_results(
            report, name="run-x", pair_key="query_id", directory=tmp_path, run_id="r1"
        )

        assert path.parent == tmp_path
        lines = path.read_text().splitlines()
        assert [json.loads(line)["case_name"] for line in lines] == [
            "1_a",
            "2_b",
            "3_c",
            "4_d",
        ]
        rows = _rows(path)
        first = rows["1_a"]
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
        assert rows["2_b"]["passed"] is False
        assert rows["2_b"]["cited"] is False
        assert rows["2_b"]["answer"] == "other"
        failed = rows["3_c"]
        assert failed["aborted"] is True
        assert failed["passed"] is None
        assert failed["cited_map"] is None
        assert "dead" in failed["reason"]
        assert failed["attributes"] == {} and failed["task_duration"] is None
        assert rows["4_d"]["key"] is None

    async def test_number_match_scores_pass_at_one(self, tmp_path: Path) -> None:
        report = await _report(Numbers())
        rows = _rows(
            write_results(
                report,
                name="run-x",
                pair_key="query_id",
                directory=tmp_path,
                run_id="r1",
            )
        )
        assert rows["1_a"]["passed"] is True
        assert rows["2_b"]["passed"] is False

    async def test_the_file_name_carries_the_trace_or_says_it_has_none(
        self, tmp_path: Path
    ) -> None:
        report = await _report()
        path = write_results(
            report, name="run-x", pair_key="query_id", directory=tmp_path, run_id="r1"
        )
        tag = path.name[len("run-x.") : -len(".jsonl")]
        if report.trace_id is None:
            assert tag == "notrace-r1"
        else:
            assert tag == report.trace_id

    async def test_runs_started_in_one_millisecond_keep_separate_files(
        self, tmp_path: Path
    ) -> None:
        """A trace id opens with its start time, so two runs of one name
        started together share a prefix."""
        report = await _report()
        paths = set()
        for suffix in ("0" * 20, "f" * 20):
            report.trace_id = "01a0cd944d57" + suffix
            paths.add(
                write_results(
                    report,
                    name="run-x",
                    pair_key="query_id",
                    directory=tmp_path,
                    run_id="r1",
                )
            )
        assert len(paths) == 2

    async def test_a_name_that_is_not_a_file_name_is_refused(
        self, tmp_path: Path
    ) -> None:
        report = await _report()
        for name in ("../escape", "a/b", "/abs", ".hidden", ""):
            with pytest.raises(ValueError, match="run name"):
                write_results(
                    report,
                    name=name,
                    pair_key="query_id",
                    directory=tmp_path,
                    run_id="r1",
                )
            with pytest.raises(ValueError, match="run name"):
                case_writer(tmp_path, name=name, pair_key="query_id", run_id="r1")
        assert list(tmp_path.iterdir()) == []


class TestPartialResults:
    async def test_rows_are_written_as_cases_finish(self, tmp_path: Path) -> None:
        """A killed run keeps the cases it finished."""
        partial = partial_path(tmp_path, "run-x", "r1")

        report = await _dataset().evaluate(
            _task,
            name="run-x",
            max_concurrency=1,
            progress=False,
            lifecycle=case_writer(
                tmp_path, name="run-x", pair_key="query_id", run_id="r1"
            ),
        )

        rows = _rows(partial)
        assert set(rows) == {"1_a", "2_b", "3_c", "4_d"}
        assert rows["3_c"]["aborted"] is True
        assert {o.case_name for o in read_results(partial)} == set(rows)

        write_results(
            report, name="run-x", pair_key="query_id", directory=tmp_path, run_id="r1"
        )
        assert not partial.exists()

    async def test_two_runs_of_one_name_keep_separate_files(
        self, tmp_path: Path
    ) -> None:
        reports = {}
        for run_id in ("r1", "r2"):
            reports[run_id] = await _dataset().evaluate(
                _task,
                name="run-x",
                max_concurrency=1,
                progress=False,
                lifecycle=case_writer(
                    tmp_path, name="run-x", pair_key="query_id", run_id=run_id
                ),
            )
        cases = {"1_a", "2_b", "3_c", "4_d"}
        assert set(_rows(partial_path(tmp_path, "run-x", "r1"))) == cases
        assert set(_rows(partial_path(tmp_path, "run-x", "r2"))) == cases

        write_results(
            reports["r1"],
            name="run-x",
            pair_key="query_id",
            directory=tmp_path,
            run_id="r1",
        )
        assert not partial_path(tmp_path, "run-x", "r1").exists()
        assert set(_rows(partial_path(tmp_path, "run-x", "r2"))) == cases

    async def test_an_interrupted_case_writes_no_row(self, tmp_path: Path) -> None:
        writer = case_writer(tmp_path, name="run-x", pair_key="query_id", run_id="r1")
        await writer(Case(name="1_a", inputs="x")).teardown(None)
        assert not partial_path(tmp_path, "run-x", "r1").exists()

    async def test_a_case_without_a_verdict_is_unjudged(self, tmp_path: Path) -> None:
        @dataclass
        class MapOnly(Evaluator):
            def evaluate(self, ctx: EvaluatorContext) -> dict[str, float]:
                return {"cited_map": 1.0}

        report = await _report(MapOnly())
        rows = _rows(
            write_results(
                report,
                name="run-x",
                pair_key="query_id",
                directory=tmp_path,
                run_id="r1",
            )
        )
        assert rows["1_a"]["passed"] is None

    async def test_a_result_file_is_never_overwritten(self, tmp_path: Path) -> None:
        report = await _report()
        report.trace_id = None
        first = write_results(
            report, name="run-x", pair_key="query_id", directory=tmp_path, run_id="r1"
        )
        second = write_results(
            report, name="run-x", pair_key="query_id", directory=tmp_path, run_id="r2"
        )
        assert first != second
        with pytest.raises(FileExistsError):
            write_results(
                report,
                name="run-x",
                pair_key="query_id",
                directory=tmp_path,
                run_id="r1",
            )


class TestReadResults:
    async def test_round_trips_to_outcomes(self, tmp_path: Path) -> None:
        report = await _report()
        path = write_results(
            report, name="run-x", pair_key="query_id", directory=tmp_path, run_id="r1"
        )

        outcomes = read_results(path)

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

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "run-x.jsonl"
        row = {
            "case_name": "1_a",
            "key": "a",
            "passed": None,
            "cited": False,
            "cited_map": None,
            "aborted": False,
        }
        path.write_text(json.dumps(row) + "\n\n")
        assert [o.case_name for o in read_results(path)] == ["1_a"]


def _spec() -> DatasetSpec:
    return DatasetSpec(
        key="test",
        db_filename="test.lancedb",
        document_loader=lambda: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        document_mapper=lambda doc: None,
        qa_loader=lambda: [{"q": "q", "a": "42"}],  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        qa_case_builder=lambda idx, doc: Case(
            name=f"case-{idx}",
            inputs=doc["q"],
            expected_output=doc["a"],
            metadata={"question_id": str(idx)},
        ),
        qa_evaluator=NumberMatchEvaluator(),
    )


class TestRunWritesResults:
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
        outcomes = read_results(files[0])
        assert [(o.case_name, o.key, o.passed) for o in outcomes] == [
            ("case-1", "1", True)
        ]

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

    async def test_the_metadata_names_the_pairing_key(self, tmp_path: Path) -> None:
        from evaluations.qa import _prepare_qa_run

        spec = _spec()
        spec.pair_key = "query_id"
        run = await _prepare_qa_run(
            spec, AppConfig(), None, None, tmp_path / "test.lancedb", None, None, None
        )
        assert run.experiment_metadata["pair_key"] == "query_id"

    async def test_a_live_run_writes_no_result_file(self, tmp_path: Path) -> None:
        from evaluations.benchmark import evaluate_dataset

        spec = _spec()
        spec.live = True
        with (
            patch(
                "evaluations.benchmark.run_live_qa_benchmark", new_callable=AsyncMock
            ) as live,
            patch(
                "evaluations.benchmark.run_qa_benchmark", new_callable=AsyncMock
            ) as qa,
        ):
            await evaluate_dataset(
                spec=spec,
                config=AppConfig(),
                skip_db=True,
                skip_retrieval=True,
                skip_qa=False,
                limit=None,
                name=None,
                db_path=None,
                results_dir=tmp_path / "results",
            )
        qa.assert_not_called()
        assert "results_dir" not in live.call_args[1]
