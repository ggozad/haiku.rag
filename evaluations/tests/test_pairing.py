import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evaluations.datasets import DATASETS
from evaluations.pairing import (
    check_pair_rows,
    mcnemar_exact,
    orient,
    pair_outcomes,
    render,
    resolution,
    sign_test,
    summarize,
)
from evaluations.registry import ArmRecord, Registry
from evaluations.traces import CaseOutcome

_ORB_ROW = {
    "query_id": "q_abcdef12",
    "query": "Explain transformers.",
    "answer": "Transformers are...",
    "type": "factual",
    "source": "text",
}
_T2_ROW = {
    "id": "finqa_dev_0",
    "context_id": "finqa_dev_ctx_138",
    "question": "What was the average payment volume per transaction?",
    "program_answer": "127.4",
}
_MTRAG_TASK = {
    "id": "conv1<::>2",
    "turn": "2",
    "answerability": "ANSWERABLE",
    "multi_turn_type": "Follow-up",
    "question_type": ["Factoid"],
    "relevant_uris": ["p1"],
    "turns": [
        {"speaker": "user", "text": "q1"},
        {"speaker": "agent", "text": "a1"},
        {"speaker": "user", "text": "q2"},
    ],
    "answer": "reference answer",
}
_MTRAG_CONVERSATION = {
    "id": "conv1",
    "turns": [
        {
            "task_id": "conv1<::>1",
            "turn": "1",
            "question": "q1",
            "reference": "r1",
            "answerability": "ANSWERABLE",
            "multi_turn_type": "Follow-up",
            "question_type": ["Factoid"],
            "relevant_uris": [],
        }
    ],
}
_ROWS = {
    "frames": {
        "id": "7",
        "Prompt": "Q?",
        "Answer": "A",
        "reasoning_types": "Numerical",
    },
    "hotpotqa": {
        "id": "abc123",
        "question": "What is X?",
        "answer": "X is Y.",
        "type": "comparison",
        "level": "hard",
    },
    "orb_text": _ORB_ROW,
    "orb_multimodal": _ORB_ROW,
    "orb_multimodal_nemotron": _ORB_ROW,
    "t2_finqa": _T2_ROW,
    "t2_tatdqa": _T2_ROW,
    "mtrag_clapnq": _MTRAG_TASK,
    "mtrag_clapnq_rewrite": _MTRAG_TASK,
    "mtrag_clapnq_live": _MTRAG_CONVERSATION,
    "mtrag_clapnq_live_uncompacted": _MTRAG_CONVERSATION,
}


class TestPairKey:
    def test_every_registered_dataset_has_a_sample_row(self) -> None:
        assert set(_ROWS) == set(DATASETS)

    @pytest.mark.parametrize("key", sorted(DATASETS))
    def test_every_case_carries_the_dataset_pair_key(self, key: str) -> None:
        spec = DATASETS[key]
        case = spec.qa_case_builder(1, _ROWS[key])
        assert case.metadata is not None
        assert case.metadata[spec.pair_key]


class TestExactTests:
    def test_mcnemar_known_values(self) -> None:
        assert mcnemar_exact(3, 0) == pytest.approx(0.25)
        assert mcnemar_exact(5, 5) == 1.0
        assert mcnemar_exact(0, 0) == 1.0
        assert mcnemar_exact(22, 6) == pytest.approx(0.00372, rel=0.01)

    def test_sign_test_is_the_same_binomial(self) -> None:
        assert sign_test(22, 6) == mcnemar_exact(22, 6)

    def test_resolution_is_the_smallest_significant_swing(self) -> None:
        assert resolution(28) == 12
        assert resolution(6) == 6
        assert resolution(3) is None
        assert resolution(0) is None


def _o(
    key: str | None,
    passed: bool | None = True,
    cited: bool = True,
    cited_map: float | None = 0.5,
    aborted: bool = False,
) -> CaseOutcome:
    return CaseOutcome(
        case_name=f"n_{key}",
        key=key,
        passed=passed,
        cited=cited,
        cited_map=cited_map,
        aborted=aborted,
    )


class TestSummarize:
    def test_reports_judged_and_floor_rates(self) -> None:
        outcomes = [
            _o("1", passed=True, cited=True, cited_map=1.0),
            _o("2", passed=False, cited=False, cited_map=0.0),
            _o("3", passed=None, cited=False, cited_map=None, aborted=True),
            _o("4", passed=True, cited=True, cited_map=0.5),
        ]
        summary = summarize("arm", outcomes)
        assert summary.cases == 4
        assert summary.judged == 3
        assert summary.accuracy == pytest.approx(2 / 3)
        assert summary.floor == pytest.approx(0.5)
        assert summary.cite_rate == pytest.approx(0.5)
        assert summary.cited_map == pytest.approx(0.5)
        assert summary.aborts == 1
        assert summary.unjudged == 1

    def test_empty_arm_has_no_rates(self) -> None:
        summary = summarize("arm", [])
        assert summary.cases == 0
        assert summary.accuracy is None
        assert summary.cited_map is None


class TestPairOutcomes:
    def _arms(self):
        treated = [
            _o("1", passed=True, cited_map=0.9),
            _o("2", passed=True, cited_map=0.5),
            _o("3", passed=False, cited_map=0.2),
            _o("4", passed=True, cited_map=0.5),
            _o("5", passed=None, cited_map=None, aborted=True),
            _o("6", passed=False, cited_map=0.7),
        ]
        baseline = [
            _o("1", passed=False, cited_map=0.5),
            _o("2", passed=False, cited_map=0.5),
            _o("3", passed=True, cited_map=0.4),
            _o("4", passed=True, cited_map=0.5),
            _o("5", passed=True, cited_map=0.5),
            _o("6", passed=False, cited_map=0.1),
            _o("7", passed=True, cited_map=0.5),
        ]
        return treated, baseline

    def test_joins_on_the_key_and_counts_discordance(self) -> None:
        treated, baseline = self._arms()
        result = pair_outcomes("query_id", "branch", treated, "main", baseline)

        assert result.key == "query_id"
        assert result.paired == 6
        assert result.only_treated == 0
        assert result.only_baseline == 1
        assert result.judged_pairs == 5
        assert (result.b, result.c) == (2, 1)
        assert result.mcnemar_p == pytest.approx(mcnemar_exact(2, 1))
        assert (result.up, result.down, result.ties) == (2, 1, 2)
        assert result.sign_p == pytest.approx(sign_test(2, 1))
        assert result.treated.name == "branch" and result.baseline.name == "main"

    def test_a_null_key_on_either_side_refuses(self) -> None:
        treated, baseline = self._arms()
        baseline[2] = _o(None, passed=True)
        with pytest.raises(ValueError, match="main.*1 case.*query_id"):
            pair_outcomes("query_id", "branch", treated, "main", baseline)

    def test_a_duplicate_key_refuses(self) -> None:
        treated, baseline = self._arms()
        treated.append(_o("1", passed=False))
        with pytest.raises(ValueError, match="branch.*1"):
            pair_outcomes("query_id", "branch", treated, "main", baseline)

    def test_render_carries_the_standard_table(self) -> None:
        treated, baseline = self._arms()
        result = pair_outcomes("query_id", "branch", treated, "main", baseline)
        text = render(result, decision_rule="McNemar exact, p < 0.05 fails")
        assert "paired on query_id" in text
        assert "branch" in text and "main" in text
        for column in (
            "cases",
            "accuracy",
            "floor",
            "cite rate",
            "cited_map",
            "aborts",
        ):
            assert column in text
        assert "McNemar" in text and "sign test" in text
        assert "McNemar exact, p < 0.05 fails" in text
        assert "1 case only in main" in text


def _record(name: str, **overrides: Any) -> ArmRecord:
    fields: dict[str, Any] = {
        "name": name,
        "dataset": "orb_text",
        "kind": "qa",
        "status": "valid",
        "started_at": "2026-01-01T00:00:00+00:00",
        "source": "harness",
        "git_sha": "a" * 40,
        "config_hash": "c" * 64,
        "db_path": "/dbs/orb.lancedb",
        "limit_cases": 100,
        "trace_id": "1" * 32,
    }
    fields.update(overrides)
    return ArmRecord(**fields)


class TestPairRows:
    def test_orientation_comes_from_the_recorded_comparator(self) -> None:
        main = _record("main")
        branch = _record(
            "branch", comparator="main", git_sha="b" * 40, differences='["sha"]'
        )
        assert orient(main, branch) == (branch, main)
        assert orient(branch, main) == (branch, main)

    def test_without_a_recorded_comparator_there_is_no_orientation(self) -> None:
        with pytest.raises(ValueError, match="comparator"):
            orient(_record("a"), _record("b"))

    def test_void_arms_and_different_datasets_are_refused(self) -> None:
        main = _record("main", status="void", void_reason="no telemetry")
        branch = _record("branch", comparator="main", dataset="frames")
        problems = check_pair_rows(branch, main)
        assert any("void" in p and "no telemetry" in p for p in problems)
        assert any("dataset" in p for p in problems)

    def test_a_missing_trace_is_refused(self) -> None:
        main = _record("main", trace_id=None)
        branch = _record("branch", comparator="main", differences="[]")
        assert any("trace" in p for p in check_pair_rows(branch, main))

    def test_named_differences_must_match_the_rows(self) -> None:
        main = _record("main")
        same_sha = _record("branch", comparator="main", differences='["sha"]')
        assert any("sha" in p for p in check_pair_rows(same_sha, main))
        unnamed_sha = _record(
            "branch", comparator="main", git_sha="b" * 40, differences="[]"
        )
        assert any("sha" in p for p in check_pair_rows(unnamed_sha, main))
        unnamed_config = _record(
            "branch", comparator="main", config_hash="d" * 64, differences="[]"
        )
        assert any("config" in p for p in check_pair_rows(unnamed_config, main))
        stale_config = _record(
            "branch", comparator="main", differences='["qa.max_searches"]'
        )
        assert any("qa.max_searches" in p for p in check_pair_rows(stale_config, main))

    def test_a_null_pair_and_a_named_pair_pass(self) -> None:
        main = _record("main")
        null_pair = _record("null", comparator="main", differences="[]")
        assert check_pair_rows(null_pair, main) == []
        treated = _record(
            "branch",
            comparator="main",
            git_sha="b" * 40,
            config_hash="d" * 64,
            limit_cases=50,
            differences='["sha", "qa.max_searches", "limit"]',
        )
        assert check_pair_rows(treated, main) == []


class TestPairCommand:
    def _fake_query(self, by_trace: dict[str, list[dict]]):
        def query(sql: str, *, min_timestamp: str) -> list[dict]:
            trace = next(t for t in by_trace if t in sql)
            rows = by_trace[trace]
            if "count(*)" in sql:
                return [{"n": len(rows)}]
            return rows

        return query

    def _rows(self, verdicts: list[bool]) -> list[dict]:
        return [
            {
                "case_name": f"{i}_{i}",
                "pair_key": str(i),
                "answer_equivalent": "true" if passed else "false",
                "number_match": None,
                "cited_map": 0.5,
                "n_cited": 1,
                "is_exception": False,
            }
            for i, passed in enumerate(verdicts)
        ]

    def test_prints_the_table_for_a_registered_pair(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import evaluations.benchmark as benchmark

        registry_path = tmp_path / "registry.sqlite"
        registry = Registry(registry_path)
        registry.register_launch(_record("main", trace_id="1" * 32))
        registry.register_launch(
            _record(
                "branch",
                comparator="main",
                git_sha="b" * 40,
                differences='["sha"]',
                decision_rule="McNemar exact, two-sided, p < 0.05 fails",
                trace_id="2" * 32,
            )
        )
        monkeypatch.setattr(
            benchmark,
            "query_logfire",
            self._fake_query(
                {
                    "1" * 32: self._rows([True, False, True, False]),
                    "2" * 32: self._rows([True, True, True, False]),
                }
            ),
        )

        result = CliRunner().invoke(
            benchmark.app,
            ["arms", "pair", "main", "branch", "--registry", str(registry_path)],
        )

        assert result.exit_code == 0, result.output
        assert "paired on query_id" in result.output
        assert "branch" in result.output and "main" in result.output
        assert "McNemar" in result.output
        assert "p < 0.05 fails" in result.output

    def test_refuses_a_void_arm_before_fetching(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import evaluations.benchmark as benchmark

        registry_path = tmp_path / "registry.sqlite"
        registry = Registry(registry_path)
        registry.register_launch(
            _record("main", status="void", void_reason="wrong target")
        )
        registry.register_launch(_record("branch", comparator="main", differences="[]"))

        def never(sql: str, *, min_timestamp: str) -> list[dict]:
            raise AssertionError("must not query Logfire for a void pair")

        monkeypatch.setattr(benchmark, "query_logfire", never)
        result = CliRunner().invoke(
            benchmark.app,
            ["arms", "pair", "branch", "main", "--registry", str(registry_path)],
        )
        assert result.exit_code == 1
        assert "void" in result.output and "wrong target" in result.output

    def test_refuses_unknown_arms(self, tmp_path: Path) -> None:
        import evaluations.benchmark as benchmark

        registry_path = tmp_path / "registry.sqlite"
        Registry(registry_path)
        result = CliRunner().invoke(
            benchmark.app, ["arms", "pair", "a", "b", "--registry", str(registry_path)]
        )
        assert result.exit_code == 1
        assert "a" in result.output


def test_differences_survive_a_registry_round_trip(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.register_launch(
        _record("branch", comparator="main", differences='["sha"]')
    )
    record = registry.get("branch")
    assert record is not None
    assert json.loads(record.differences or "null") == ["sha"]
