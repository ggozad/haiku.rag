import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evaluations.datasets import DATASETS
from evaluations.pairing import (
    mcnemar_exact,
    pair_outcomes,
    render,
    resolution,
    sign_test,
    summarize,
)
from evaluations.results import CaseOutcome

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
        assert summary.cite_rate_all_cases == pytest.approx(0.5)
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
        result = pair_outcomes("branch", treated, "main", baseline)

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
        with pytest.raises(ValueError, match="main: 1 case.*no pairing key"):
            pair_outcomes("branch", treated, "main", baseline)

    def test_a_duplicate_key_refuses(self) -> None:
        treated, baseline = self._arms()
        treated.append(_o("1", passed=False))
        with pytest.raises(ValueError, match="branch.*1"):
            pair_outcomes("branch", treated, "main", baseline)

    def test_render_carries_the_standard_table(self) -> None:
        treated, baseline = self._arms()
        result = pair_outcomes("branch", treated, "main", baseline)
        text = render(result)
        assert "6 cases on both sides" in text
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
        assert "1 case only in main" in text
        assert "no split reaches p < 0.05" in text

    def test_render_warns_when_only_one_arm_was_gated(self) -> None:
        treated, baseline = self._arms()
        treated[0] = CaseOutcome(
            case_name="n_1",
            key="1",
            passed=True,
            cited=True,
            cited_map=0.9,
            aborted=False,
            judge_decided_by="system_one",
        )
        result = pair_outcomes("branch", treated, "main", baseline)

        assert result.treated.gated and not result.baseline.gated
        assert (
            "warning: branch was judged through system_one and main was not; "
            "verdict differences include judge differences" in render(result)
        )

    def test_render_warns_when_the_arms_were_gated_by_different_models(
        self,
    ) -> None:
        def gated(key: str, model: str) -> CaseOutcome:
            return CaseOutcome(
                case_name=f"n_{key}",
                key=key,
                passed=True,
                cited=True,
                cited_map=0.5,
                aborted=False,
                judge_decided_by="system_one",
                system_one_model=model,
            )

        result = pair_outcomes(
            "branch", [gated("1", "jev-1.13.0")], "main", [gated("1", "decider-4b-v1")]
        )

        assert (
            "warning: branch was judged through system_one model jev-1.13.0 and "
            "main through decider-4b-v1" in render(result)
        )

    def test_render_does_not_warn_when_both_arms_share_a_judge(self) -> None:
        treated, baseline = self._arms()
        assert "warning" not in render(
            pair_outcomes("branch", treated, "main", baseline)
        )

    def test_render_states_the_resolution_and_one_sided_cases(self) -> None:
        treated = [_o(str(i), passed=True) for i in range(8)] + [_o("x")]
        baseline = [_o(str(i), passed=False) for i in range(8)]
        text = render(pair_outcomes("branch", treated, "main", baseline))
        assert "1 case only in branch" in text
        assert "rejects at |b - c| >= 8 (100.0% of 8 paired cases)" in text


def _write(path: Path, passed: list[bool]) -> Path:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "case_name": f"{i}_q{i}",
                    "key": f"q{i}",
                    "passed": verdict,
                    "cited": True,
                    "cited_map": 0.5,
                    "aborted": False,
                }
            )
            + "\n"
            for i, verdict in enumerate(passed)
        )
    )
    return path


class TestReadResults:
    def test_a_file_from_before_the_gated_judge_reads_as_ungated(
        self, tmp_path: Path
    ) -> None:
        from evaluations.results import read_results

        (outcome,) = read_results(_write(tmp_path / "old.jsonl", [True]))
        assert outcome.judge_decided_by is None

    def test_the_decider_is_read_back(self, tmp_path: Path) -> None:
        from evaluations.results import read_results

        path = tmp_path / "gated.jsonl"
        row = {
            "case_name": "0_q0",
            "key": "q0",
            "passed": True,
            "cited": True,
            "cited_map": 0.5,
            "aborted": False,
            "judge_decided_by": "fallback",
        }
        path.write_text(json.dumps(row) + "\n")
        (outcome,) = read_results(path)
        assert outcome.judge_decided_by == "fallback"


class TestPairCommand:
    def test_prints_the_table_for_two_result_files(self, tmp_path: Path) -> None:
        from evaluations.benchmark import app

        main = _write(tmp_path / "main.jsonl", [True, False, True, False])
        branch = _write(tmp_path / "branch.jsonl", [True, True, True, False])

        result = CliRunner().invoke(app, ["pair", str(branch), str(main)])

        assert result.exit_code == 0, result.output
        assert "4 cases on both sides" in result.output
        assert "branch pass / main fail 1" in result.output

    def test_files_with_no_case_in_common_are_refused(self, tmp_path: Path) -> None:
        from evaluations.benchmark import app

        a = _write(tmp_path / "a.jsonl", [True])
        b = tmp_path / "b.jsonl"
        b.write_text(a.read_text().replace('"q0"', '"other"'))

        result = CliRunner().invoke(app, ["pair", str(a), str(b)])

        assert result.exit_code == 1
        assert "no case in common" in result.output

    def test_an_unpairable_file_is_named(self, tmp_path: Path) -> None:
        from evaluations.benchmark import app

        a = _write(tmp_path / "a.jsonl", [True, True])
        b = tmp_path / "b.jsonl"
        b.write_text(a.read_text().replace('"q1"', '"q0"'))

        result = CliRunner().invoke(app, ["pair", str(a), str(b)])

        assert result.exit_code == 1
        assert "b: 1 duplicate" in result.output
