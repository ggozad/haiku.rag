import re

import pytest

from evaluations.traces import case_outcomes


def _row(i: int, **overrides) -> dict:
    row = {
        "case_name": f"{i}_{i:04d}",
        "pair_key": f"{i:04d}",
        "answer_equivalent": "true" if i % 3 else "false",
        "number_match": None,
        "cited_map": 0.5 if i % 5 else None,
        "n_cited": 2 if i % 7 else 0,
        "is_exception": i % 50 == 0,
    }
    row.update(overrides)
    return row


def _fake_query(rows: list[dict], reported_count: int | None = None):
    calls: list[str] = []

    def query(sql: str, *, min_timestamp: str) -> list[dict]:
        calls.append(sql)
        if "count(*)" in sql:
            return [{"n": len(rows) if reported_count is None else reported_count}]
        match = re.search(r"LIMIT (\d+) OFFSET (\d+)", sql)
        assert match is not None
        limit, offset = int(match[1]), int(match[2])
        return rows[offset : offset + limit]

    return query, calls


class TestCaseOutcomes:
    def test_pages_past_the_row_cap_and_parses_every_field(self) -> None:
        rows = [_row(i) for i in range(1, 251)]
        query, calls = _fake_query(rows)

        outcomes = case_outcomes(
            "0" * 32, "query_id", query=query, min_timestamp="2026-01-01T00:00:00Z"
        )

        assert len(outcomes) == 250
        assert [o.case_name for o in outcomes] == [r["case_name"] for r in rows]
        assert sum("OFFSET" in call for call in calls) == 3
        first, third, fifth, fiftieth = (
            outcomes[0],
            outcomes[2],
            outcomes[4],
            outcomes[49],
        )
        assert first.key == "0001" and first.passed is True and first.cited is True
        assert first.cited_map == 0.5 and first.aborted is False
        assert third.passed is False
        assert fifth.cited_map is None
        assert fiftieth.aborted is True
        assert outcomes[6].cited is False

    def test_reads_the_key_as_text_and_scopes_to_case_spans(self) -> None:
        query, calls = _fake_query([_row(1)])
        case_outcomes("a" * 32, "query_id", query=query, min_timestamp="t")
        page = next(call for call in calls if "OFFSET" in call)
        assert "attributes->'metadata'->>'query_id'" in page
        assert "span_name = 'case: {case_name}'" in page
        assert f"trace_id = '{'a' * 32}'" in page

    def test_number_match_scores_pass_at_one(self) -> None:
        rows = [
            _row(1, answer_equivalent=None, number_match=1.0),
            _row(2, answer_equivalent=None, number_match=0.0),
            _row(3, answer_equivalent=None, number_match=None),
        ]
        query, _ = _fake_query(rows)
        outcomes = case_outcomes("0" * 32, "id", query=query, min_timestamp="t")
        assert [o.passed for o in outcomes] == [True, False, None]

    def test_an_absent_key_is_none_not_a_string(self) -> None:
        query, _ = _fake_query([_row(1, pair_key=None)])
        outcomes = case_outcomes(
            "0" * 32, "question_id", query=query, min_timestamp="t"
        )
        assert outcomes[0].key is None

    def test_a_count_mismatch_raises(self) -> None:
        query, _ = _fake_query([_row(i) for i in range(1, 11)], reported_count=11)
        with pytest.raises(ValueError, match="11.*10|10.*11"):
            case_outcomes("0" * 32, "id", query=query, min_timestamp="t")

    def test_rejects_a_malformed_trace_id_or_key(self) -> None:
        query, _ = _fake_query([])
        with pytest.raises(ValueError, match="trace id"):
            case_outcomes("not-a-trace", "id", query=query, min_timestamp="t")
        with pytest.raises(ValueError, match="key"):
            case_outcomes("0" * 32, "id; drop", query=query, min_timestamp="t")
