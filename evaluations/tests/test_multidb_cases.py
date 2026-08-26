from collections import Counter

from evaluations.config import ScopedQuestion
from evaluations.datasets.multidb import (
    DATABASE_NAMES,
    NEAR_NAME_PAIRS,
    behaviour_rows,
    build_multidb_case,
    surface_rows,
)


def by_family(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["family"], []).append(row)
    return grouped


def test_gate_families_carry_enough_instances():
    """B3 and B4 are pass/fail gates, so a handful of cases is not evidence of
    absence. Everything else is a rate and three is enough."""
    families = by_family(behaviour_rows())
    assert len(families["B3"]) >= 8
    assert len(families["B4"]) >= 8


def test_half_the_scoped_cases_exclude_the_better_match():
    """If the scoped-in database always holds the most relevant content, the
    model answers correctly without honouring scope and no-leakage is trivially
    satisfied. Half the B4 instances ask about a near-name pair member, whose
    twin is the excluded strong match."""
    b4 = by_family(behaviour_rows())["B4"]
    pair_cases = [r for r in b4 if r["id"].startswith("b4-pair-")]
    assert len(pair_cases) >= len(b4) / 2
    twins = {name for pair in NEAR_NAME_PAIRS for name in pair}
    for row in pair_cases:
        assert any(f"Station {name}" in row["question"] for name in twins)


def test_single_source_families_expect_exactly_one_database():
    for row in behaviour_rows():
        if row["family"] in {"B1", "B3", "B4"}:
            assert len(row["expected_sources"]) == 1


def test_shared_entity_family_expects_both_databases():
    for row in by_family(behaviour_rows())["B5"]:
        assert set(row["expected_sources"]) == {"northern", "southern"}


def test_cross_database_family_rotates_database_order():
    """RRF ties resolve to the order the databases are listed, so a fixed order
    would make this family measure ordering rather than fusion."""
    orders = [tuple(row["sources"]) for row in by_family(behaviour_rows())["B2"]]
    assert len(set(orders)) == len(orders)
    for order in orders:
        assert set(order) == set(DATABASE_NAMES)


def test_refusal_families_are_labelled_unanswerable():
    """The label is what makes RefusalJudge score them and what feeds refusal
    precision and recall."""
    families = by_family(behaviour_rows())
    for family in ("B6", "B7"):
        assert all(r["answerability"] == "UNANSWERABLE" for r in families[family])
    answerable = [
        r for f, rows in families.items() if f not in {"B6", "B7"} for r in rows
    ]
    assert all(r["answerability"] == "ANSWERABLE" for r in answerable)


def test_empty_scope_reaches_the_case_as_an_empty_list():
    """`sources=[]` covers nothing and must not collapse into None, which covers
    everything."""
    rows = [r for r in behaviour_rows() if r["family"] == "B7"]
    assert rows
    for index, row in enumerate(rows):
        case = build_multidb_case(index, row)
        assert isinstance(case.inputs, ScopedQuestion)
        assert case.inputs.sources == []
        assert case.metadata is not None
        assert case.metadata["scope"] == []


def test_unscoped_cases_pass_a_bare_question():
    row = next(r for r in behaviour_rows() if r["family"] == "B1")
    case = build_multidb_case(0, row)
    assert isinstance(case.inputs, str)


def test_no_case_asks_for_both_a_number_and_ordered_text():
    """The two scorers share the answer_correct key, and the composite raises
    rather than letting one silently win."""
    for row in (*behaviour_rows(), *surface_rows()):
        numeric = (
            row["expected_value"] is not None or row["expected_values"] is not None
        )
        assert not (numeric and row["expected_ordered"] is not None)


def test_case_ids_are_unique_and_stable():
    rows = (*behaviour_rows(), *surface_rows())
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids)
    assert Counter(r["family"] for r in surface_rows()).keys() >= {
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
    }
