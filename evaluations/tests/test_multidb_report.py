from types import SimpleNamespace

from evaluations.datasets.multidb_report import (
    attribution_gate,
    family_rates,
    render,
    scope_gate,
    surface_coverage,
)


def case(
    name="0_x",
    *,
    family=None,
    scope=None,
    expected_sources=None,
    scope_honoured=None,
    attribution_correct=None,
    answer_correct=None,
    code=None,
):
    assertions = {}
    if scope_honoured is not None:
        assertions["scope_honoured"] = SimpleNamespace(value=scope_honoured)
    if attribution_correct is not None:
        assertions["attribution_correct"] = SimpleNamespace(value=attribution_correct)
    scores = {}
    if answer_correct is not None:
        scores["answer_correct"] = SimpleNamespace(value=answer_correct)
    metadata = {}
    if family is not None:
        metadata["family"] = family
    if scope is not None:
        metadata["scope"] = scope
    if expected_sources is not None:
        metadata["expected_sources"] = expected_sources
    return SimpleNamespace(
        name=name,
        assertions=assertions,
        scores=scores,
        metadata=metadata,
        attributes={"executed_code": code} if code else {},
    )


def test_scope_gate_counts_only_scoped_cases():
    cases = [
        case("0_b1", family="B1"),
        case("1_b4", family="B4", scope=["northern"], scope_honoured=True),
        case("2_b4", family="B4", scope=["northern"], scope_honoured=False),
    ]
    gate = scope_gate(cases)
    assert gate.checked == 2
    assert gate.failures == ["2_b4"]
    assert gate.passed is False


def test_scope_gate_passes_when_nothing_leaks():
    gate = scope_gate([case("0_b4", family="B4", scope=[], scope_honoured=True)])
    assert gate.passed
    assert "PASS" in gate.line()


def test_attribution_gate_ignores_families_with_more_than_one_answer():
    """B5 is answered by two databases, so it is not an attribution error to cite
    both and the gate must not police it."""
    cases = [
        case(
            "0_b5",
            family="B5",
            expected_sources=["northern", "southern"],
            attribution_correct=False,
        ),
        case(
            "1_b1", family="B1", expected_sources=["northern"], attribution_correct=True
        ),
    ]
    gate = attribution_gate(cases)
    assert gate.checked == 1
    assert gate.passed


def test_attribution_gate_reports_the_offending_case():
    gate = attribution_gate(
        [
            case(
                "3_b3",
                family="B3",
                expected_sources=["northern"],
                attribution_correct=False,
            )
        ]
    )
    assert gate.failures == ["3_b3"]
    assert "FAIL (1)" in gate.line()
    assert "3_b3" in gate.line()


def test_family_rates_skip_cases_with_no_answer_score():
    """The refusal families have no numeric expectation, so they must not appear
    as 0/0 and drag a family rate down."""
    cases = [
        case("0_b1", family="B1", answer_correct=1.0),
        case("1_b1", family="B1", answer_correct=0.0),
        case("2_b6", family="B6"),
    ]
    rates = family_rates(cases)
    assert rates == {"B1": (1, 2)}
    assert "B6" not in rates


def test_surface_coverage_counts_cases_not_snippets():
    cases = [
        case("0", code=["print(toc)", "open('/documents/1/toc.json')"]),
        case("1", code=["open('/documents/2/content.txt')"]),
        case("2"),
    ]
    coverage = surface_coverage(cases)
    assert coverage["toc.json"] == 1
    assert coverage["content.txt"] == 1
    assert coverage["items.jsonl"] == 0


def test_render_says_passed_only_when_both_gates_pass():
    good = [
        case(
            "0_b4",
            family="B4",
            scope=["northern"],
            scope_honoured=True,
            expected_sources=["northern"],
            attribution_correct=True,
            answer_correct=1.0,
        ),
    ]
    text, passed = render(good)
    assert passed
    assert "GATES: PASSED" in text

    bad = good + [case("1_b4", family="B4", scope=["northern"], scope_honoured=False)]
    text, passed = render(bad)
    assert not passed
    assert "GATES: FAILED" in text
    assert "1_b4" in text


def test_render_flags_a_surface_nothing_reached():
    text, _ = render([case("0", family="S5", code=["open('/documents/1/toc.json')"])])
    assert "never reached" in text
    assert "toc.json" in text
