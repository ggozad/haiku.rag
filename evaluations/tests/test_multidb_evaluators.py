from types import SimpleNamespace

from evaluations.evaluators.multidb import (
    AttributionGate,
    NumericAnswer,
    ScopeGate,
    TextAnswer,
    numbers_in,
)


def ctx(output="", metadata=None, attributes=None):
    return SimpleNamespace(
        output=output, metadata=metadata or {}, attributes=attributes or {}
    )


def test_numbers_survive_comma_separators_and_units():
    assert numbers_in("Station Kestrel sits at 1,240 metres") == {1240.0}
    assert numbers_in("1240 m") == {1240.0}
    assert numbers_in("no digits here") == set()


def test_gold_number_counts_as_correct():
    result = NumericAnswer().evaluate(
        ctx("It sits at 1240 metres.", {"expected_value": 1240})
    )
    assert result == {"answer_correct": True}


def test_hedging_between_gold_and_distractor_fails():
    """The failure the near-name pair exists to catch: a presence check would
    pass this, since the gold value is in the answer."""
    result = NumericAnswer().evaluate(
        ctx(
            "Station Kestrel sits at either 1240 m or 2310 m.",
            {"expected_value": 1240, "distractor_value": 2310},
        )
    )
    assert result == {"answer_correct": False}


def test_numeric_answer_abstains_without_a_gold_value():
    assert NumericAnswer().evaluate(ctx("anything", {})) == {}


def test_attribution_requires_the_exact_database_set():
    good = AttributionGate().evaluate(
        ctx(
            metadata={"expected_sources": ["northern"]},
            attributes={"cited_sources": ["northern"]},
        )
    )
    assert good == {"attribution_correct": True}
    wrong = AttributionGate().evaluate(
        ctx(
            metadata={"expected_sources": ["northern"]},
            attributes={"cited_sources": ["southern"]},
        )
    )
    assert wrong == {"attribution_correct": False}
    extra = AttributionGate().evaluate(
        ctx(
            metadata={"expected_sources": ["northern"]},
            attributes={"cited_sources": ["northern", "equipment"]},
        )
    )
    assert extra == {"attribution_correct": False}


def test_scope_allows_a_subset_and_rejects_an_outsider():
    """Citing fewer databases than allowed honours scope; citing one outside it
    does not."""
    inside = ScopeGate().evaluate(
        ctx(
            metadata={"scope": ["northern", "southern"]},
            attributes={"cited_sources": ["northern"]},
        )
    )
    assert inside == {"scope_honoured": True}
    outside = ScopeGate().evaluate(
        ctx(
            metadata={"scope": ["northern"]},
            attributes={"cited_sources": ["northern", "southern"]},
        )
    )
    assert outside == {"scope_honoured": False}


def test_empty_scope_is_honoured_only_by_citing_nothing():
    assert ScopeGate().evaluate(
        ctx(metadata={"scope": []}, attributes={"cited_sources": []})
    ) == {"scope_honoured": True}
    assert ScopeGate().evaluate(
        ctx(metadata={"scope": []}, attributes={"cited_sources": ["northern"]})
    ) == {"scope_honoured": False}


def test_ordered_text_must_appear_in_order():
    meta = {"expected_ordered": ["Overview", "Instruments", "Measurements"]}
    assert TextAnswer().evaluate(
        ctx("Overview, then Instruments, then Measurements", meta)
    ) == {"answer_correct": True}
    assert TextAnswer().evaluate(
        ctx("Measurements, then Overview, then Instruments", meta)
    ) == {"answer_correct": False}


def test_forbidden_text_fails_even_when_required_text_is_present():
    result = TextAnswer().evaluate(
        ctx(
            "station://northern/kestrel and station://southern/kestrel-ridge",
            {
                "expected_ordered": ["station://northern/kestrel"],
                "forbidden_text": ["station://southern/kestrel-ridge"],
            },
        )
    )
    assert result == {"answer_correct": False}
