from unittest.mock import MagicMock

from evaluations.evaluators.citation import (
    CitationMAPEvaluator,
    CitationMRREvaluator,
)


def _ctx(cited: list[str], relevant: list[str]) -> MagicMock:
    ctx = MagicMock()
    ctx.metadata = {"relevant_uris": relevant}
    ctx.attributes = {"cited_uris": cited}
    return ctx


class TestCitationMRREvaluator:
    def setup_method(self) -> None:
        self.evaluator = CitationMRREvaluator()

    def test_first_citation_is_relevant(self) -> None:
        assert self.evaluator.evaluate(_ctx(["a", "b"], ["a"])) == 1.0

    def test_second_citation_is_relevant(self) -> None:
        assert self.evaluator.evaluate(_ctx(["a", "b"], ["b"])) == 0.5

    def test_no_citations(self) -> None:
        assert self.evaluator.evaluate(_ctx([], ["a"])) == 0.0

    def test_no_relevant(self) -> None:
        assert self.evaluator.evaluate(_ctx(["a"], [])) == 0.0

    def test_no_matches(self) -> None:
        assert self.evaluator.evaluate(_ctx(["a", "b"], ["c"])) == 0.0

    def test_metadata_none(self) -> None:
        ctx = MagicMock()
        ctx.metadata = None
        ctx.attributes = {"cited_uris": ["a"]}
        assert self.evaluator.evaluate(ctx) == 0.0

    def test_attribute_missing(self) -> None:
        ctx = MagicMock()
        ctx.metadata = {"relevant_uris": ["a"]}
        ctx.attributes = {}
        assert self.evaluator.evaluate(ctx) == 0.0

    def test_evaluation_name(self) -> None:
        assert self.evaluator.evaluation_name == "cited_mrr"


class TestCitationMAPEvaluator:
    def setup_method(self) -> None:
        self.evaluator = CitationMAPEvaluator()

    def test_all_relevant_first(self) -> None:
        # Both relevant docs cited at ranks 1 and 2: AP = (1/1 + 2/2) / 2 = 1.0
        assert self.evaluator.evaluate(_ctx(["a", "b"], ["a", "b"])) == 1.0

    def test_partial_match(self) -> None:
        # Cited a, x, b. relevant a, b. P@1 = 1/1, P@3 = 2/3. AP = (1 + 2/3)/2
        assert (
            self.evaluator.evaluate(_ctx(["a", "x", "b"], ["a", "b"]))
            == (1.0 + 2 / 3) / 2
        )

    def test_no_matches(self) -> None:
        assert self.evaluator.evaluate(_ctx(["x", "y"], ["a", "b"])) == 0.0

    def test_no_relevant(self) -> None:
        assert self.evaluator.evaluate(_ctx(["a"], [])) == 0.0

    def test_no_citations(self) -> None:
        assert self.evaluator.evaluate(_ctx([], ["a"])) == 0.0

    def test_metadata_none(self) -> None:
        ctx = MagicMock()
        ctx.metadata = None
        ctx.attributes = {"cited_uris": ["a"]}
        assert self.evaluator.evaluate(ctx) == 0.0

    def test_evaluation_name(self) -> None:
        assert self.evaluator.evaluation_name == "cited_map"
