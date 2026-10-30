from unittest.mock import MagicMock

import pytest

from evaluations.evaluators.citation import CitationMAPEvaluator


def _ctx(cited: list[str], metadata: dict | None) -> MagicMock:
    ctx = MagicMock()
    ctx.metadata = metadata
    ctx.attributes = {"cited_uris": cited}
    return ctx


class TestCitationMAPEvaluator:
    @pytest.mark.parametrize(
        ("cited", "expected"),
        [
            # P@1 = 1/1, P@3 = 2/3, AP = (1 + 2/3)/2
            (["a", "x", "b"], (1.0 + 2 / 3) / 2),
            ([], 0.0),
        ],
        ids=["cited-uris", "no-citations"],
    )
    def test_scores_the_cited_uris(self, cited: list[str], expected: float) -> None:
        ctx = _ctx(cited, {"relevant_uris": ["a", "b"]})
        assert CitationMAPEvaluator().evaluate(ctx) == expected

    @pytest.mark.parametrize(
        "metadata",
        [{"relevant_uris": []}, {"answerability": "UNANSWERABLE"}, None],
        ids=["no-relevant-uris", "relevant-uris-missing", "metadata-none"],
    )
    def test_a_case_without_gold_passages_is_ineligible(
        self, metadata: dict | None
    ) -> None:
        """Turns without gold passages produce no score at all, not a zero."""
        assert CitationMAPEvaluator().evaluate(_ctx(["a"], metadata)) == {}

    def test_evaluation_name(self) -> None:
        assert CitationMAPEvaluator().get_default_evaluation_name() == "cited_map"
