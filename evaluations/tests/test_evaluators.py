from unittest.mock import MagicMock

import pytest

from evaluations.evaluators.map import MAPEvaluator
from evaluations.evaluators.mrr import MRREvaluator


class TestMRREvaluator:
    def setup_method(self) -> None:
        self.evaluator = MRREvaluator()

    def _make_ctx(
        self, relevant_uris: list[str], retrieved_uris: list[str]
    ) -> MagicMock:
        ctx = MagicMock()
        ctx.metadata = {"relevant_uris": relevant_uris}
        ctx.output = retrieved_uris
        return ctx

    def test_first_result_relevant(self) -> None:
        ctx = self._make_ctx(["doc1"], ["doc1", "doc2", "doc3"])
        assert self.evaluator.evaluate(ctx) == 1.0

    def test_second_result_relevant(self) -> None:
        ctx = self._make_ctx(["doc2"], ["doc1", "doc2", "doc3"])
        assert self.evaluator.evaluate(ctx) == 0.5

    def test_third_result_relevant(self) -> None:
        ctx = self._make_ctx(["doc3"], ["doc1", "doc2", "doc3"])
        assert self.evaluator.evaluate(ctx) == pytest.approx(1 / 3)

    def test_no_relevant_found(self) -> None:
        ctx = self._make_ctx(["doc_x"], ["doc1", "doc2", "doc3"])
        assert self.evaluator.evaluate(ctx) == 0.0

    def test_empty_retrieved(self) -> None:
        ctx = self._make_ctx(["doc1"], [])
        assert self.evaluator.evaluate(ctx) == 0.0

    def test_multiple_relevant_returns_first_match(self) -> None:
        ctx = self._make_ctx(["doc2", "doc3"], ["doc1", "doc2", "doc3"])
        assert self.evaluator.evaluate(ctx) == 0.5

    def test_none_metadata(self) -> None:
        ctx = MagicMock()
        ctx.metadata = None
        ctx.output = ["doc1"]
        assert self.evaluator.evaluate(ctx) == 0.0

    def test_empty_relevant_uris(self) -> None:
        ctx = self._make_ctx([], ["doc1", "doc2"])
        assert self.evaluator.evaluate(ctx) == 0.0


class TestMAPEvaluator:
    def setup_method(self) -> None:
        self.evaluator = MAPEvaluator()

    def _make_ctx(
        self, relevant_uris: list[str], retrieved_uris: list[str]
    ) -> MagicMock:
        ctx = MagicMock()
        ctx.metadata = {"relevant_uris": relevant_uris}
        ctx.output = retrieved_uris
        return ctx

    def test_perfect_single_doc(self) -> None:
        ctx = self._make_ctx(["doc1"], ["doc1", "doc2", "doc3"])
        assert self.evaluator.evaluate(ctx) == 1.0

    def test_perfect_two_docs(self) -> None:
        # Both relevant at positions 1 and 2: P@1=1/1, P@2=2/2 → AP = (1+1)/2 = 1.0
        ctx = self._make_ctx(["doc1", "doc2"], ["doc1", "doc2", "doc3"])
        assert self.evaluator.evaluate(ctx) == 1.0

    def test_one_relevant_at_second_position(self) -> None:
        # 1 relevant doc at position 2: P@2=1/2 → AP = 0.5/1 = 0.5
        ctx = self._make_ctx(["doc2"], ["doc1", "doc2", "doc3"])
        assert self.evaluator.evaluate(ctx) == 0.5

    def test_two_relevant_with_gap(self) -> None:
        # Relevant at positions 1 and 3: P@1=1/1, P@3=2/3 → AP = (1 + 2/3)/2
        ctx = self._make_ctx(["doc1", "doc3"], ["doc1", "doc2", "doc3"])
        expected = (1.0 + 2 / 3) / 2
        assert self.evaluator.evaluate(ctx) == pytest.approx(expected)

    def test_no_relevant_found(self) -> None:
        ctx = self._make_ctx(["doc_x"], ["doc1", "doc2", "doc3"])
        assert self.evaluator.evaluate(ctx) == 0.0

    def test_empty_retrieved(self) -> None:
        ctx = self._make_ctx(["doc1"], [])
        assert self.evaluator.evaluate(ctx) == 0.0

    def test_none_metadata(self) -> None:
        ctx = MagicMock()
        ctx.metadata = None
        ctx.output = ["doc1"]
        assert self.evaluator.evaluate(ctx) == 0.0

    def test_empty_relevant_uris(self) -> None:
        ctx = self._make_ctx([], ["doc1", "doc2"])
        assert self.evaluator.evaluate(ctx) == 0.0
