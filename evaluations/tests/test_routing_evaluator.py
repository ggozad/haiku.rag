from unittest.mock import MagicMock

import pytest

from evaluations.evaluators.routing import CollectionRoutingEvaluator

COLLECTIONS = ["a", "b", "c"]


def _ctx(
    searches: list,
    expected: list[str] | None = None,
    searched_uris: list[str] | None = None,
    relevant_uris: list[str] | None = None,
    collections: list[str] | None = COLLECTIONS,
    counts: dict[str, int] | None = None,
) -> MagicMock:
    ctx = MagicMock()
    ctx.metadata = {
        "expected_sources": ["a"] if expected is None else expected,
        "relevant_uris": ["u1", "u2"] if relevant_uris is None else relevant_uris,
    }
    if collections is not None:
        ctx.metadata["collections"] = collections
    ctx.attributes = {
        "search_sources": searches,
        "searched_uris": searched_uris or [],
        **(counts or {}),
    }
    return ctx


class TestCollectionRoutingEvaluator:
    def setup_method(self) -> None:
        self.evaluator = CollectionRoutingEvaluator()

    def test_not_a_routing_case(self) -> None:
        assert self.evaluator.evaluate(_ctx([["a"]], collections=None)) == {}

    def test_an_omitted_selection_is_the_whole_run(self) -> None:
        scores = self.evaluator.evaluate(_ctx([None]))
        assert scores["first_search_covers"] == 1.0
        assert scores["first_search_exact"] == 0.0
        assert scores["sources_recall"] == 1.0
        assert scores["effective_collections"] == 3
        assert scores["n_broad_searches"] == 1

    def test_naming_the_whole_run_is_a_broad_search(self) -> None:
        scores = self.evaluator.evaluate(_ctx([["c", "a", "b"]]))
        assert scores["n_broad_searches"] == 1
        assert scores["effective_collections"] == 3

    def test_an_exact_first_selection(self) -> None:
        scores = self.evaluator.evaluate(_ctx([["a"]]))
        assert scores["first_search_covers"] == 1.0
        assert scores["first_search_exact"] == 1.0
        assert scores["effective_collections"] == 1
        assert scores["n_broad_searches"] == 0

    def test_recall_over_the_union_of_selections(self) -> None:
        scores = self.evaluator.evaluate(_ctx([["b"], ["a"]], expected=["a", "c"]))
        assert scores["first_search_covers"] == 0.0
        assert scores["sources_recall"] == 0.5

    def test_a_name_outside_the_run_searched_nothing(self) -> None:
        scores = self.evaluator.evaluate(_ctx([["zzz"], None]))
        assert scores["n_invalid_selections"] == 1
        assert scores["first_search_covers"] == 0.0
        assert scores["effective_collections"] == 3
        assert scores["n_broadenings"] == 1

    def test_broadening_counts_strictly_wider_follow_ups(self) -> None:
        scores = self.evaluator.evaluate(_ctx([["a"], ["a", "b"], ["b"], None, None]))
        assert scores["n_broadenings"] == 2

    def test_no_search_at_all(self) -> None:
        scores = self.evaluator.evaluate(_ctx([]))
        assert scores["first_search_covers"] == 0.0
        assert scores["first_search_exact"] == 0.0
        assert scores["sources_recall"] == 0.0
        assert scores["effective_collections"] == 0
        assert scores["n_broadenings"] == 0

    def test_searched_recall_over_the_gold_documents(self) -> None:
        scores = self.evaluator.evaluate(
            _ctx([None], searched_uris=["u2", "x"], relevant_uris=["u1", "u2"])
        )
        assert scores["searched_recall"] == 0.5

    def test_no_gold_documents_gives_no_searched_recall(self) -> None:
        scores = self.evaluator.evaluate(_ctx([None], relevant_uris=[]))
        assert "searched_recall" not in scores

    def test_missing_attributes_read_as_no_searches(self) -> None:
        ctx = MagicMock()
        ctx.metadata = {"collections": COLLECTIONS, "expected_sources": ["a"]}
        ctx.attributes = {}
        scores = self.evaluator.evaluate(ctx)
        assert scores["effective_collections"] == 0
        assert scores["sources_recall"] == 0.0


class TestMalformedSelections:
    def setup_method(self) -> None:
        self.evaluator = CollectionRoutingEvaluator()

    @pytest.mark.parametrize("sent", ["a", '["a"]', 3, {"name": "a"}, ["a", 1]])
    def test_a_selection_that_is_not_a_list_of_names_read_nothing(self, sent) -> None:
        scores = self.evaluator.evaluate(_ctx([sent, None]))
        assert scores["n_invalid_selections"] == 1
        assert scores["n_broad_searches"] == 1
        assert scores["effective_collections"] == 3
        assert scores["first_search_covers"] == 0.0
        assert scores["n_broadenings"] == 1


class TestRecoveryAfterAMiss:
    def setup_method(self) -> None:
        self.evaluator = CollectionRoutingEvaluator()

    def test_a_first_search_that_covers_reports_no_recovery(self) -> None:
        assert "recovered_after_miss" not in self.evaluator.evaluate(_ctx([["a"]]))

    def test_a_sideways_recovery_counts(self) -> None:
        """{b} then {a} is a recovery although nothing widened."""
        scores = self.evaluator.evaluate(_ctx([["b"], ["a"]]))
        assert scores["recovered_after_miss"] == 1.0
        assert scores["n_broadenings"] == 0

    def test_a_miss_left_unrecovered(self) -> None:
        scores = self.evaluator.evaluate(_ctx([["b"], ["c"]]))
        assert scores["recovered_after_miss"] == 0.0

    def test_no_search_at_all_is_a_miss(self) -> None:
        assert self.evaluator.evaluate(_ctx([]))["recovered_after_miss"] == 0.0

    def test_an_invalid_first_selection_then_broad_recovers(self) -> None:
        scores = self.evaluator.evaluate(_ctx([["zzz"], None]))
        assert scores["recovered_after_miss"] == 1.0


class TestCostScores:
    def test_the_run_counts_are_mirrored_as_scores(self) -> None:
        counts = {
            "n_sandbox_search_calls": 4,
            "n_search_calls": 2,
            "n_executions": 3,
            "n_requests": 7,
        }
        scores = CollectionRoutingEvaluator().evaluate(_ctx([None], counts=counts))
        assert {name: scores[name] for name in counts} == counts

    def test_absent_counts_are_not_scored(self) -> None:
        scores = CollectionRoutingEvaluator().evaluate(_ctx([None]))
        assert "n_requests" not in scores
