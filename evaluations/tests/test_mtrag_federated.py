import os
import subprocess
import sys

import pytest

from evaluations.datasets import DATASETS
from evaluations.datasets.mtrag_federated import (
    DEFAULT_BUDGET,
    GOLD_TITLE_FLOOR,
    MTRAG_FEDERATED_SPEC,
    collection_names,
    collection_of,
    partition_records,
    pool_composition,
    sample_records,
)


def record(passage_id: str, title: str) -> dict[str, str]:
    return {"_id": passage_id, "title": title, "text": f"text of {passage_id}"}


def corpus(titles: dict[str, int]) -> list[dict[str, str]]:
    """One record per passage, `titles` mapping a title to its passage count."""
    return [
        record(f"{title}_{index}", title)
        for title, count in titles.items()
        for index in range(count)
    ]


class TestCollectionOf:
    def test_assigns_within_range(self) -> None:
        for n in (2, 4, 8):
            assigned = {collection_of(f"title {i}", n) for i in range(200)}
            assert assigned <= set(range(n))

    def test_uses_every_collection(self) -> None:
        """A partition that leaves a collection empty is not a partition."""
        for n in (2, 4, 8):
            assigned = {collection_of(f"title {i}", n) for i in range(200)}
            assert assigned == set(range(n))

    def test_is_stable_across_processes(self) -> None:
        """Salted `hash()` would make a build unreproducible between runs.

        The partition is never stored, so scoring recomputes it in a different
        process than the one that ingested.
        """
        code = (
            "from evaluations.datasets.mtrag_federated import collection_of;"
            "print([collection_of(f'title {i}', 8) for i in range(12)])"
        )
        runs = {
            subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                check=True,
                env={**os.environ, "PYTHONHASHSEED": seed},
            ).stdout.strip()
            for seed in ("0", "1", "12345")
        }
        assert len(runs) == 1, f"assignment varies with PYTHONHASHSEED: {runs}"

    def test_seed_changes_the_assignment(self) -> None:
        titles = [f"title {i}" for i in range(200)]
        one = [collection_of(t, 8, seed=1) for t in titles]
        two = [collection_of(t, 8, seed=2) for t in titles]
        assert one != two

    def test_rejects_a_collection_count_below_one(self) -> None:
        with pytest.raises(ValueError, match="at least one collection"):
            collection_of("title", 0)


class TestCollectionNames:
    def test_names_one_per_collection(self) -> None:
        assert collection_names(3) == (
            "clapnq_0",
            "clapnq_1",
            "clapnq_2",
        )

    def test_declaration_order_is_the_name_order(self) -> None:
        """Fusion resolves ties to configured order, so the order is load-bearing."""
        names = collection_names(4)
        assert list(names) == sorted(names, key=lambda name: int(name.split("_")[1]))


class TestPartitionRecords:
    def test_keeps_every_record(self) -> None:
        records = corpus({"a": 3, "b": 2, "c": 4})
        grouped = partition_records(records, 2)
        assert sum(len(rows) for rows in grouped.values()) == len(records)

    def test_never_splits_a_title(self) -> None:
        """A title is the atom: its passages must share a collection, or a
        query's gold spreads for reasons the partition never intended."""
        records = corpus({f"title {i}": 5 for i in range(40)})
        grouped = partition_records(records, 4)
        holders: dict[str, set[str]] = {}
        for name, rows in grouped.items():
            for row in rows:
                holders.setdefault(row["title"], set()).add(name)
        split = {title: names for title, names in holders.items() if len(names) > 1}
        assert not split, f"titles split across collections: {split}"

    def test_names_every_collection_even_when_one_is_empty(self) -> None:
        """The config declares n databases, so the build must create n."""
        records = corpus({"only": 2})
        grouped = partition_records(records, 4)
        assert set(grouped) == set(collection_names(4))


class TestSampleRecords:
    def test_keeps_every_gold_passage(self) -> None:
        records = corpus({f"title {i}": 10 for i in range(50)})
        gold = {"title 3_1", "title 17_4", "title 42_9"}
        sampled = sample_records(records, gold, budget=60)
        assert gold <= {row["_id"] for row in sampled}

    def test_keeps_whole_titles_holding_gold(self) -> None:
        records = corpus({f"title {i}": 10 for i in range(50)})
        sampled = sample_records(records, {"title 3_1"}, budget=0)
        assert sorted(row["_id"] for row in sampled) == sorted(
            f"title 3_{i}" for i in range(10)
        )

    def test_respects_the_budget(self) -> None:
        records = corpus({f"title {i}": 10 for i in range(50)})
        sampled = sample_records(records, {"title 3_1"}, budget=100)
        assert len(sampled) <= 100

    def test_budget_below_the_gold_floor_still_keeps_gold(self) -> None:
        records = corpus({f"title {i}": 10 for i in range(50)})
        gold = {f"title {i}_0" for i in range(20)}
        sampled = sample_records(records, gold, budget=5)
        assert len(sampled) == 200
        assert gold <= {row["_id"] for row in sampled}

    def test_is_stable_for_a_seed(self) -> None:
        records = corpus({f"title {i}": 10 for i in range(50)})
        first = sample_records(records, {"title 0_0"}, budget=100, seed=7)
        second = sample_records(records, {"title 0_0"}, budget=100, seed=7)
        assert [row["_id"] for row in first] == [row["_id"] for row in second]

    def test_seed_changes_the_distractors(self) -> None:
        records = corpus({f"title {i}": 10 for i in range(50)})
        first = sample_records(records, {"title 0_0"}, budget=100, seed=7)
        second = sample_records(records, {"title 0_0"}, budget=100, seed=8)
        assert {row["_id"] for row in first} != {row["_id"] for row in second}

    def test_rejects_gold_the_corpus_does_not_hold(self) -> None:
        records = corpus({"a": 2})
        with pytest.raises(ValueError, match="do not resolve"):
            sample_records(records, {"missing"}, budget=10)


class TestSpec:
    def test_registers_under_its_key(self) -> None:
        assert DATASETS[MTRAG_FEDERATED_SPEC.key] is MTRAG_FEDERATED_SPEC

    def test_opts_out_of_the_shared_population(self) -> None:
        """The databases are built by build_databases, not populate_db."""
        with pytest.raises(RuntimeError, match="build_databases"):
            MTRAG_FEDERATED_SPEC.document_loader()

    def test_retrieval_limit_matches_the_product_default(self) -> None:
        """5 is config's search.limit, the setting the depth quota bites at."""
        assert MTRAG_FEDERATED_SPEC.retrieval_limit == 5

    def test_scores_retrieval_without_a_judge(self) -> None:
        assert MTRAG_FEDERATED_SPEC.retrieval_evaluators
        assert MTRAG_FEDERATED_SPEC.retrieval_loader is not None
        assert MTRAG_FEDERATED_SPEC.retrieval_mapper is not None


class TestPoolComposition:
    def test_separates_gold_bearing_titles_from_distractors(self) -> None:
        records = corpus({"answers": 4, "filler": 6})
        gold_side, distractors = pool_composition(records, {"answers_2"})
        assert (gold_side, distractors) == (4, 6)

    def test_reports_no_distractors_when_the_budget_is_at_the_floor(self) -> None:
        """A pool of only answer-bearing articles scores as an easy task and
        says nothing, so the build has to be able to see it."""
        records = corpus({f"title {i}": 10 for i in range(5)})
        gold = {f"title {i}_0" for i in range(5)}
        sampled = sample_records(records, gold, budget=1)
        assert pool_composition(sampled, gold) == (50, 0)

    def test_the_default_budget_clears_the_gold_floor(self) -> None:
        assert DEFAULT_BUDGET > GOLD_TITLE_FLOOR
