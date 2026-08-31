import os
import subprocess
import sys

import pytest

from evaluations.datasets import DATASETS
from evaluations.datasets.mtrag_federated import (
    DEFAULT_BUDGET,
    DOMAINS,
    FTSIndexNotCoveringRows,
    GOLD_TITLE_FLOOR,
    MTRAG_FEDERATED_SPEC,
    assert_fts_covers_rows,
    collection_names,
    collection_of,
    domain_files,
    partition_pooled,
    pooled_collection_names,
    pooled_database_paths,
    partition_records,
    pool_composition,
    sample_records,
)


def _smoke_config():
    """A config placing two databases, so the run resolves a federated client."""
    from haiku.rag.config.models import AppConfig

    return AppConfig.model_validate(
        {
            "lancedb": {
                "databases": {
                    "clapnq_0": "/tmp/a.lancedb",
                    "clapnq_1": "/tmp/b.lancedb",
                }
            }
        }
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


class TestRetrievalLimitOverride:
    async def test_override_replaces_the_spec_value(self, monkeypatch) -> None:
        """Fetch depth is a run knob: hybrid search degenerates below roughly 50
        candidates, so every regime would otherwise need its own dataset."""
        seen: list[int | None] = []

        async def fake_search(self, query, limit=None, **kwargs):  # noqa: ANN001
            seen.append(limit)
            return []

        from haiku.rag.client import HaikuRAG

        monkeypatch.setattr(HaikuRAG, "search", fake_search)
        from evaluations.retrieval import run_retrieval_benchmark

        await run_retrieval_benchmark(
            MTRAG_FEDERATED_SPEC,
            _smoke_config(),
            limit=1,
            retrieval_limit=77,
        )
        assert seen and set(seen) == {77}

    async def test_spec_value_is_the_default(self, monkeypatch) -> None:
        seen: list[int | None] = []

        async def fake_search(self, query, limit=None, **kwargs):  # noqa: ANN001
            seen.append(limit)
            return []

        from haiku.rag.client import HaikuRAG

        monkeypatch.setattr(HaikuRAG, "search", fake_search)
        from evaluations.retrieval import run_retrieval_benchmark

        await run_retrieval_benchmark(MTRAG_FEDERATED_SPEC, _smoke_config(), limit=1)
        assert seen and set(seen) == {MTRAG_FEDERATED_SPEC.retrieval_limit}


class TestFTSCoverageAssertion:
    """The chunks FTS index is built once over zero rows and only an optimize
    folds later rows in, so a build that skips it ships dead full-text search
    that still returns results."""

    class _Index:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Stats:
        def __init__(self, indexed: int) -> None:
            self.num_indexed_rows = indexed

    class _Table:
        def __init__(self, rows: int, indexed: int | None) -> None:
            self._rows = rows
            self._indexed = indexed

        async def count_rows(self) -> int:
            return self._rows

        async def list_indices(self):
            if self._indexed is None:
                return []
            return [TestFTSCoverageAssertion._Index("content_fts_idx")]

        async def index_stats(self, name: str):
            assert name == "content_fts_idx"
            return TestFTSCoverageAssertion._Stats(self._indexed or 0)

    async def test_passes_when_the_index_covers_every_row(self) -> None:
        await assert_fts_covers_rows(self._Table(100, 100), "clapnq_0")

    async def test_rejects_a_zero_row_index(self) -> None:
        with pytest.raises(FTSIndexNotCoveringRows, match="covers 0 of 100"):
            await assert_fts_covers_rows(self._Table(100, 0), "clapnq_0")

    async def test_rejects_a_partially_covering_index(self) -> None:
        with pytest.raises(FTSIndexNotCoveringRows, match="covers 60 of 100"):
            await assert_fts_covers_rows(self._Table(100, 60), "clapnq_0")

    async def test_rejects_a_missing_index(self) -> None:
        with pytest.raises(FTSIndexNotCoveringRows, match="no content_fts_idx"):
            await assert_fts_covers_rows(self._Table(100, None), "clapnq_0")


class TestDomains:
    def test_names_the_four_upstream_domains(self) -> None:
        assert DOMAINS == ("clapnq", "cloud", "fiqa", "govt")

    def test_paths_follow_the_upstream_layout(self) -> None:
        assert domain_files("govt") == (
            "corpora/passage_level/govt.jsonl.zip",
            "mtrag-human/retrieval_tasks/govt/qrels/dev.tsv",
            "mtrag-human/retrieval_tasks/govt/govt_lastturn.jsonl",
        )


class TestDomainPartition:
    """With four real domains, alpha finally means something: 0 keeps a
    collection to one topic, 1 shards titles across all of them."""

    def test_alpha_zero_keeps_a_domain_together_when_n_matches(self) -> None:
        for domain_index, domain in enumerate(DOMAINS):
            assigned = {
                collection_of(f"{domain} title {i}", 4, alpha=0.0, domain=domain)
                for i in range(50)
            }
            assert assigned == {domain_index}

    def test_alpha_zero_subdivides_within_a_domain_when_n_exceeds_it(self) -> None:
        for domain_index, domain in enumerate(DOMAINS):
            assigned = {
                collection_of(f"{domain} title {i}", 8, alpha=0.0, domain=domain)
                for i in range(200)
            }
            assert assigned == {domain_index * 2, domain_index * 2 + 1}

    def test_alpha_zero_groups_domains_when_n_is_below_it(self) -> None:
        assigned = {
            (domain, collection_of(f"t{i}", 2, alpha=0.0, domain=domain))
            for domain in DOMAINS
            for i in range(20)
        }
        by_collection: dict[int, set[str]] = {}
        for domain, collection in assigned:
            by_collection.setdefault(collection, set()).add(domain)
        assert set(by_collection) == {0, 1}
        assert all(len(v) == 2 for v in by_collection.values())

    def test_alpha_one_ignores_the_domain(self) -> None:
        """The shard endpoint: a title's collection must not depend on its domain."""
        titles = [f"title {i}" for i in range(200)]
        as_clapnq = [collection_of(t, 8, alpha=1.0, domain="clapnq") for t in titles]
        as_govt = [collection_of(t, 8, alpha=1.0, domain="govt") for t in titles]
        assert as_clapnq == as_govt

    def test_alpha_one_spreads_a_single_domain_across_every_collection(self) -> None:
        assigned = {
            collection_of(f"title {i}", 8, alpha=1.0, domain="clapnq")
            for i in range(400)
        }
        assert assigned == set(range(8))

    def test_intermediate_alpha_moves_some_titles_off_their_domain(self) -> None:
        titles = [f"title {i}" for i in range(400)]
        home = [collection_of(t, 4, alpha=0.0, domain="fiqa") for t in titles]
        mixed = [collection_of(t, 4, alpha=0.5, domain="fiqa") for t in titles]
        moved = sum(1 for a, b in zip(home, mixed) if a != b)
        assert 0 < moved < len(titles), f"alpha=0.5 moved {moved} of {len(titles)}"

    def test_default_alpha_is_the_domain_partition(self) -> None:
        for domain in DOMAINS:
            assert collection_of("t", 4, domain=domain) == collection_of(
                "t", 4, alpha=0.0, domain=domain
            )


class TestPooledPartition:
    def test_names_are_distinct_from_the_single_domain_set(self) -> None:
        """The two datasets must never share database paths."""
        assert not set(pooled_collection_names(4)) & set(collection_names(4))

    def test_database_paths_separate_alpha_and_n(self) -> None:
        a = pooled_database_paths(4, 0.0)
        b = pooled_database_paths(4, 1.0)
        c = pooled_database_paths(8, 0.0)
        assert not set(a.values()) & set(b.values())
        assert not set(a.values()) & set(c.values())

    def test_routes_each_record_by_its_own_domain(self) -> None:
        records = [
            {
                "_id": f"{domain}-{i}",
                "title": f"{domain} t{i}",
                "text": "x",
                "domain": domain,
            }
            for domain in DOMAINS
            for i in range(20)
        ]
        grouped = partition_pooled(records, 4, alpha=0.0)
        for name, rows in grouped.items():
            domains = {row["domain"] for row in rows}
            assert len(domains) == 1, f"{name} mixes domains at alpha=0: {domains}"

    def test_alpha_one_mixes_domains_in_every_collection(self) -> None:
        records = [
            {"_id": f"{domain}-{i}", "title": f"t{i}", "text": "x", "domain": domain}
            for domain in DOMAINS
            for i in range(60)
        ]
        grouped = partition_pooled(records, 4, alpha=1.0)
        assert all(len({r["domain"] for r in rows}) > 1 for rows in grouped.values())

    def test_keeps_every_record(self) -> None:
        records = [
            {"_id": f"{d}-{i}", "title": f"{d} t{i}", "text": "x", "domain": d}
            for d in DOMAINS
            for i in range(15)
        ]
        for alpha in (0.0, 0.5, 1.0):
            grouped = partition_pooled(records, 8, alpha=alpha)
            assert sum(len(v) for v in grouped.values()) == len(records)
