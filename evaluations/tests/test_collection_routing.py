from dataclasses import replace
from unittest.mock import patch

import pytest
from datasets import Dataset

from evaluations.datasets import DATASETS
from evaluations.datasets import collection_routing as routing
from evaluations.datasets.collection_routing import (
    COLLECTION_ROUTING_OPAQUE_SPEC,
    COLLECTION_ROUTING_SPEC,
    CUES,
    OUTSIDE_NAME,
    build_routing_case,
    configure,
    cue_for,
    frames_role,
    load_routing_corpus,
    resolve_names,
    shard_of,
    with_cue,
)
from evaluations.evaluators.routing import CollectionRoutingEvaluator
from haiku.rag.config.models import AppConfig, LanceDBConfig

WIKI = "https://en.wikipedia.org/wiki/"


def _config(**names: str) -> AppConfig:
    """`names` maps a configured name to the database filename it places."""
    return AppConfig(
        lancedb=LanceDBConfig(
            databases={name: f"/dbs/{filename}" for name, filename in names.items()}
        )
    )


def _three() -> AppConfig:
    return _config(
        **{
            "wikipedia-a": "frames_a.lancedb",
            "wikipedia-b": "frames_b.lancedb",
            "papers": "open_rag_bench_multimodal_nemotron.lancedb",
        }
    )


def _article(shard: int) -> str:
    """A Wikipedia uri the hash places in `shard`."""
    for i in range(1000):
        uri = f"{WIKI}Article_{i}"
        if shard_of(uri, 2) == shard:
            return uri
    raise AssertionError("no uri hashed into the shard")


class TestShards:
    def test_the_hash_is_stable_and_in_range(self) -> None:
        assert shard_of("a", 2) == shard_of("a", 2)
        assert {shard_of(f"u{i}", 3) for i in range(60)} == {0, 1, 2}

    def test_frames_role_names_the_shard(self) -> None:
        assert frames_role(_article(0)) == "frames_a"
        assert frames_role(_article(1)) == "frames_b"


class TestResolveNames:
    def test_maps_every_role_to_its_configured_name_in_config_order(self) -> None:
        assert resolve_names(_three()) == {
            "frames_a": "wikipedia-a",
            "frames_b": "wikipedia-b",
            "papers": "papers",
        }

    def test_a_uri_location_resolves_by_its_last_segment(self) -> None:
        config = AppConfig(
            lancedb=LanceDBConfig(
                databases={
                    "a": "s3://bucket/frames_a.lancedb",
                    "b": "s3://bucket/frames_b.lancedb",
                    "c": "s3://bucket/open_rag_bench_multimodal_nemotron.lancedb",
                }
            )
        )
        assert resolve_names(config) == {
            "frames_a": "a",
            "frames_b": "b",
            "papers": "c",
        }

    def test_a_database_with_no_role_refuses(self) -> None:
        config = _config(
            **{
                "wikipedia-a": "frames_a.lancedb",
                "wikipedia-b": "frames_b.lancedb",
                "papers": "open_rag_bench_multimodal_nemotron.lancedb",
                "extra": "hotpotqa.lancedb",
            }
        )
        with pytest.raises(ValueError, match="hotpotqa.lancedb"):
            resolve_names(config)

    def test_a_missing_role_refuses(self) -> None:
        config = _config(**{"wikipedia-a": "frames_a.lancedb"})
        with pytest.raises(ValueError, match="frames_b.lancedb"):
            resolve_names(config)


class TestCues:
    def test_cycle(self) -> None:
        assert [cue_for(i) for i in range(7)] == [*CUES, CUES[0], CUES[1]]

    def test_no_names_leaves_the_question(self) -> None:
        assert with_cue("Q?", []) == "Q?"

    def test_one_name(self) -> None:
        assert with_cue("Q?", ["papers"]) == "In the papers collection: Q?"

    def test_several_names(self) -> None:
        assert (
            with_cue("Q?", ["wikipedia-a", "wikipedia-b", "papers"])
            == "In the wikipedia-a, wikipedia-b and papers collections: Q?"
        )


def _frames(*rows: tuple[str, str, list[str]]) -> Dataset:
    return Dataset.from_list(
        [
            {
                "id": qid,
                "Prompt": question,
                "Answer": f"answer {qid}",
                "wiki_links": str(uris),
                "reasoning_types": "Numerical",
            }
            for qid, question, uris in rows
        ]
    )


def _orb(n: int) -> tuple[Dataset, Dataset]:
    qa = Dataset.from_list(
        [
            {
                "query_id": f"q{i}",
                "query": f"orb question {i}",
                "type": "factual",
                "source": "text",
                "answer": f"orb answer {i}",
            }
            for i in range(n)
        ]
    )
    retrieval = Dataset.from_list(
        [
            {
                "query_id": f"q{i}",
                "query": f"orb question {i}",
                "type": "factual",
                "source": "text",
                "doc_id": f"paper{i}",
            }
            for i in range(n)
        ]
    )
    return qa, retrieval


def _corpus(frames: Dataset, orb: tuple[Dataset, Dataset]) -> list[dict]:
    with (
        patch.object(routing, "load_frames_questions", return_value=frames),
        patch.object(routing, "load_orb_qa", return_value=orb[0]),
        patch.object(routing, "load_orb_retrieval", return_value=orb[1]),
    ):
        return list(load_routing_corpus())


class TestCorpus:
    def test_interleaves_the_members_and_truncates_orb_to_frames(self) -> None:
        frames = _frames(
            ("1", "f1", [_article(0)]),
            ("2", "f2", [_article(0), _article(1)]),
        )
        rows = _corpus(frames, _orb(5))
        assert [row["member"] for row in rows] == ["frames", "orb"] * 2
        assert [row["question_id"] for row in rows] == [
            "frames:1",
            "orb:q0",
            "frames:2",
            "orb:q1",
        ]

    def test_frames_rows_carry_the_shards_holding_their_articles(self) -> None:
        both = [_article(1), _article(0), _article(1)]
        rows = _corpus(_frames(("1", "f1", both)), _orb(1))
        assert rows[0]["expected_roles"] == ["frames_a", "frames_b"]
        assert rows[0]["relevant_uris"] == [_article(1), _article(0)]
        assert rows[0]["answer"] == "answer 1"

    def test_orb_rows_join_the_answer_to_the_gold_document(self) -> None:
        rows = _corpus(_frames(("1", "f1", [_article(0)])), _orb(1))
        assert rows[1] == {
            "member": "orb",
            "question_id": "orb:q0",
            "question": "orb question 0",
            "answer": "orb answer 0",
            "relevant_uris": ["paper0"],
            "expected_roles": ["papers"],
            "cue": "named",
        }

    def test_cues_follow_the_row_index(self) -> None:
        frames = _frames(*((str(i), f"f{i}", [_article(0)]) for i in range(5)))
        rows = _corpus(frames, _orb(5))
        assert [row["cue"] for row in rows] == [cue_for(i) for i in range(10)]

    def test_orb_queries_without_a_gold_document_are_dropped(self) -> None:
        qa, retrieval = _orb(3)
        retrieval = retrieval.select([0, 2])
        frames = _frames(("1", "f1", [_article(0)]), ("2", "f2", [_article(0)]))
        rows = _corpus(frames, (qa, retrieval))
        assert [row["question_id"] for row in rows if row["member"] == "orb"] == [
            "orb:q0",
            "orb:q2",
        ]


def _row(**overrides) -> dict:
    row = {
        "member": "frames",
        "question_id": "frames:7",
        "question": "Q?",
        "answer": "A",
        "relevant_uris": [_article(0)],
        "expected_roles": ["frames_a"],
        "cue": "none",
    }
    return {**row, **overrides}


NAMES = {"frames_a": "wikipedia-a", "frames_b": "wikipedia-b", "papers": "papers"}


class TestBuildRoutingCase:
    def test_metadata_carries_what_the_routing_scores_need(self) -> None:
        case = build_routing_case(3, _row(), names=NAMES)
        assert case.name == "3_none_frames_7"
        assert case.inputs == "Q?"
        assert case.expected_output == "A"
        assert case.metadata == {
            "question_id": "frames:7",
            "member": "frames",
            "cue": "none",
            "cued_sources": [],
            "expected_sources": ["wikipedia-a"],
            "collections": ["wikipedia-a", "wikipedia-b", "papers"],
            "relevant_uris": [_article(0)],
            "case_index": "3",
        }

    def test_a_named_cue_names_every_expected_collection(self) -> None:
        row = _row(cue="named", expected_roles=["frames_a", "frames_b"])
        case = build_routing_case(1, row, names=NAMES)
        assert case.metadata is not None
        assert case.inputs == "In the wikipedia-a and wikipedia-b collections: Q?"
        assert case.metadata["cued_sources"] == ["wikipedia-a", "wikipedia-b"]
        assert case.metadata["expected_sources"] == ["wikipedia-a", "wikipedia-b"]

    def test_a_misnamed_cue_names_a_covered_collection_outside_the_expected(
        self,
    ) -> None:
        row = _row(cue="misnamed", expected_roles=["frames_a", "frames_b"])
        case = build_routing_case(1, row, names=NAMES)
        assert case.metadata is not None
        assert case.inputs == "In the papers collection: Q?"
        assert case.metadata["cued_sources"] == ["papers"]

    def test_a_misnamed_cue_is_stable_for_a_question(self) -> None:
        row = _row(cue="misnamed", member="orb", expected_roles=["papers"])
        first = build_routing_case(1, row, names=NAMES).inputs
        assert first == build_routing_case(9, row, names=NAMES).inputs
        assert first in {
            "In the wikipedia-a collection: Q?",
            "In the wikipedia-b collection: Q?",
        }

    def test_an_outside_cue_names_a_collection_the_run_lacks(self) -> None:
        case = build_routing_case(1, _row(cue="outside"), names=NAMES)
        assert case.metadata is not None
        assert case.inputs == f"In the {OUTSIDE_NAME} collection: Q?"
        assert case.metadata["cued_sources"] == [OUTSIDE_NAME]
        assert OUTSIDE_NAME not in case.metadata["collections"]

    def test_unconfigured_the_roles_stand_in_for_the_names(self) -> None:
        case = build_routing_case(1, _row(cue="named"))
        assert case.metadata is not None
        assert case.inputs == "In the frames_a collection: Q?"
        assert case.metadata["collections"] == ["frames_a", "frames_b", "papers"]


class TestSpecs:
    def test_configure_binds_the_configured_names(self) -> None:
        bound = configure(COLLECTION_ROUTING_SPEC, _three())
        case = bound.qa_case_builder(1, _row(cue="named"))
        assert case.inputs == "In the wikipedia-a collection: Q?"
        assert bound.key == COLLECTION_ROUTING_SPEC.key

    def test_configure_refuses_a_set_that_is_not_the_three_collections(self) -> None:
        with pytest.raises(ValueError):
            configure(COLLECTION_ROUTING_SPEC, AppConfig())

    @pytest.mark.parametrize(
        "spec", [COLLECTION_ROUTING_SPEC, COLLECTION_ROUTING_OPAQUE_SPEC]
    )
    def test_the_specs_score_routing_and_citations(self, spec) -> None:
        assert spec.configure is configure
        assert any(
            isinstance(e, CollectionRoutingEvaluator) for e in spec.case_evaluators
        )
        assert spec.citation_evaluator is not None
        assert spec.retrieval_loader is None
        assert spec.pair_key == "question_id"
        assert DATASETS[spec.key] is spec

    def test_the_corpus_is_never_populated_from_the_spec(self) -> None:
        with pytest.raises(RuntimeError, match="evaluations split"):
            COLLECTION_ROUTING_SPEC.document_loader()
        assert COLLECTION_ROUTING_SPEC.document_mapper({"uri": "x"}) is None

    def test_the_two_keys_differ_only_in_how_the_collections_are_named(self) -> None:
        assert COLLECTION_ROUTING_SPEC.experiment_metadata == {
            "collection_names": "descriptive"
        }
        assert COLLECTION_ROUTING_OPAQUE_SPEC.experiment_metadata == {
            "collection_names": "opaque"
        }
        assert (
            replace(
                COLLECTION_ROUTING_OPAQUE_SPEC,
                key=COLLECTION_ROUTING_SPEC.key,
                experiment_metadata=COLLECTION_ROUTING_SPEC.experiment_metadata,
            )
            == COLLECTION_ROUTING_SPEC
        )
