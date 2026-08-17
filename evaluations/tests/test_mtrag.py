import pytest

from evaluations.config import ConversationInput
from evaluations.datasets import DATASETS
from evaluations.datasets.mtrag import (
    MTRAG_CLAPNQ_LIVE_SPEC,
    MTRAG_CLAPNQ_REWRITE_SPEC,
    MTRAG_CLAPNQ_SPEC,
    _group_conversations,
    _join_queries_qrels,
    _parse_qrels,
    _task_to_record,
    _validate_qrels_resolve,
    build_mtrag_case,
    build_mtrag_live_case,
    map_mtrag_document,
    map_mtrag_retrieval,
)
from evaluations.evaluators import (
    CitationMAPEvaluator,
    MAPEvaluator,
    NDCGEvaluator,
    RecallEvaluator,
)

GENERATION_TASK = {
    "task_id": "conv1<::>2",
    "conversation_id": "conv1",
    "turn": "2",
    "Collection": "mt-rag-clapnq-elser-512-100-20240503",
    "Answerability": ["ANSWERABLE"],
    "Multi-Turn": ["Follow-up"],
    "Question Type": ["Factoid"],
    "input": [
        {"speaker": "user", "text": "q1", "metadata": {}},
        {"speaker": "agent", "text": "a1", "metadata": {}},
        {"speaker": "user", "text": "q2", "metadata": {}},
    ],
    "targets": [{"text": "reference answer"}],
    "contexts": [{"document_id": "retrieved-not-gold"}],
}


class TestDocumentMapper:
    def test_maps_passage_to_payload(self) -> None:
        payload = map_mtrag_document(
            {"_id": "837799097_6931-7548-0-617", "title": "T", "text": "body"}
        )
        assert payload.uri == "837799097_6931-7548-0-617"
        assert payload.title == "T"
        assert payload.content == "body"


class TestQrels:
    QRELS_TSV = (
        "query-id\tcorpus-id\tscore\n"
        "conv1<::>2\tdoc1_0-10-0-10\t1\n"
        "conv1<::>2\tdoc2_5-20-0-15\t1\n"
        "conv2<::>1\tdoc3_0-9-0-9\t1\n"
    )

    def test_parse_groups_by_query_preserving_order(self) -> None:
        qrels = _parse_qrels(self.QRELS_TSV.splitlines())
        assert qrels == {
            "conv1<::>2": ["doc1_0-10-0-10", "doc2_5-20-0-15"],
            "conv2<::>1": ["doc3_0-9-0-9"],
        }

    def test_join_builds_records(self) -> None:
        qrels = _parse_qrels(self.QRELS_TSV.splitlines())
        queries = [
            {"_id": "conv1<::>2", "text": "q one"},
            {"_id": "conv2<::>1", "text": "q two"},
        ]
        records = _join_queries_qrels(queries, qrels)
        assert records == [
            {
                "query_id": "conv1<::>2",
                "question": "q one",
                "expected_uris": ["doc1_0-10-0-10", "doc2_5-20-0-15"],
            },
            {
                "query_id": "conv2<::>1",
                "question": "q two",
                "expected_uris": ["doc3_0-9-0-9"],
            },
        ]

    def test_join_raises_on_query_without_qrels(self) -> None:
        with pytest.raises(ValueError, match="no qrels"):
            _join_queries_qrels([{"_id": "missing<::>1", "text": "q"}], {})

    def test_validation_passes_when_all_resolve(self) -> None:
        qrels = {"q1": ["a", "b"]}
        _validate_qrels_resolve({"a", "b", "c"}, qrels)

    def test_validation_raises_on_unresolved_id(self) -> None:
        qrels = {"q1": ["a", "ghost"]}
        with pytest.raises(ValueError, match="ghost"):
            _validate_qrels_resolve({"a"}, qrels)


class TestRetrievalMapper:
    def test_maps_joined_record(self) -> None:
        sample = map_mtrag_retrieval(
            {
                "query_id": "conv1<::>2",
                "question": "who?",
                "expected_uris": ["u1", "u2"],
            }
        )
        assert sample is not None
        assert sample.question == "who?"
        assert sample.expected_uris == ("u1", "u2")


class TestSpecs:
    def test_registered(self) -> None:
        assert DATASETS["mtrag_clapnq"] is MTRAG_CLAPNQ_SPEC
        assert DATASETS["mtrag_clapnq_rewrite"] is MTRAG_CLAPNQ_REWRITE_SPEC

    def test_variants_share_db(self) -> None:
        assert MTRAG_CLAPNQ_SPEC.db_filename == MTRAG_CLAPNQ_REWRITE_SPEC.db_filename

    def test_retrieval_configuration(self) -> None:
        for spec in (MTRAG_CLAPNQ_SPEC, MTRAG_CLAPNQ_REWRITE_SPEC):
            assert spec.retrieval_limit == 10
            assert spec.ingest_batch_size == 512
            assert spec.retrieval_evaluators is not None
            kinds = {
                (type(e), getattr(e, "k", None)) for e in spec.retrieval_evaluators
            }
            assert kinds == {
                (RecallEvaluator, 5),
                (RecallEvaluator, 10),
                (NDCGEvaluator, 5),
                (NDCGEvaluator, 10),
                (MAPEvaluator, None),
            }
            assert isinstance(spec.citation_evaluator, CitationMAPEvaluator)


class TestGenerationTasks:
    def test_task_to_record(self) -> None:
        record = _task_to_record(GENERATION_TASK, {"conv1<::>2": ["p1", "p2"]})
        assert record == {
            "id": "conv1<::>2",
            "turn": "2",
            "turns": [
                {"speaker": "user", "text": "q1"},
                {"speaker": "agent", "text": "a1"},
                {"speaker": "user", "text": "q2"},
            ],
            "answer": "reference answer",
            "answerability": "ANSWERABLE",
            "multi_turn_type": "Follow-up",
            "question_type": ["Factoid"],
            "relevant_uris": ["p1", "p2"],
        }

    def test_task_without_qrels_has_no_relevant_uris(self) -> None:
        record = _task_to_record(GENERATION_TASK, {})
        assert record is not None
        assert record["relevant_uris"] is None

    def test_other_collections_excluded(self) -> None:
        task = {**GENERATION_TASK, "Collection": "mt-rag-govt-elser-512-100-20240611"}
        assert _task_to_record(task, {}) is None

    def test_build_case_conversation_and_metadata(self) -> None:
        record = _task_to_record(GENERATION_TASK, {"conv1<::>2": ["p1"]})
        assert record is not None
        case = build_mtrag_case(3, record)

        assert isinstance(case.inputs, ConversationInput)
        assert case.inputs.question == "q2"
        assert [t.speaker for t in case.inputs.turns] == ["user", "agent", "user"]
        assert case.expected_output == "reference answer"
        assert case.metadata == {
            "task_id": "conv1<::>2",
            "turn": "2",
            "answerability": "ANSWERABLE",
            "multi_turn_type": "Follow-up",
            "question_type": ["Factoid"],
            "relevant_uris": ["p1"],
        }

    def test_build_case_omits_relevant_uris_when_absent(self) -> None:
        record = _task_to_record(
            {**GENERATION_TASK, "Answerability": ["UNANSWERABLE"]}, {}
        )
        assert record is not None
        case = build_mtrag_case(1, record)

        assert case.metadata is not None
        assert "relevant_uris" not in case.metadata
        assert case.metadata["answerability"] == "UNANSWERABLE"


class TestLiveConversations:
    def _records(self) -> list[dict]:
        turn1 = _task_to_record(
            {
                **GENERATION_TASK,
                "task_id": "conv1<::>1",
                "turn": "1",
                "input": [{"speaker": "user", "text": "q1", "metadata": {}}],
                "targets": [{"text": "r1"}],
            },
            {"conv1<::>1": ["p1"]},
        )
        turn2 = _task_to_record(GENERATION_TASK, {"conv1<::>2": ["p2", "p3"]})
        other = _task_to_record(
            {
                **GENERATION_TASK,
                "task_id": "conv2<::>1",
                "turn": "1",
                "input": [{"speaker": "user", "text": "other q", "metadata": {}}],
                "targets": [{"text": "other r"}],
                "Answerability": ["UNANSWERABLE"],
            },
            {},
        )
        assert turn1 and turn2 and other
        # turn 2 first: grouping must sort turns numerically within a conversation
        return [turn2, turn1, other]

    def test_grouping_sorts_turns_within_conversations(self) -> None:
        conversations = _group_conversations(self._records())
        assert [c["id"] for c in conversations] == ["conv1", "conv2"]
        conv1 = conversations[0]
        assert [t["question"] for t in conv1["turns"]] == ["q1", "q2"]
        assert [t["reference"] for t in conv1["turns"]] == ["r1", "reference answer"]
        assert conv1["turns"][1]["relevant_uris"] == ["p2", "p3"]

    def test_build_live_case(self) -> None:
        conversations = _group_conversations(self._records())
        case = build_mtrag_live_case(1, conversations[0])

        assert case.inputs == ["q1", "q2"]
        assert case.metadata is not None
        assert case.metadata["conversation_id"] == "conv1"
        turns = case.metadata["turns"]
        assert turns[0] == {
            "task_id": "conv1<::>1",
            "turn": "1",
            "reference": "r1",
            "answerability": "ANSWERABLE",
            "multi_turn_type": "Follow-up",
            "question_type": ["Factoid"],
            "relevant_uris": ["p1"],
        }
        other_case = build_mtrag_live_case(2, conversations[1])
        assert other_case.metadata is not None
        assert other_case.metadata["turns"][0]["relevant_uris"] == []

    def test_live_spec(self) -> None:
        assert DATASETS["mtrag_clapnq_live"] is MTRAG_CLAPNQ_LIVE_SPEC
        assert MTRAG_CLAPNQ_LIVE_SPEC.db_filename == MTRAG_CLAPNQ_SPEC.db_filename
        assert MTRAG_CLAPNQ_LIVE_SPEC.live is True
        assert MTRAG_CLAPNQ_LIVE_SPEC.retrieval_loader is None
        assert MTRAG_CLAPNQ_LIVE_SPEC.experiment_metadata == {
            "mtrag_mode": "live_session",
            "compaction": True,
        }
        assert MTRAG_CLAPNQ_SPEC.experiment_metadata == {"mtrag_mode": "gold_prefix"}

    def test_live_compaction_arms(self) -> None:
        assert MTRAG_CLAPNQ_LIVE_SPEC.compaction is True
        uncompacted = DATASETS["mtrag_clapnq_live_uncompacted"]
        assert uncompacted.compaction is False
        assert uncompacted.live is True
        assert uncompacted.db_filename == MTRAG_CLAPNQ_LIVE_SPEC.db_filename
        assert uncompacted.qa_case_builder is MTRAG_CLAPNQ_LIVE_SPEC.qa_case_builder
        assert uncompacted.experiment_metadata == {
            "mtrag_mode": "live_session",
            "compaction": False,
        }
