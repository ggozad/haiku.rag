import json
import zipfile
from collections.abc import Iterable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

import httpx
from datasets import Dataset
from pydantic_evals import Case

from evaluations.config import (
    ConversationInput,
    DatasetSpec,
    DocumentPayload,
    RetrievalSample,
    Turn,
)
from evaluations.evaluators import (
    CitationMAPEvaluator,
    MAPEvaluator,
    NDCGEvaluator,
    RecallEvaluator,
)

REPO_SHA = "cc5b1d481b391181b89f7ced860308482e785463"
_BASE_URL = f"https://raw.githubusercontent.com/IBM/mt-rag-benchmark/{REPO_SHA}"

_CORPUS_FILE = "corpora/passage_level/clapnq.jsonl.zip"
_QRELS_FILE = "mtrag-human/retrieval_tasks/clapnq/qrels/dev.tsv"
_QUERY_FILES = {
    "lastturn": "mtrag-human/retrieval_tasks/clapnq/clapnq_lastturn.jsonl",
    "rewrite": "mtrag-human/retrieval_tasks/clapnq/clapnq_rewrite.jsonl",
}
_GEN_TASKS_FILE = "mtrag-human/generation_tasks/reference.jsonl"
_CLAPNQ_COLLECTION = "mt-rag-clapnq-elser-512-100-20240503"


def get_cache_dir() -> Path:
    cache_dir = Path.home() / ".cache" / "haiku.rag" / "evaluations" / "mtrag"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _download(rel_path: str) -> Path:
    dest = get_cache_dir() / rel_path.replace("/", "_")
    if dest.exists():
        return dest

    with httpx.stream(
        "GET", f"{_BASE_URL}/{rel_path}", timeout=120.0, follow_redirects=True
    ) as response:
        response.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as fh:
            for data in response.iter_bytes():
                fh.write(data)
        tmp.rename(dest)
    return dest


def _parse_qrels(lines: Iterable[str]) -> dict[str, list[str]]:
    """Group qrel corpus-ids by query-id, preserving file order."""
    qrels: dict[str, list[str]] = {}
    rows = iter(lines)
    next(rows)  # header: query-id / corpus-id / score
    for line in rows:
        if not line.strip():
            continue
        query_id, corpus_id, _score = line.rstrip("\n").split("\t")
        qrels.setdefault(query_id, []).append(corpus_id)
    return qrels


def _validate_qrels_resolve(
    corpus_ids: set[str], qrels: Mapping[str, list[str]]
) -> None:
    unresolved = sorted(
        {cid for ids in qrels.values() for cid in ids if cid not in corpus_ids}
    )
    if unresolved:
        raise ValueError(
            f"{len(unresolved)} qrel corpus-ids do not resolve to corpus "
            f"passages, e.g. {unresolved[:3]}"
        )


def _join_queries_qrels(
    queries: Iterable[Mapping[str, Any]], qrels: Mapping[str, list[str]]
) -> list[dict[str, Any]]:
    records = []
    for query in queries:
        query_id = query["_id"]
        expected = qrels.get(query_id)
        if expected is None:
            raise ValueError(f"query {query_id} has no qrels")
        records.append(
            {
                "query_id": query_id,
                "question": query["text"],
                "expected_uris": expected,
            }
        )
    return records


def _load_qrels() -> dict[str, list[str]]:
    path = _download(_QRELS_FILE)
    return _parse_qrels(path.read_text().splitlines())


def load_clapnq_corpus() -> Dataset:
    path = _download(_CORPUS_FILE)
    records: list[dict[str, str]] = []
    with zipfile.ZipFile(path) as zf:
        with zf.open(zf.namelist()[0]) as fh:
            for line in fh:
                rec = json.loads(line)
                records.append(
                    {"_id": rec["_id"], "title": rec["title"], "text": rec["text"]}
                )
    _validate_qrels_resolve({rec["_id"] for rec in records}, _load_qrels())
    return Dataset.from_list(records)


def map_mtrag_document(doc: Mapping[str, Any]) -> DocumentPayload:
    return DocumentPayload(uri=doc["_id"], content=doc["text"], title=doc["title"])


def load_clapnq_retrieval(variant: str) -> Dataset:
    path = _download(_QUERY_FILES[variant])
    queries = [json.loads(line) for line in path.read_text().splitlines() if line]
    return Dataset.from_list(_join_queries_qrels(queries, _load_qrels()))


def map_mtrag_retrieval(doc: Mapping[str, Any]) -> RetrievalSample | None:
    return RetrievalSample(
        question=doc["question"],
        expected_uris=tuple(doc["expected_uris"]),
    )


def _task_to_record(
    task: Mapping[str, Any], qrels: Mapping[str, list[str]]
) -> dict[str, Any] | None:
    """Reduce a reference.jsonl generation task to the fields QA cases need.

    Task `contexts` are the original system's retrievals, never gold relevance;
    gold passages come from the qrels keyed by task_id.
    """
    if task["Collection"] != _CLAPNQ_COLLECTION:
        return None
    return {
        "id": task["task_id"],
        "turn": task["turn"],
        "turns": [
            {"speaker": message["speaker"], "text": message["text"]}
            for message in task["input"]
        ],
        "answer": task["targets"][0]["text"],
        "answerability": task["Answerability"][0],
        "multi_turn_type": task["Multi-Turn"][0],
        "question_type": list(task["Question Type"]),
        "relevant_uris": qrels.get(task["task_id"]),
    }


def _qa_records() -> list[dict[str, Any]]:
    path = _download(_GEN_TASKS_FILE)
    qrels = _load_qrels()
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = _task_to_record(json.loads(line), qrels)
        if record is not None:
            records.append(record)
    return records


def load_clapnq_qa() -> Dataset:
    return Dataset.from_list(_qa_records())


def build_mtrag_case(
    index: int, doc: Mapping[str, Any]
) -> Case[ConversationInput, str, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "task_id": doc["id"],
        "turn": doc["turn"],
        "answerability": doc["answerability"],
        "multi_turn_type": doc["multi_turn_type"],
        "question_type": list(doc["question_type"]),
    }
    if doc["relevant_uris"]:
        metadata["relevant_uris"] = list(doc["relevant_uris"])
    return Case(
        name=f"{index}_{doc['id']}",
        inputs=ConversationInput(
            turns=[Turn(**turn) for turn in doc["turns"]],
        ),
        expected_output=doc["answer"],
        metadata=metadata,
    )


def _group_conversations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group per-turn generation records into full conversations.

    Turns are ordered numerically within each conversation; each turn carries
    its user question, reference answer, answerability label, and gold
    passages when the turn has qrels.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        conversation_id = record["id"].split("<::>")[0]
        grouped.setdefault(conversation_id, []).append(record)

    conversations = []
    for conversation_id, tasks in grouped.items():
        tasks.sort(key=lambda record: int(record["turn"]))
        turns = []
        for task in tasks:
            turn: dict[str, Any] = {
                "task_id": task["id"],
                "turn": task["turn"],
                "question": task["turns"][-1]["text"],
                "reference": task["answer"],
                "answerability": task["answerability"],
                "multi_turn_type": task["multi_turn_type"],
                "question_type": list(task["question_type"]),
                "relevant_uris": list(task["relevant_uris"] or []),
            }
            turns.append(turn)
        conversations.append({"id": conversation_id, "turns": turns})
    return conversations


def load_clapnq_conversations() -> Dataset:
    return Dataset.from_list(_group_conversations(_qa_records()))


def build_mtrag_live_case(
    index: int, doc: Mapping[str, Any]
) -> Case[list[str], list[str], dict[str, Any]]:
    questions = [turn["question"] for turn in doc["turns"]]
    metadata_turns = [
        {key: value for key, value in turn.items() if key != "question"}
        for turn in doc["turns"]
    ]
    return Case(
        name=f"{index}_{doc['id']}",
        inputs=questions,
        metadata={"conversation_id": doc["id"], "turns": metadata_turns},
    )


def _mtrag_spec(key: str, variant: str) -> DatasetSpec:
    return DatasetSpec(
        key=key,
        db_filename="mtrag_clapnq.lancedb",
        document_loader=load_clapnq_corpus,
        document_mapper=map_mtrag_document,
        qa_loader=load_clapnq_qa,
        qa_case_builder=build_mtrag_case,
        retrieval_loader=partial(load_clapnq_retrieval, variant),
        retrieval_mapper=map_mtrag_retrieval,
        retrieval_evaluators=[
            RecallEvaluator(k=5),
            RecallEvaluator(k=10),
            NDCGEvaluator(k=5),
            NDCGEvaluator(k=10),
            MAPEvaluator(),
        ],
        citation_evaluator=CitationMAPEvaluator(),
        retrieval_limit=10,
        ingest_batch_size=512,
        experiment_metadata={"mtrag_mode": "gold_prefix"},
    )


MTRAG_CLAPNQ_SPEC = _mtrag_spec("mtrag_clapnq", "lastturn")
MTRAG_CLAPNQ_REWRITE_SPEC = _mtrag_spec("mtrag_clapnq_rewrite", "rewrite")


def _mtrag_live_spec(key: str, compaction: bool) -> DatasetSpec:
    return DatasetSpec(
        key=key,
        db_filename="mtrag_clapnq.lancedb",
        document_loader=load_clapnq_corpus,
        document_mapper=map_mtrag_document,
        qa_loader=load_clapnq_conversations,
        qa_case_builder=build_mtrag_live_case,
        ingest_batch_size=512,
        live=True,
        compaction=compaction,
        experiment_metadata={"mtrag_mode": "live_session", "compaction": compaction},
    )


MTRAG_CLAPNQ_LIVE_SPEC = _mtrag_live_spec("mtrag_clapnq_live", compaction=True)
MTRAG_CLAPNQ_LIVE_UNCOMPACTED_SPEC = _mtrag_live_spec(
    "mtrag_clapnq_live_uncompacted", compaction=False
)
