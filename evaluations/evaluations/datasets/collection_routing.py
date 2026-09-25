import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

from datasets import Dataset
from pydantic_evals import Case

from evaluations.config import DatasetSpec
from evaluations.datasets.frames import load_frames_questions, question_expected_uris
from evaluations.datasets.open_rag_bench import load_orb_qa, load_orb_retrieval
from evaluations.evaluators import CitationMAPEvaluator, CollectionRoutingEvaluator
from haiku.rag.config.models import AppConfig

FRAMES_SHARDS = ("frames_a", "frames_b")
ROLE_BY_FILENAME = {
    "frames_a.lancedb": "frames_a",
    "frames_b.lancedb": "frames_b",
    "open_rag_bench_multimodal_nemotron.lancedb": "papers",
}
CUES = ("none", "named", "none", "misnamed", "outside")
OUTSIDE_NAME = "archive"


def shard_of(key: str, shards: int) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % shards


def frames_role(uri: str) -> str:
    return FRAMES_SHARDS[shard_of(uri, len(FRAMES_SHARDS))]


def resolve_names(config: AppConfig) -> dict[str, str]:
    """The configured name of every role, in configuration order.

    An entry of `lancedb.databases` is matched to a role by the last segment
    of its location. All three roles, or nothing.
    """
    names: dict[str, str] = {}
    for name, location in config.lancedb.databases.items():
        filename = Path(str(location)).name
        role = ROLE_BY_FILENAME.get(filename)
        if role is None:
            raise ValueError(
                f"{filename} is not a collection routing database; "
                f"the run needs {', '.join(ROLE_BY_FILENAME)}"
            )
        names[role] = name
    missing = [f for f, role in ROLE_BY_FILENAME.items() if role not in names]
    if missing:
        raise ValueError(f"lancedb.databases places no {', '.join(missing)}")
    return names


def cue_for(index: int) -> str:
    return CUES[index % len(CUES)]


def with_cue(question: str, names: Sequence[str]) -> str:
    if not names:
        return question
    if len(names) == 1:
        return f"In the {names[0]} collection: {question}"
    return f"In the {', '.join(names[:-1])} and {names[-1]} collections: {question}"


def _frames_rows() -> list[dict[str, Any]]:
    rows = []
    for doc in load_frames_questions():
        uris = list(question_expected_uris(doc))
        rows.append(
            {
                "member": "frames",
                "question_id": f"frames:{doc['id']}",
                "question": doc["Prompt"],
                "answer": doc["Answer"],
                "relevant_uris": uris,
                "expected_roles": sorted({frames_role(uri) for uri in uris}),
            }
        )
    return rows


def _orb_rows(limit: int) -> list[dict[str, Any]]:
    answers = {row["query_id"]: row["answer"] for row in load_orb_qa()}
    rows: list[dict[str, Any]] = []
    for row in load_orb_retrieval():
        if len(rows) == limit:
            break
        rows.append(
            {
                "member": "orb",
                "question_id": f"orb:{row['query_id']}",
                "question": row["query"],
                "answer": answers[row["query_id"]],
                "relevant_uris": [row["doc_id"]],
                "expected_roles": ["papers"],
            }
        )
    return rows


def load_routing_corpus() -> Dataset:
    """FRAMES and ORB questions interleaved, as many of each, with the cue
    each question carries assigned by its position."""
    frames = _frames_rows()
    orb = _orb_rows(len(frames))
    rows = [row for pair in zip(frames, orb, strict=False) for row in pair]
    for index, row in enumerate(rows):
        row["cue"] = cue_for(index)
    return Dataset.from_list(rows)


def build_routing_case(
    index: int, row: Mapping[str, Any], names: Mapping[str, str] | None = None
) -> Case[str, str, dict[str, Any]]:
    """`names` maps roles to configured names; unconfigured, the roles stand in."""
    if names is None:
        names = {role: role for role in ROLE_BY_FILENAME.values()}
    collections = list(names.values())
    expected = [names[role] for role in row["expected_roles"]]
    cue = row["cue"]
    if cue == "named":
        cued = expected
    elif cue == "misnamed":
        others = [name for name in collections if name not in expected]
        cued = [others[shard_of(row["question_id"], len(others))]]
    elif cue == "outside":
        cued = [OUTSIDE_NAME]
    else:
        cued = []
    raw_id = row["question_id"].split(":", 1)[1]
    return Case(
        name=f"{index}_{cue}_{row['member']}_{raw_id[:8]}",
        inputs=with_cue(row["question"], cued),
        expected_output=row["answer"],
        metadata={
            "question_id": row["question_id"],
            "member": row["member"],
            "cue": cue,
            "cued_sources": cued,
            "expected_sources": expected,
            "collections": collections,
            "relevant_uris": list(row["relevant_uris"]),
            "case_index": str(index),
        },
    )


def configure(spec: DatasetSpec, config: AppConfig) -> DatasetSpec:
    """The spec bound to the names `lancedb.databases` gives the collections."""
    return replace(
        spec, qa_case_builder=partial(build_routing_case, names=resolve_names(config))
    )


def _no_documents() -> Dataset:
    raise RuntimeError(
        "collection_routing reads the frames shards and the ORB database; "
        "build the shards with `evaluations split`"
    )


def _no_document(doc: Mapping[str, Any]) -> None:
    return None


def _routing_spec(key: str, collection_names: str) -> DatasetSpec:
    return DatasetSpec(
        key=key,
        # The downloadable artifact; the shards are split from it.
        db_filename="frames.lancedb",
        document_loader=_no_documents,
        document_mapper=_no_document,
        qa_loader=load_routing_corpus,
        qa_case_builder=build_routing_case,
        citation_evaluator=CitationMAPEvaluator(),
        case_evaluators=[CollectionRoutingEvaluator()],
        configure=configure,
        experiment_metadata={"collection_names": collection_names},
    )


COLLECTION_ROUTING_SPEC = _routing_spec("collection_routing", "descriptive")
COLLECTION_ROUTING_OPAQUE_SPEC = _routing_spec("collection_routing_opaque", "opaque")
