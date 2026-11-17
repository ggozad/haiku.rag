import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from string import ascii_uppercase
from typing import Any

from datasets import Dataset
from pydantic_evals import Case

from evaluations.config import DatasetSpec, DocumentPayload, ScopedQuestion
from evaluations.evaluators import LetterMatchEvaluator
from evaluations.evaluators.letter_match import FINAL_ANSWER

SAMPLE_SIZE = 300
SAMPLE_SEED = 0


def get_cache_dir() -> Path:
    return Path.home() / ".cache" / "haiku.rag" / "evaluations" / "cg_bench"


def _video_uids() -> list[str]:
    return sorted(path.stem for path in (get_cache_dir() / "videos").glob("*.mp4"))


def _questions() -> list[dict[str, Any]]:
    """The questions about downloaded videos, in qid order, qid a string as `--filter-ids` gives it."""
    uids = set(_video_uids())
    questions = json.loads((get_cache_dir() / "repo" / "cgbench.json").read_text())
    return [
        {**q, "qid": str(q["qid"])}
        for q in sorted(questions, key=lambda q: q["qid"])
        if q["video_uid"] in uids
    ]


def stratified_sample(
    questions: Sequence[dict[str, Any]], size: int, seed: int
) -> list[dict[str, Any]]:
    """`size` questions, allocated to each sub_category in proportion by largest remainder."""
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in sorted(questions, key=lambda q: int(q["qid"])):
        strata[question["sub_category"]].append(question)
    shares = {name: size * len(rows) / len(questions) for name, rows in strata.items()}
    counts = {name: math.floor(share) for name, share in shares.items()}
    by_remainder = sorted(shares, key=lambda name: (counts[name] - shares[name], name))
    for name in by_remainder[: size - sum(counts.values())]:
        counts[name] += 1
    rng = random.Random(seed)
    sample = [
        q for name in sorted(strata) for q in rng.sample(strata[name], counts[name])
    ]
    return sorted(sample, key=lambda q: int(q["qid"]))


def load_videos() -> Dataset:
    return Dataset.from_dict({"video_uid": _video_uids()})


def load_questions(sample: int | None) -> Dataset:
    questions = _questions()
    if sample is not None:
        questions = stratified_sample(questions, sample, SAMPLE_SEED)
    return Dataset.from_list(questions)


def _video_path(video_uid: str) -> Path:
    return get_cache_dir() / "videos" / f"{video_uid}.mp4"


def video_uri(video_uid: str) -> str:
    return _video_path(video_uid).absolute().as_uri()


def map_video(row: Mapping[str, Any]) -> DocumentPayload | None:
    video_uid = row["video_uid"]
    return DocumentPayload(
        uri=video_uri(video_uid), source_path=_video_path(video_uid), title=video_uid
    )


def build_case(
    index: int, row: Mapping[str, Any]
) -> Case[ScopedQuestion, str, dict[str, Any]]:
    letters = ascii_uppercase[: len(row["choices"])]
    options = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(letters, row["choices"])
    )
    instruction = (
        f"Select the best option from A through {letters[-1]} and end your answer "
        f"with the line '{FINAL_ANSWER} <letter>'."
    )
    question = (
        f"The question is about a video.\n\n{row['question']}\n\n{options}\n\n"
        f"{instruction}"
    )
    metadata = {
        key: row[key]
        for key in ("qid", "video_uid", "sub_category", "domain", "clue_intervals")
    }
    return Case(
        name=f"{index}_{row['qid']}",
        inputs=ScopedQuestion(
            question=question,
            document_filter=f"uri = '{video_uri(row['video_uid'])}'",
        ),
        expected_output=row["right_answer"],
        metadata=metadata,
    )


def _spec(key: str, sample: int | None) -> DatasetSpec:
    return DatasetSpec(
        key=key,
        db_filename="cg_bench_subset.lancedb",
        document_loader=load_videos,
        document_mapper=map_video,
        qa_loader=partial(load_questions, sample),
        qa_case_builder=build_case,
        qa_evaluator=LetterMatchEvaluator(standalone=True),
        pair_key="qid",
        case_id_key="qid",
    )


CG_BENCH_SPEC = _spec("cg_bench", None)
CG_BENCH_SAMPLE_SPEC = _spec("cg_bench_sample", SAMPLE_SIZE)
