import ast
import zipfile
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download
from pydantic_evals import Case

from evaluations.config import DatasetSpec, DocumentPayload, ScopedQuestion
from evaluations.evaluators import LetterMatchEvaluator
from evaluations.evaluators.letter_match import FINAL_ANSWER

REPO_ID = "MME-Benchmarks/Video-MME-v2"
VIDEOS_PER_ARCHIVE = 20
PILOT_VIDEOS = 20

_INSTRUCTION = (
    "Select the best option from A through H and end your answer with the line "
    f"'{FINAL_ANSWER} <letter>'."
)


def get_cache_dir() -> Path:
    cache_dir = Path.home() / ".cache" / "haiku.rag" / "evaluations" / "video_mme_v2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _download(filename: str) -> Path:
    return Path(
        hf_hub_download(
            REPO_ID,
            filename,
            repo_type="dataset",
            local_dir=get_cache_dir() / "repo",
        )
    )


def _questions() -> pd.DataFrame:
    return pd.read_parquet(_download("test.parquet")).sort_values("question_id")


def _video_ids(limit: int | None) -> list[str]:
    return sorted(_questions()["video_id"].unique())[:limit]


def load_videos(limit: int | None) -> Dataset:
    return Dataset.from_dict({"video_id": _video_ids(limit)})


def load_questions(limit: int | None) -> Dataset:
    questions = _questions()
    questions = questions[questions["video_id"].isin(_video_ids(limit))]
    return Dataset.from_pandas(questions, preserve_index=False)


def video_uri(video_id: str) -> str:
    return f"video-mme-v2/{video_id}.mp4"


def map_video(row: Mapping[str, Any]) -> DocumentPayload | None:
    video_id = row["video_id"]
    target = get_cache_dir() / "videos" / f"{video_id}.mp4"
    if not target.exists():
        archive = (int(video_id) - 1) // VIDEOS_PER_ARCHIVE + 1
        with zipfile.ZipFile(_download(f"videos/{archive:03d}.zip")) as z:
            z.extract(f"{video_id}.mp4", target.parent)
    return DocumentPayload(uri=video_uri(video_id), source_path=target, title=video_id)


def build_case(
    index: int, row: Mapping[str, Any]
) -> Case[ScopedQuestion, str, dict[str, str]]:
    question = (
        f"The question is about a video.\n\n{row['question']}\n\n"
        f"{row['options']}\n\n{_INSTRUCTION}"
    )
    metadata = {
        key: str(row[key])
        for key in (
            "question_id",
            "video_id",
            "group_type",
            "group_structure",
            "level",
            "second_head",
            "third_head",
        )
    }
    return Case(
        name=f"{index}_{row['question_id']}",
        inputs=ScopedQuestion(
            question=question,
            document_filter=f"uri = '{video_uri(row['video_id'])}'",
        ),
        expected_output=row["answer"],
        metadata=metadata,
    )


# Video-MME-v2's group scoring (evaluation/test_video_mme_v2.py): a group of four
# questions scores by how many it answers, and a logic group stops at its first error.
_RATINGS = {
    "4": (0.0, 100 / 16, 100 * 4 / 16, 100 * 9 / 16, 100.0),
    "[1,2,3,4]": (0.0, 100 / 16, 100 * 4 / 16, 100 * 9 / 16, 100.0),
    "[1,[2,3],4]": (0.0, 100 / 12, 100 * 4 / 12, 100 * 7 / 12, 100.0),
    "[[1,2],3,4]": (0.0, 100 / 10, 100 * 2 / 10, 100 * 5 / 10, 100.0),
}


def group_rating(group_type: str, group_structure: str, scores: list[int]) -> float:
    """The rating, out of 100, of one video's four answers in question order."""
    if group_type == "relevance":
        return _RATINGS["4"][sum(scores)]
    structure = ast.literal_eval(group_structure)
    last_correct = -1
    for index, score in enumerate(scores):
        if not score:
            break
        last_correct = index
    if structure == [1, [2, 3], 4] and last_correct == 0 and scores[2]:
        last_correct = 1
    if structure == [[1, 2], 3, 4] and last_correct == -1 and scores[1]:
        last_correct = 0
    return _RATINGS[group_structure][last_correct + 1]


def rating(questions: pd.DataFrame, passed: Mapping[str, bool]) -> dict[str, Any]:
    """Mean group rating overall and by level and category. Unanswered questions count wrong."""
    groups = []
    for _, group in questions.sort_values("question_id").groupby("video_id"):
        last = group.iloc[-1]
        scores = [int(passed.get(q, False)) for q in group["question_id"]]
        value = group_rating(last["group_type"], last["group_structure"], scores)
        groups.append({**last.to_dict(), "rating": value})
    frame = pd.DataFrame(groups)
    return {
        "overall": float(frame["rating"].mean()),
        "group_type": frame.groupby("group_type")["rating"].mean().to_dict(),
        "level": frame.groupby("level")["rating"].mean().to_dict(),
        "second_head": frame.groupby("second_head")["rating"].mean().to_dict(),
        "accuracy": sum(passed.get(q, False) for q in questions["question_id"])
        / len(questions),
        "answered": sum(q in passed for q in questions["question_id"]),
    }


def _spec(key: str, db_filename: str, videos: int | None) -> DatasetSpec:
    return DatasetSpec(
        key=key,
        db_filename=db_filename,
        document_loader=partial(load_videos, videos),
        document_mapper=map_video,
        qa_loader=partial(load_questions, videos),
        qa_case_builder=build_case,
        qa_evaluator=LetterMatchEvaluator(),
        document_limit=videos,
        pair_key="question_id",
        case_id_key="question_id",
    )


VIDEO_MME_V2_SPEC = _spec("video_mme_v2", "video_mme_v2.lancedb", None)
VIDEO_MME_V2_PILOT_SPEC = _spec(
    "video_mme_v2_pilot", "video_mme_v2_pilot.lancedb", PILOT_VIDEOS
)
