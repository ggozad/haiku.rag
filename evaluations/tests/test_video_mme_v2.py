import zipfile
from pathlib import Path

import pandas as pd
import pytest

from evaluations.config import ScopedQuestion
from evaluations.datasets import DATASETS, video_mme_v2


def _row(video_id: str, n: int, answer: str = "B") -> dict:
    return {
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "group_type": "relevance",
        "group_structure": "4",
        "question_id": f"{video_id}-{n}",
        "question": f"Question {n} about {video_id}?",
        "options": "A. One.\nB. Two.",
        "answer": answer,
        "level": "1",
        "second_head": "Frame-Only",
        "third_head": "Basic Counting",
    }


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A local copy of the dataset repo: three videos, two questions each."""
    repo = tmp_path / "repo"
    (repo / "videos").mkdir(parents=True)
    rows = [_row(v, n) for v in ("001", "002", "021") for n in (1, 2)]
    pd.DataFrame(rows).to_parquet(repo / "test.parquet")
    for archive, videos in (("001", ("001", "002")), ("002", ("021",))):
        with zipfile.ZipFile(repo / "videos" / f"{archive}.zip", "w") as z:
            for v in videos:
                z.writestr(f"{v}.mp4", f"video {v}")

    def fake_download(repo_id, filename, *, repo_type, local_dir):
        assert repo_id == video_mme_v2.REPO_ID
        assert repo_type == "dataset"
        return str(repo / filename)

    monkeypatch.setattr(video_mme_v2, "hf_hub_download", fake_download)
    monkeypatch.setattr(video_mme_v2, "get_cache_dir", lambda: tmp_path / "cache")
    return repo


def test_videos_are_limited_to_the_first_n(repo: Path) -> None:
    assert list(video_mme_v2.load_videos(2)["video_id"]) == ["001", "002"]
    assert list(video_mme_v2.load_videos(None)["video_id"]) == ["001", "002", "021"]


def test_questions_are_limited_to_the_first_n_videos(repo: Path) -> None:
    questions = list(video_mme_v2.load_questions(2)["question_id"])
    assert questions == ["001-1", "001-2", "002-1", "002-2"]


def test_a_video_is_extracted_from_its_archive(repo: Path) -> None:
    payload = video_mme_v2.map_video({"video_id": "021"})

    assert payload is not None
    assert payload.uri == "video-mme-v2/021.mp4"
    assert payload.source_path is not None
    assert payload.source_path.name == "021.mp4"
    assert payload.source_path.read_text() == "video 021"


def test_a_case_asks_about_its_own_video(repo: Path) -> None:
    case = video_mme_v2.build_case(1, _row("002", 3, answer="F"))

    assert isinstance(case.inputs, ScopedQuestion)
    assert case.inputs.document_filter == "uri = 'video-mme-v2/002.mp4'"
    assert "Question 3 about 002?" in case.inputs.question
    assert "A. One.\nB. Two." in case.inputs.question
    assert "Final Answer:" in case.inputs.question
    assert case.expected_output == "F"
    assert case.metadata is not None
    assert case.metadata["question_id"] == "002-3"
    assert case.metadata["second_head"] == "Frame-Only"


def test_the_pilot_covers_twenty_videos() -> None:
    pilot = DATASETS["video_mme_v2_pilot"]
    full = DATASETS["video_mme_v2"]

    assert pilot.document_limit == 20
    assert full.document_limit is None
    assert pilot.db_filename != full.db_filename


@pytest.mark.parametrize(
    ("group_type", "structure", "scores", "expected"),
    [
        ("relevance", "4", [1, 1, 1, 1], 100.0),
        ("relevance", "4", [0, 1, 0, 1], 25.0),
        ("logic", "[1,2,3,4]", [1, 1, 0, 1], 25.0),
        ("logic", "[1,[2,3],4]", [1, 0, 1, 0], 100.0 * 4 / 12),
        ("logic", "[[1,2],3,4]", [0, 1, 1, 1], 10.0),
    ],
    ids=[
        "relevance-all",
        "relevance-two",
        "logic-truncates-at-first-error",
        "logic-parallel-middle",
        "logic-parallel-start",
    ],
)
def test_group_rating(
    group_type: str, structure: str, scores: list[int], expected: float
) -> None:
    rating = video_mme_v2.group_rating(group_type, structure, scores)
    assert rating == pytest.approx(expected)


def test_rating_averages_groups_and_counts_a_missing_answer_wrong() -> None:
    rows = [{**_row("001", n), "level": "2"} for n in (1, 2, 3, 4)] + [
        {**_row("002", n), "second_head": "Order"} for n in (1, 2, 3, 4)
    ]
    passed = {"001-1": True, "001-2": True, "001-3": True, "001-4": True}
    passed |= {"002-1": True, "002-2": False}

    rating = video_mme_v2.rating(pd.DataFrame(rows), passed)

    assert rating["overall"] == pytest.approx((100.0 + 100.0 / 16) / 2)
    assert rating["level"] == {"1": pytest.approx(100.0 / 16), "2": 100.0}
    assert rating["second_head"]["Order"] == pytest.approx(100.0 / 16)
    assert rating["accuracy"] == pytest.approx(5 / 8)
    assert rating["answered"] == 6
