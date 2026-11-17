import json
from collections import Counter
from pathlib import Path

import pytest

from evaluations.config import ScopedQuestion
from evaluations.datasets import DATASETS, cg_bench
from evaluations.evaluators import LetterMatchEvaluator


def _question(qid: int, video_uid: str, sub_category: str = "Entity Perception"):
    return {
        "qid": qid,
        "video_uid": video_uid,
        "question": f"Question {qid} about {video_uid}?",
        "answer": "corn",
        "choices": ["pumpkins", "beans", "corn"],
        "right_answer": "C",
        "clue_intervals": [[5, 7.5]],
        "domain": "Life Record",
        "sub_category": sub_category,
        "duration": 1980,
    }


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Two downloaded videos, and a question about a video that is not downloaded."""
    (tmp_path / "videos").mkdir()
    (tmp_path / "repo").mkdir()
    for uid in ("vid-b", "vid-a"):
        (tmp_path / "videos" / f"{uid}.mp4").write_text(f"video {uid}")
    questions = [_question(3, "vid-b"), _question(1, "vid-a"), _question(2, "absent")]
    (tmp_path / "repo" / "cgbench.json").write_text(json.dumps(questions))
    monkeypatch.setattr(cg_bench, "get_cache_dir", lambda: tmp_path)
    return tmp_path


def test_videos_are_the_downloaded_ones(cache: Path) -> None:
    assert list(cg_bench.load_videos()["video_uid"]) == ["vid-a", "vid-b"]


def test_questions_are_those_about_downloaded_videos(cache: Path) -> None:
    questions = cg_bench.load_questions(None)

    assert list(questions["qid"]) == ["1", "3"]
    assert questions[0]["clue_intervals"] == [[5, 7.5]]


def test_a_video_is_its_downloaded_file(cache: Path) -> None:
    payload = cg_bench.map_video({"video_uid": "vid-a"})

    assert payload is not None
    assert payload.source_path == cache / "videos" / "vid-a.mp4"
    assert payload.uri == payload.source_path.absolute().as_uri()
    assert payload.uri.startswith("file://")


def test_a_case_asks_about_its_own_video(cache: Path) -> None:
    row = {**_question(26, "vid-a"), "qid": "26"}

    case = cg_bench.build_case(4, row)

    assert isinstance(case.inputs, ScopedQuestion)
    uri = (cache / "videos" / "vid-a.mp4").absolute().as_uri()
    assert case.inputs.document_filter == f"uri = '{uri}'"
    assert "Question 26 about vid-a?" in case.inputs.question
    assert "A. pumpkins\nB. beans\nC. corn" in case.inputs.question
    assert "A through C" in case.inputs.question
    assert "Final Answer:" in case.inputs.question
    assert case.name == "4_26"
    assert case.expected_output == "C"
    assert case.metadata == {
        "qid": "26",
        "video_uid": "vid-a",
        "sub_category": "Entity Perception",
        "domain": "Life Record",
        "clue_intervals": [[5, 7.5]],
    }


def _population() -> list[dict]:
    sizes = {"Entity": 314, "Event": 289, "Text": 149, "Hallucination": 49, "Rare": 7}
    rows, qid = [], 0
    for sub_category, size in sizes.items():
        for _ in range(size):
            rows.append(_question(qid, "v", sub_category))
            qid += 1
    return rows


def test_the_sample_is_deterministic_and_of_the_requested_size() -> None:
    population = _population()

    first = cg_bench.stratified_sample(population, 300, seed=7)
    second = cg_bench.stratified_sample(list(reversed(population)), 300, seed=7)

    assert len(first) == 300
    assert len({q["qid"] for q in first}) == 300
    assert [q["qid"] for q in first] == [q["qid"] for q in second]
    assert first != cg_bench.stratified_sample(population, 300, seed=8)


def test_the_sample_is_proportional_to_each_sub_category() -> None:
    population = _population()
    sizes = Counter(q["sub_category"] for q in population)

    sample = Counter(
        q["sub_category"] for q in cg_bench.stratified_sample(population, 300, seed=7)
    )

    for sub_category, size in sizes.items():
        share = 300 * size / len(population)
        assert abs(sample[sub_category] - share) < 1


def test_the_sample_spec_draws_three_hundred_of_the_subset(cache: Path) -> None:
    rows = [_question(n, "vid-a", f"cat-{n % 4}") for n in range(400)]
    (cache / "repo" / "cgbench.json").write_text(json.dumps(rows))

    first = DATASETS["cg_bench_sample"].qa_loader()
    second = DATASETS["cg_bench_sample"].qa_loader()

    assert len(first) == 300
    assert first["qid"] == second["qid"]
    assert len(DATASETS["cg_bench"].qa_loader()) == 400


def test_both_specs_share_the_subset_database() -> None:
    full, sample = DATASETS["cg_bench"], DATASETS["cg_bench_sample"]

    for spec in (full, sample):
        assert spec.db_filename == "cg_bench_subset.lancedb"
        assert isinstance(spec.qa_evaluator, LetterMatchEvaluator)
        assert spec.pair_key == "qid"
        assert spec.case_id_key == "qid"
