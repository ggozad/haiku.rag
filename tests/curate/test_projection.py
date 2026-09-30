import numpy as np

from haiku.rag.curate.projection import project
from tests.curate.test_detectors import _centroid


def _clusters(size: int) -> dict[str, bytes]:
    rng = np.random.default_rng(0)
    centroids = {}
    for i in range(2 * size):
        axes = np.zeros(8)
        axes[i // size] = 1.0
        axes[2:] = rng.normal(0, 0.05, 6)
        centroids[f"d{i}"] = _centroid(*axes)
    return centroids


def _mean_distance(points: dict, a: list[str], b: list[str]) -> float:
    return float(
        np.mean([np.hypot(*np.subtract(points[i], points[j])) for i in a for j in b])
    )


def test_fewer_than_three_documents_get_no_map():
    assert project({"d1": _centroid(1.0), "d2": _centroid(0.0, 1.0)}) == {}


def test_similar_documents_are_placed_together():
    points = project(_clusters(10))

    assert set(points) == {f"d{i}" for i in range(20)}
    first = [f"d{i}" for i in range(10)]
    second = [f"d{i}" for i in range(10, 20)]
    within = _mean_distance(points, first, first)
    assert within * 3 < _mean_distance(points, first, second)


def test_identical_documents_share_one_point():
    same = _centroid(1.0, 1.0)
    assert project({"a": same, "b": same, "c": same}) == {
        "a": (0.0, 0.0),
        "b": (0.0, 0.0),
        "c": (0.0, 0.0),
    }


def test_copies_share_a_point_beside_one_other_document():
    centroids = {f"copy{i}": _centroid(1.0) for i in range(9)}
    centroids["other"] = _centroid(0.0, 1.0)

    points = project(centroids)

    assert len({points[f"copy{i}"] for i in range(9)}) == 1
    assert points["other"] != points["copy0"]


def test_copies_share_a_point_on_a_map():
    centroids = _clusters(10)
    for i in range(3):
        centroids[f"copy{i}"] = centroids["d0"]

    points = project(centroids)

    assert {points[f"copy{i}"] for i in range(3)} == {points["d0"]}
    assert len(set(points.values())) == 20
