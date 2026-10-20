import asyncio

import numpy as np
import pytest
from docling_core.types.doc.document import DoclingDocument
from starlette.exceptions import HTTPException

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import DuplicateDetectionConfig
from haiku.rag.doctor import VectorScan
from haiku.rag.store.models import Chunk
from haiku_rag_app.doctor import Retained
from haiku_rag_app.projection import place, project, similarity


def _scan(centroids, counts=None, ids=None) -> VectorScan:
    centroids = np.asarray(centroids, dtype=np.float32)
    n = len(centroids)
    return VectorScan(
        document_ids=ids or [f"d{i}" for i in range(n)],
        centroids=centroids,
        counts=np.asarray(counts or [3] * n),
        stored_dimension=centroids.shape[1] if n else 4,
        dimension=centroids.shape[1] if n else 4,
        unembedded_count=0,
        unembedded_sample=[],
    )


def _clusters(per=12, dim=8, seed=3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for label in range(3):
        rows.append(np.eye(dim)[label] + rng.normal(scale=0.05, size=(per, dim)))
        labels += [label] * per
    unit = np.vstack(rows)
    return unit / np.linalg.norm(unit, axis=1)[:, None], np.array(labels)


def test_separated_clusters_stay_separated():
    unit, labels = _clusters()
    coordinates = place(unit)
    assert coordinates.shape == (len(unit), 2)
    distance = np.linalg.norm(coordinates[:, None] - coordinates[None], axis=2)
    np.fill_diagonal(distance, np.inf)
    assert (labels[distance.argmin(axis=1)] == labels).all()


def test_placement_is_deterministic():
    unit, _ = _clusters()
    assert np.array_equal(place(unit), place(unit))


def test_copies_share_a_point():
    unit, _ = _clusters()
    coordinates = place(np.vstack([unit, unit[:1]]))
    assert np.array_equal(coordinates[0], coordinates[-1])
    assert np.isfinite(coordinates).all()


@pytest.mark.parametrize(
    "unit",
    [np.zeros((0, 4)), np.eye(4)[:1], np.eye(4)[:2], np.ones((5, 4)) / 2],
)
def test_fewer_than_three_distinct_documents_lie_on_a_line(unit):
    coordinates = place(unit)
    assert coordinates.shape == (unit.shape[0], 2)
    assert not coordinates[:, 1].any()
    assert len(np.unique(coordinates, axis=0)) == len(np.unique(unit, axis=0))


def test_a_document_with_no_embedded_chunk_is_left_off():
    projection = project(
        _scan([[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]], counts=[3, 0, 3]),
        {"d0": {"title": "zero", "metadata": '{"content_type": "text/html"}'}},
        DuplicateDetectionConfig(),
    )
    assert [p["id"] for p in projection.points] == ["d0", "d2"]
    assert projection.left_off == 1
    assert projection.points[0]["content_type"] == "text/html"
    assert projection.points[0]["title"] == "zero"
    assert all(np.isfinite([p["x"] for p in projection.points]))


def test_documents_without_a_centroid_count_as_left_off():
    projection = project(
        _scan([[1, 0, 0, 0], [0, 1, 0, 0]]),
        {"d0": {}, "d1": {}, "chunkless": {}},
        DuplicateDetectionConfig(),
    )
    assert projection.left_off == 1


def test_non_finite_centroids_are_left_off_and_large_ones_kept():
    projection = project(
        _scan([[np.inf, 0, 0, 0], [3e38, 3e38, 0, 0], [0, 1, 0, 0]]),
        {},
        DuplicateDetectionConfig(),
    )
    assert [p["id"] for p in projection.points] == ["d1", "d2"]
    assert projection.left_off == 1
    assert np.isfinite(projection.unit).all()
    assert np.linalg.norm(projection.unit, axis=1) == pytest.approx([1.0, 1.0])
    assert all(np.isfinite([p["x"] for p in projection.points]))


def test_an_empty_corpus_gives_an_empty_map():
    projection = project(_scan(np.zeros((0, 4))), {}, DuplicateDetectionConfig())
    assert projection.points == []


def test_near_duplicates_form_a_family_and_similarity_is_cosine():
    projection = project(
        _scan([[1, 0, 0, 0], [0.99, 0.01, 0, 0], [0, 0, 1, 0]]),
        {},
        DuplicateDetectionConfig(),
    )
    assert [p["family"] for p in projection.points] == [0, 0, None]
    assert projection.families[0].members == ["d0", "d1"]
    assert similarity(projection.unit, projection.ids, "d0") == pytest.approx(
        [1.0, 0.99995, 0.0], abs=1e-4
    )


def test_the_budget_evicts_the_oldest_and_refuses_what_never_fits():
    retained = Retained(budget=64)
    small = np.zeros((2, 4), dtype=np.float32)
    assert retained.keep("a", "g1", ["x", "y"], small)
    assert retained.keep("b", "g2", ["x", "y"], small)
    assert retained.keep("c", "g3", ["x", "y"], np.zeros((4, 4), dtype=np.float32))
    assert retained.get("a") is None and retained.get("b") is None
    assert not retained.keep("d", "g4", ["x"], np.zeros((10, 4), dtype=np.float32))
    assert retained.get("d") is None
    assert retained.keep("c", "g5", ["x"], small)
    kept = retained.get("c")
    assert kept is not None and kept.generation == "g5"
    assert kept.unit.shape == (2, 4)


@pytest.fixture
async def mapped(seeded):
    """Two documents with the same three chunk directions, and a third apart."""
    config, _ = seeded
    dim = config.embeddings.model.vector_dim
    eye = np.eye(dim)
    ids = {}
    async with HaikuRAG(config=config, sources=["alpha"]) as rag:
        for name, directions in {
            "copy-a": [1, 2, 3],
            "copy-b": [1, 2, 3],
            "other": [7, 8, 9],
        }.items():
            document = await rag.import_document(
                DoclingDocument(name=name),
                [
                    Chunk(content=f"{name} {n}", embedding=eye[d].tolist(), order=n)
                    for n, d in enumerate(directions)
                ],
                uri=f"test://alpha/{name}",
                title=name,
            )
            ids[name] = document.id
    return ids


async def _settled(client, source: str) -> dict:
    for _ in range(600):
        projection = (await client.get(f"/api/doctor/{source}/projection")).json()[
            "projection"
        ]
        if projection and projection["status"] in ("done", "failed"):
            return projection
        await asyncio.sleep(0.02)
    raise AssertionError("the projection did not settle")


async def test_the_map_places_documents_and_their_families(client, mapped):
    assert (await client.get("/api/doctor/alpha/projection")).json() == {
        "projection": None
    }
    assert (await client.get("/api/doctor/alpha/duplicates.yaml")).status_code == 404
    started = await client.post("/api/doctor/alpha/projection")
    assert started.status_code == 202
    projection = await _settled(client, "alpha")
    result = projection["result"]
    by_title = {p["title"]: p for p in result["points"]}
    assert set(by_title) >= {"copy-a", "copy-b", "other"}
    assert by_title["copy-a"]["family"] == by_title["copy-b"]["family"] == 0
    assert by_title["other"]["family"] is None
    assert by_title["copy-a"]["chunks"] == 3
    assert result["families"][0]["label"][mapped["copy-a"]] == "test://alpha/copy-a"
    assert result["neighbours"] is True
    assert projection["stale"] is False
    assert "Projecting documents" in projection["progress"]

    similar = (
        await client.get(
            "/api/doctor/alpha/projection/similarity",
            params={"id": mapped["copy-a"], "generation": result["generation"]},
        )
    ).json()["similarity"]
    assert len(similar) == len(result["points"])
    by_id = {p["id"]: s for p, s in zip(result["points"], similar)}
    assert by_id[mapped["copy-b"]] == pytest.approx(1.0)
    assert by_id[mapped["other"]] == pytest.approx(0.0)

    exported = await client.get("/api/doctor/alpha/duplicates.yaml")
    assert exported.headers["content-type"].startswith("application/yaml")
    assert "test://alpha/copy-b" in exported.text


async def test_a_changed_document_marks_the_map_stale(app, client, mapped, seeded):
    config, _ = seeded
    await client.post("/api/doctor/alpha/projection")
    await _settled(client, "alpha")
    async with HaikuRAG(config=config, sources=["alpha"]) as rag:
        document = await rag.get_document_by_id(mapped["other"])
        document.title = "renamed"
        await rag.document_repository.update_meta(document)
    projection = (await client.get("/api/doctor/alpha/projection")).json()["projection"]
    assert projection["stale"] is True


async def test_staleness_is_unknown_while_the_database_is_unavailable(
    app, client, mapped, monkeypatch
):
    await client.post("/api/doctor/alpha/projection")
    await _settled(client, "alpha")

    async def unavailable(name):
        raise HTTPException(503, "gone")

    monkeypatch.setattr(app.state.databases, "reader", unavailable)
    projection = (await client.get("/api/doctor/alpha/projection")).json()["projection"]
    assert projection["stale"] is None


async def test_staleness_is_unknown_when_the_versions_cannot_be_read(
    client, mapped, monkeypatch
):
    import lancedb

    await client.post("/api/doctor/alpha/projection")
    await _settled(client, "alpha")

    async def unreadable(self):
        raise OSError("object store unreachable")

    monkeypatch.setattr(lancedb.AsyncTable, "checkout_latest", unreadable)
    projection = (await client.get("/api/doctor/alpha/projection")).json()["projection"]
    assert projection["stale"] is None


async def test_similarity_needs_a_kept_map_and_a_known_document(app, client, mapped):
    url = "/api/doctor/alpha/projection/similarity"
    params = {"generation": "g"}
    assert (await client.get(url, params={"id": "x", **params})).status_code == 404
    app.state.retained = Retained(budget=1)
    await client.post("/api/doctor/alpha/projection")
    projection = await _settled(client, "alpha")
    assert projection["result"]["neighbours"] is False
    params = {"generation": projection["result"]["generation"]}
    other = {"id": mapped["other"], **params}
    assert (await client.get(url, params=other)).status_code == 404
    app.state.retained = Retained()
    await client.post("/api/doctor/alpha/projection")
    projection = await _settled(client, "alpha")
    params = {"generation": projection["result"]["generation"]}
    assert (await client.get(url, params={"id": "nope", **params})).status_code == 404
    app.state.retained = Retained()
    evicted = (await client.get("/api/doctor/alpha/projection")).json()
    assert evicted["projection"]["result"]["neighbours"] is False


async def test_similarity_answers_only_for_the_map_it_names(client, mapped):
    url = "/api/doctor/alpha/projection/similarity"
    await client.post("/api/doctor/alpha/projection")
    first = (await _settled(client, "alpha"))["result"]["generation"]
    await client.post("/api/doctor/alpha/projection")
    second = (await _settled(client, "alpha"))["result"]["generation"]
    assert first != second

    doc = mapped["copy-a"]
    replaced = await client.get(url, params={"id": doc, "generation": first})
    assert replaced.status_code == 409
    current = await client.get(url, params={"id": doc, "generation": second})
    assert current.status_code == 200
    assert current.json()["generation"] == second
    assert (await client.get(url, params={"id": doc})).status_code == 400
