import numpy as np
import pytest
from docling_core.types.doc.document import DoclingDocument
from starlette.exceptions import HTTPException

from haiku.rag.client import HaikuRAG
from haiku.rag.client.scope import DatabaseRef
from haiku.rag.config.models import DuplicateDetectionConfig
from haiku.rag.doctor import VectorScan
from haiku.rag.store.models import Chunk
from haiku_rag_app.similarity import Retained, compare, similarity
from tests.support import settled


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


def test_documents_with_no_embedded_chunk_are_left_out():
    compared = compare(
        _scan(
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            counts=[3, 0, 3, 3],
        ),
        {
            "d0": {"title": "zero", "uri": "test://zero"},
            "d2": {"title": "two", "uri": None},
            "chunkless": {},
        },
        DuplicateDetectionConfig(),
    )
    assert compared.documents == [
        {"id": "d0", "title": "zero", "uri": "test://zero"},
        {"id": "d2", "title": "two", "uri": None},
        {"id": "d3", "title": None, "uri": None},
    ]
    assert compared.ids == ["d0", "d2", "d3"]
    assert compared.left_off == 2


def test_non_finite_centroids_are_left_out_and_large_ones_kept():
    compared = compare(
        _scan([[np.inf, 0, 0, 0], [3e38, 3e38, 0, 0], [0, 1, 0, 0]]),
        {},
        DuplicateDetectionConfig(),
    )
    assert compared.ids == ["d1", "d2"]
    assert compared.left_off == 1
    assert np.isfinite(compared.unit).all()
    assert np.linalg.norm(compared.unit, axis=1) == pytest.approx([1.0, 1.0])


def test_an_empty_corpus_has_no_documents():
    compared = compare(_scan(np.zeros((0, 4))), {}, DuplicateDetectionConfig())
    assert compared.documents == []
    assert compared.families == []


def test_near_duplicates_form_a_family_and_similarity_is_cosine():
    compared = compare(
        _scan([[1, 0, 0, 0], [0.99, 0.01, 0, 0], [0, 0, 1, 0]]),
        {},
        DuplicateDetectionConfig(),
    )
    assert [family.members for family in compared.families] == [["d0", "d1"]]
    assert similarity(compared.unit, compared.ids, "d0") == pytest.approx(
        [1.0, 0.99995, 0.0], abs=1e-4
    )


def test_the_budget_evicts_the_oldest_and_refuses_what_never_fits():
    refs = {name: DatabaseRef.configured(name, f"{name}.lancedb") for name in "abcd"}
    retained = Retained(budget=64)
    small = np.zeros((2, 4), dtype=np.float32)
    assert retained.keep(refs["a"], "g1", ["x", "y"], small)
    assert retained.keep(refs["b"], "g2", ["x", "y"], small)
    assert retained.keep(
        refs["c"], "g3", ["x", "y"], np.zeros((4, 4), dtype=np.float32)
    )
    assert retained.get(refs["a"]) is None and retained.get(refs["b"]) is None
    assert not retained.keep(
        refs["d"], "g4", ["x"], np.zeros((10, 4), dtype=np.float32)
    )
    assert retained.get(refs["d"]) is None
    assert retained.keep(refs["c"], "g5", ["x"], small)
    kept = retained.get(refs["c"])
    assert kept is not None and kept.generation == "g5"
    assert kept.unit.shape == (2, 4)


@pytest.fixture
async def compared(seeded):
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


async def test_the_job_finds_families_and_scores_documents(client, compared):
    assert (await client.get("/api/doctor/alpha/similarity")).json() == {
        "similarity": None
    }
    assert (await client.get("/api/doctor/alpha/duplicates.yaml")).status_code == 404
    started = await client.post("/api/doctor/alpha/similarity")
    assert started.status_code == 202
    job = await settled(client, "alpha", "/similarity")
    result = job["result"]
    documents = {d["id"]: d for d in result["documents"]}
    assert documents[compared["copy-a"]] == {
        "id": compared["copy-a"],
        "title": "copy-a",
        "uri": "test://alpha/copy-a",
    }
    assert {compared["copy-a"], compared["copy-b"]} in [
        set(family["members"]) for family in result["families"]
    ]
    assert result["families"][0]["label"][compared["copy-a"]] == "test://alpha/copy-a"
    assert result["neighbours"] is True
    assert job["stale"] is False
    assert "Comparing documents" in job["progress"]

    similar = (
        await client.get(
            "/api/doctor/alpha/similarity/documents",
            params={"id": compared["copy-a"], "generation": result["generation"]},
        )
    ).json()["similarity"]
    assert len(similar) == len(result["documents"])
    by_id = {d["id"]: s for d, s in zip(result["documents"], similar)}
    assert by_id[compared["copy-b"]] == pytest.approx(1.0)
    assert by_id[compared["other"]] == pytest.approx(0.0)

    exported = await client.get("/api/doctor/alpha/duplicates.yaml")
    assert exported.headers["content-type"].startswith("application/yaml")
    assert "test://alpha/copy-b" in exported.text


async def test_a_changed_document_marks_the_job_stale(app, client, compared, seeded):
    config, _ = seeded
    await client.post("/api/doctor/alpha/similarity")
    await settled(client, "alpha", "/similarity")
    async with HaikuRAG(config=config, sources=["alpha"]) as rag:
        document = await rag.get_document_by_id(compared["other"])
        document.title = "renamed"
        await rag.document_repository.update_meta(document)
    job = (await client.get("/api/doctor/alpha/similarity")).json()["similarity"]
    assert job["stale"] is True


async def test_staleness_is_unknown_while_the_database_is_unavailable(
    app, client, compared, monkeypatch
):
    await client.post("/api/doctor/alpha/similarity")
    await settled(client, "alpha", "/similarity")

    async def unavailable(name):
        raise HTTPException(503, "gone")

    monkeypatch.setattr(app.state.configuration.current, "reader", unavailable)
    job = (await client.get("/api/doctor/alpha/similarity")).json()["similarity"]
    assert job["stale"] is None


async def test_staleness_is_unknown_when_the_versions_cannot_be_read(
    client, compared, monkeypatch
):
    import lancedb

    await client.post("/api/doctor/alpha/similarity")
    await settled(client, "alpha", "/similarity")

    async def unreadable(self):
        raise OSError("object store unreachable")

    monkeypatch.setattr(lancedb.AsyncTable, "checkout_latest", unreadable)
    job = (await client.get("/api/doctor/alpha/similarity")).json()["similarity"]
    assert job["stale"] is None


async def test_scores_need_kept_centroids_and_a_known_document(app, client, compared):
    url = "/api/doctor/alpha/similarity/documents"
    params = {"generation": "g"}
    assert (await client.get(url, params={"id": "x", **params})).status_code == 404
    app.state.retained = Retained(budget=1)
    await client.post("/api/doctor/alpha/similarity")
    job = await settled(client, "alpha", "/similarity")
    assert job["result"]["neighbours"] is False
    params = {"generation": job["result"]["generation"]}
    other = {"id": compared["other"], **params}
    assert (await client.get(url, params=other)).status_code == 404
    app.state.retained = Retained()
    await client.post("/api/doctor/alpha/similarity")
    job = await settled(client, "alpha", "/similarity")
    params = {"generation": job["result"]["generation"]}
    assert (await client.get(url, params={"id": "nope", **params})).status_code == 404
    app.state.retained = Retained()
    evicted = (await client.get("/api/doctor/alpha/similarity")).json()
    assert evicted["similarity"]["result"]["neighbours"] is False


async def test_scores_answer_only_for_the_job_they_name(client, compared):
    url = "/api/doctor/alpha/similarity/documents"
    await client.post("/api/doctor/alpha/similarity")
    first = (await settled(client, "alpha", "/similarity"))["result"]["generation"]
    await client.post("/api/doctor/alpha/similarity")
    second = (await settled(client, "alpha", "/similarity"))["result"]["generation"]
    assert first != second

    doc = compared["copy-a"]
    replaced = await client.get(url, params={"id": doc, "generation": first})
    assert replaced.status_code == 409
    current = await client.get(url, params={"id": doc, "generation": second})
    assert current.status_code == 200
    assert current.json()["generation"] == second
    assert (await client.get(url, params={"id": doc})).status_code == 400
