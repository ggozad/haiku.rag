from contextlib import nullcontext

import httpx
import pytest
from httpx import ASGITransport

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, get_config
from haiku.rag.ingester.api.server import APIState, build_app
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.store.models.document import Document
from tests.conftest import writing
from tests.documents import docling_doc


@pytest.fixture
async def rag(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        yield client


@pytest.fixture
def api(jobs, sync, rag):
    def editing(name):
        if name not in ("docs", "team/docs"):
            raise KeyError(name)
        return nullcontext(rag)

    state = APIState(config=AppConfig(), job_repo=jobs, sync_repo=sync)
    state.editing = editing
    return httpx.AsyncClient(
        transport=ASGITransport(app=build_app(state)), base_url="http://testserver"
    )


async def _import(client: HaikuRAG, uri: str, **kwargs) -> Document:
    dim = get_config().embeddings.model.vector_dim
    doc = await client.import_document(
        docling_doc("d", "body"),
        [Chunk(content="body", embedding=[0.1] * dim, order=0)],
        uri=uri,
        **kwargs,
    )
    assert doc.id is not None
    return doc


async def test_patch_sets_title_and_editor_metadata(api, rag):
    doc = await _import(rag, "mem://a", metadata={"team": "x"})

    async with api:
        resp = await api.patch(
            f"/databases/docs/documents/{doc.id}",
            json={"title": "Edited", "user": {"reviewer": "ana"}},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == doc.id
    assert body["title"] == "Edited"
    assert body["metadata"] == {"team": "x", "user": {"reviewer": "ana"}}
    assert body["updated_at"] > doc.updated_at.isoformat()
    stored = await rag.get_document_by_id(doc.id)
    assert stored is not None
    assert stored.title == "Edited"
    assert stored.metadata["user"] == {"reviewer": "ana"}


async def test_patch_leaves_an_omitted_field_unchanged(api, rag):
    doc = await _import(rag, "mem://a", title="Old")

    async with api:
        resp = await api.patch(
            f"/databases/docs/documents/{doc.id}", json={"user": {"k": "v"}}
        )

    assert resp.status_code == 200
    assert resp.json()["title"] == "Old"


async def test_patch_answers_under_a_database_name_with_a_slash(api, rag):
    doc = await _import(rag, "mem://a")

    async with api:
        resp = await api.patch(
            f"/databases/team%2Fdocs/documents/{doc.id}", json={"title": "Edited"}
        )

    assert resp.status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        {"title": None},
        {"user": None},
        {"user": ["not", "an", "object"]},
        {"metadata": {"k": "v"}},
    ],
    ids=["null-title", "null-user", "user-not-object", "unknown-field"],
)
async def test_patch_rejects_an_invalid_body(api, rag, body):
    doc = await _import(rag, "mem://a", title="Old")

    async with api:
        resp = await api.patch(f"/databases/docs/documents/{doc.id}", json=body)

    assert resp.status_code == 422
    stored = await rag.get_document_by_id(doc.id)
    assert stored is not None
    assert stored.title == "Old"


async def test_patch_of_a_missing_document_is_not_found(api):
    async with api:
        resp = await api.patch("/databases/docs/documents/missing", json={"title": "x"})

    assert resp.status_code == 404


@pytest.mark.parametrize("method", ["PATCH", "DELETE"])
async def test_a_database_the_ingester_does_not_serve_is_not_found(api, rag, method):
    doc = await _import(rag, "mem://a")

    async with api:
        resp = await api.request(
            method, f"/databases/other/documents/{doc.id}", json={"title": "x"}
        )

    assert resp.status_code == 404
    assert await rag.get_document_by_id(doc.id) is not None


async def test_delete_removes_a_document_no_source_owns(api, rag):
    doc = await _import(rag, "mem://a")

    async with api:
        resp = await api.delete(f"/databases/docs/documents/{doc.id}")

    assert resp.status_code == 200
    assert resp.json() == {"document_id": doc.id, "deleted": True}
    assert await rag.get_document_by_id(doc.id) is None


async def test_delete_refuses_a_source_owned_document(api, rag):
    doc = await _import(rag, "mem://a")
    await rag.set_document_source([doc.id], "fs:docs")

    async with api:
        resp = await api.delete(f"/databases/docs/documents/{doc.id}")

    assert resp.status_code == 409
    assert "fs:docs" in resp.json()["detail"]
    assert await rag.get_document_by_id(doc.id) is not None


async def test_delete_refuses_an_attachment_of_a_stored_document(api, rag):
    await _import(rag, "mem://parent.pdf")
    child = await _import(rag, "mem://parent.pdf#attachment=a.txt")
    child.metadata = {"parent_uri": "mem://parent.pdf"}
    await writing(rag).document_repository.update_meta(child)

    async with api:
        resp = await api.delete(f"/databases/docs/documents/{child.id}")

    assert resp.status_code == 409
    assert await rag.get_document_by_id(child.id) is not None


async def test_delete_of_a_missing_document_is_not_found(api):
    async with api:
        resp = await api.delete("/databases/docs/documents/missing")

    assert resp.status_code == 404


@pytest.mark.parametrize("method", ["PATCH", "DELETE"])
async def test_edits_need_a_running_app(jobs, sync, method):
    state = APIState(config=AppConfig(), job_repo=jobs, sync_repo=sync)
    async with httpx.AsyncClient(
        transport=ASGITransport(app=build_app(state)), base_url="http://testserver"
    ) as client:
        resp = await client.request(
            method, "/databases/docs/documents/d", json={"title": "x"}
        )

    assert resp.status_code == 503
