import base64
import json

import pytest
from starlette.exceptions import HTTPException

from haiku_rag_app.images import checked_image
from tests.support import image_bytes, png


async def _picture(client, seeded, ref, **params):
    _, ids = seeded
    return await client.get(
        "/api/pictures",
        params={
            "source": "alpha",
            "document_id": ids["alpha"]["document_id"],
            "ref": ref,
            **params,
        },
    )


async def test_a_picture_is_served_with_its_media_type(client, seeded):
    png = await _picture(client, seeded, "#/pictures/0")
    assert png.headers["content-type"] == "image/png"
    assert png.content.startswith(b"\x89PNG")
    jpeg = await _picture(client, seeded, "#/pictures/1")
    assert jpeg.headers["content-type"] == "image/jpeg"


async def test_a_picture_in_no_known_format_is_refused(client, seeded):
    for ref in ("#/pictures/2", "#/pictures/3"):
        response = await _picture(client, seeded, ref)
        assert response.status_code == 415


async def test_a_missing_picture_is_not_found(client, seeded):
    response = await _picture(client, seeded, "#/pictures/9")
    assert response.status_code == 404


async def test_a_picture_needs_its_document_and_ref(client):
    response = await client.get("/api/pictures", params={"source": "alpha"})
    assert response.status_code == 400


@pytest.mark.parametrize("refs", [{}, {"refs": json.dumps(["#/texts/0"])}])
async def test_a_chunk_is_drawn_on_its_page(client, seeded, refs):
    _, ids = seeded
    chunk_id = ids["beta"]["chunk_id"]
    body = (
        await client.get(
            f"/api/visualize/{chunk_id}", params={"source": "beta", **refs}
        )
    ).json()
    assert len(body["images"]) == 1
    assert body["message"] is None


async def test_refs_that_draw_nothing_say_so(client, seeded):
    _, ids = seeded
    chunk_id = ids["beta"]["chunk_id"]
    body = (
        await client.get(
            f"/api/visualize/{chunk_id}",
            params={"source": "beta", "refs": json.dumps(["#/texts/99"])},
        )
    ).json()
    assert body["images"] == []
    assert body["message"] == "No visual grounding available"


async def test_refs_must_be_a_json_list(client, seeded):
    _, ids = seeded
    chunk_id = ids["beta"]["chunk_id"]
    for refs in ("not json", json.dumps({"a": 1})):
        response = await client.get(
            f"/api/visualize/{chunk_id}", params={"source": "beta", "refs": refs}
        )
        assert response.status_code == 400


async def test_a_chunk_of_another_database_is_not_found(client, seeded):
    _, ids = seeded
    response = await client.get(
        f"/api/visualize/{ids['beta']['chunk_id']}", params={"source": "alpha"}
    )
    assert response.status_code == 404


def _halved(raw: bytes) -> str:
    return base64.b64encode(raw[: len(raw) // 2]).decode()


def _broken_load(self):
    raise ValueError("truncated sRGB chunk")


@pytest.mark.parametrize(
    ("value", "declared", "patch", "reason"),
    [
        (5, None, None, "base64"),
        ("not base64!", None, None, "base64"),
        ("é", None, None, "base64"),
        (base64.b64encode(b"hello").decode(), None, None, "not an image"),
        (png(), "image/jpeg", None, "an image is image/png, not image/jpeg"),
        (png(2049, 1), None, None, "2048"),
        (_halved(image_bytes("PNG", 64, 64)), None, None, "not an image"),
        (png()[: len(png()) // 2 // 4 * 4], None, None, "not an image"),
        (png(), None, ("haiku_rag_app.images.MAX_IMAGE_BYTES", 10), "exceeds 10 bytes"),
        (png(), None, ("PIL.Image.MAX_IMAGE_PIXELS", 10), "not an image"),
        (
            png(),
            None,
            ("PIL.PngImagePlugin.PngImageFile.load", _broken_load),
            "not an image",
        ),
    ],
)
def test_an_unacceptable_image_is_refused(monkeypatch, value, declared, patch, reason):
    if patch is not None:
        monkeypatch.setattr(*patch)
    with pytest.raises(HTTPException) as refused:
        checked_image(value, declared)
    assert refused.value.status_code == 400
    assert reason in refused.value.detail
