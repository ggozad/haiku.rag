import json


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


async def test_a_chunk_is_drawn_on_its_page(client, seeded):
    _, ids = seeded
    chunk_id = ids["beta"]["chunk_id"]
    body = (
        await client.get(f"/api/visualize/{chunk_id}", params={"source": "beta"})
    ).json()
    assert len(body["images"]) == 1
    assert body["message"] is None


async def test_a_chunk_is_drawn_from_the_refs_the_model_saw(client, seeded):
    _, ids = seeded
    chunk_id = ids["beta"]["chunk_id"]
    body = (
        await client.get(
            f"/api/visualize/{chunk_id}",
            params={"source": "beta", "refs": json.dumps(["#/texts/0"])},
        )
    ).json()
    assert len(body["images"]) == 1


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
