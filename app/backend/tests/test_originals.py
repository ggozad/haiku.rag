import hashlib
import shutil
from io import BytesIO

from PIL import Image

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models import Chunk
from tests.support import image_bytes, letter_pdf, paged_document, served


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def _store_original(config, source: str, document_id: str, data: bytes):
    async with HaikuRAG(config=config, sources=[source]) as rag:
        await rag.document_repository.store_originals({document_id: data})


async def test_a_chunk_grounding_gives_the_original_and_the_boxes_by_page(
    client, seeded
):
    _, ids = seeded
    body = (
        await client.get(
            f"/api/visualize/{ids['beta']['chunk_id']}", params={"source": "beta"}
        )
    ).json()
    assert body == {
        "source": "beta",
        "document_id": ids["beta"]["document_id"],
        "original": {"sha256": _sha256(letter_pdf(1)), "pdf": True},
        "pages": [
            {
                "page": 1,
                "width": 612.0,
                "height": 792.0,
                "boxes": [
                    {"l": 50.0, "t": 700.0, "r": 550.0, "b": 650.0, "matched": True}
                ],
            }
        ],
        "message": None,
    }


async def test_a_page_the_original_lacks_is_left_out(client, seeded):
    from docling_core.types.doc.base import BoundingBox
    from docling_core.types.doc.document import ProvenanceItem
    from docling_core.types.doc.labels import DocItemLabel

    config, _ = seeded
    doc = paged_document("first")
    doc.add_page(page_no=2, size=doc.pages[1].size)
    doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text="second",
        prov=ProvenanceItem(
            page_no=2, bbox=BoundingBox(l=50, t=700, r=550, b=650), charspan=(0, 6)
        ),
    )
    dim = config.embeddings.model.vector_dim
    async with HaikuRAG(config=config, sources=["alpha"]) as rag:
        document = await rag.import_document(
            doc,
            [
                Chunk(
                    content="first second",
                    embedding=[0.1] * dim,
                    order=0,
                    metadata={
                        "doc_item_refs": ["#/texts/0", "#/texts/1"],
                        "page_numbers": [1, 2],
                    },
                )
            ],
            uri="test://alpha/short-original",
            original=letter_pdf(1),
        )
        assert document.id is not None
        [chunk] = await rag.chunk_repository.get_by_document_id(document.id)

    body = (
        await client.get(f"/api/visualize/{chunk.id}", params={"source": "alpha"})
    ).json()
    assert [page["page"] for page in body["pages"]] == [1]


async def test_refs_drawing_nothing_say_so(client, seeded):
    _, ids = seeded
    body = (
        await client.get(
            f"/api/visualize/{ids['alpha']['chunk_id']}",
            params={"source": "alpha", "refs": '["#/texts/99"]'},
        )
    ).json()
    assert body == {
        "source": "alpha",
        "document_id": None,
        "original": None,
        "pages": [],
        "message": "No visual grounding available",
    }


async def test_a_document_reads_with_its_original_and_pages(client, seeded):
    config, ids = seeded
    async with HaikuRAG(config=config, sources=["alpha"]) as rag:
        bare = await rag.import_document(
            paged_document("no original"), [], uri="test://alpha/bare"
        )
        scan = await rag.import_document(
            paged_document("scan"),
            [],
            uri="test://alpha/scan.png",
            original=image_bytes("PNG", 30, 20),
        )
    base = "/api/inspect/alpha/documents"

    seeded_doc = (await client.get(f"{base}/{ids['alpha']['document_id']}")).json()
    assert seeded_doc["original"] == {"sha256": _sha256(letter_pdf(1)), "pdf": True}
    assert seeded_doc["pages"] == [{"page": 1, "width": 612.0, "height": 792.0}]

    scan_doc = (await client.get(f"{base}/{scan.id}")).json()
    assert scan_doc["original"]["pdf"] is False
    assert scan_doc["pages"] == [{"page": 1, "width": 30.0, "height": 20.0}]

    bare_doc = (await client.get(f"{base}/{bare.id}")).json()
    assert (bare_doc["original"], bare_doc["pages"]) == (None, [])


async def test_the_document_and_its_original_read_without_the_structure(
    client, seeded, monkeypatch
):
    from haiku.rag.store.repositories.document import DocumentRepository

    async def no_structure(self, entity_id, with_original=False):
        raise AssertionError("the structure blob was read")

    monkeypatch.setattr(DocumentRepository, "get_docling_data", no_structure)
    _, ids = seeded
    base = f"/api/inspect/alpha/documents/{ids['alpha']['document_id']}"
    params = {"sha256": _sha256(letter_pdf(1))}

    assert (await client.get(base)).status_code == 200
    assert (await client.get(f"{base}/original", params=params)).status_code == 200


async def test_the_original_pdf_is_served_from_the_selected_database(tmp_path, seeded):
    config, ids = seeded
    document_id = ids["alpha"]["document_id"]
    alpha, beta = config.lancedb.databases["alpha"], config.lancedb.databases["beta"]
    shutil.rmtree(beta)
    shutil.copytree(alpha, beta)
    await _store_original(config, "beta", document_id, letter_pdf(2))

    async with served(config, tmp_path / "sessions.db") as (_, client):
        for source, original in (("alpha", letter_pdf(1)), ("beta", letter_pdf(2))):
            response = await client.get(
                f"/api/inspect/{source}/documents/{document_id}/original",
                params={"sha256": _sha256(original)},
            )
            assert response.status_code == 200
            assert response.content == original
            assert response.headers["content-type"] == "application/pdf"
            assert "immutable" in response.headers["cache-control"]


async def test_the_original_answers_byte_ranges(client, seeded, monkeypatch):
    import lancedb

    async def whole(*_args, **_kwargs):
        raise AssertionError("the whole original was read")

    monkeypatch.setattr(lancedb.AsyncTable, "fetch_blobs", whole)
    _, ids = seeded
    pdf = letter_pdf(1)
    size = len(pdf)
    url = f"/api/inspect/alpha/documents/{ids['alpha']['document_id']}/original"
    params = {"sha256": _sha256(pdf)}

    for header, start, end in (
        ("bytes=10-19", 10, 19),
        (f"bytes={size - 7}-", size - 7, size - 1),
        ("bytes=-5", size - 5, size - 1),
        (f"bytes=10-{size + 100}", 10, size - 1),
    ):
        response = await client.get(url, params=params, headers={"Range": header})
        assert response.status_code == 206, header
        assert response.content == pdf[start : end + 1], header
        assert response.headers["content-range"] == f"bytes {start}-{end}/{size}"
        assert response.headers["accept-ranges"] == "bytes"
        assert "immutable" in response.headers["cache-control"]

    whole_file = await client.get(url, params=params)
    assert whole_file.status_code == 200
    assert whole_file.content == pdf
    assert whole_file.headers["content-length"] == str(size)
    assert whole_file.headers["accept-ranges"] == "bytes"

    for beyond in (f"bytes={size}-", "bytes=-0"):
        response = await client.get(url, params=params, headers={"Range": beyond})
        assert response.status_code == 416, beyond
        assert response.headers["content-range"] == f"bytes */{size}"

    for unread in ("bytes=5-2", "lines=1-2", "bytes=1-2,4-5", "bytes=-"):
        response = await client.get(url, params=params, headers={"Range": unread})
        assert response.status_code == 200, unread
        assert response.content == pdf


async def test_an_original_is_read_a_chunk_at_a_time(client, seeded, monkeypatch):
    from haiku.rag.store.repositories.document import StoredOriginal
    from haiku_rag_app import inspect

    monkeypatch.setattr(inspect, "_CHUNK", 256)
    reads: list[int] = []
    read = StoredOriginal.read

    async def recorded(self, offset, length):
        reads.append(length)
        return await read(self, offset, length)

    monkeypatch.setattr(StoredOriginal, "read", recorded)
    _, ids = seeded
    pdf = letter_pdf(1)
    url = f"/api/inspect/alpha/documents/{ids['alpha']['document_id']}/original"
    params = {"sha256": _sha256(pdf)}

    for headers, body in (({"Range": "bytes=3-"}, pdf[3:]), ({}, pdf)):
        reads.clear()
        response = await client.get(url, params=params, headers=headers)
        assert response.content == body
        assert response.headers["content-length"] == str(len(body))
        # The first read checks the PDF header.
        assert len(reads) > 2 and max(reads[1:]) <= 256


async def test_a_replaced_original_answers_conflict(tmp_path, seeded):
    config, ids = seeded
    config.lancedb.read_consistency_interval_seconds = 0
    document_id, chunk_id = ids["alpha"]["document_id"], ids["alpha"]["chunk_id"]
    async with served(config, tmp_path / "sessions.db") as (_, client):
        grounding = (
            await client.get(f"/api/visualize/{chunk_id}", params={"source": "alpha"})
        ).json()
        await _store_original(config, "alpha", document_id, letter_pdf(2))

        base = f"/api/inspect/alpha/documents/{document_id}"
        params = {"sha256": grounding["original"]["sha256"]}
        original = await client.get(f"{base}/original", params=params)
        assert original.status_code == 409
        ranged = await client.get(
            f"{base}/original", params=params, headers={"Range": "bytes=0-9"}
        )
        assert ranged.status_code == 409
        assert (await client.get(f"{base}/pages/1", params=params)).status_code == 409


async def test_an_original_needs_its_sha256(client, seeded):
    _, ids = seeded
    base = f"/api/inspect/alpha/documents/{ids['alpha']['document_id']}"
    assert (await client.get(f"{base}/original")).status_code == 400
    assert (await client.get(f"{base}/pages/1")).status_code == 400


async def test_only_a_stored_pdf_is_served_as_the_original(client, seeded):
    config, _ = seeded
    scan_bytes = image_bytes("PNG", 30, 20)
    async with HaikuRAG(config=config, sources=["alpha"]) as rag:
        bare = await rag.import_document(
            paged_document("no original"), [], uri="test://alpha/bare"
        )
        scan = await rag.import_document(
            paged_document("scan"), [], uri="test://alpha/scan.png", original=scan_bytes
        )
    base = "/api/inspect/alpha/documents"
    for document_id, sha256 in (
        (scan.id, _sha256(scan_bytes)),
        (bare.id, "x"),
        ("nope", "x"),
    ):
        response = await client.get(
            f"{base}/{document_id}/original", params={"sha256": sha256}
        )
        assert response.status_code == 404
    page = await client.get(f"{base}/{bare.id}/pages/1", params={"sha256": "x"})
    assert page.status_code == 404


async def test_an_image_page_renders_plain(client, seeded):
    config, _ = seeded
    scan_bytes = image_bytes("PNG", 30, 20)
    async with HaikuRAG(config=config, sources=["alpha"]) as rag:
        scan = await rag.import_document(
            paged_document("scan"), [], uri="test://alpha/scan.png", original=scan_bytes
        )
    base = f"/api/inspect/alpha/documents/{scan.id}/pages"
    params = {"sha256": _sha256(scan_bytes)}

    response = await client.get(f"{base}/1", params=params)
    assert response.headers["content-type"] == "image/png"
    with Image.open(BytesIO(response.content)) as image:
        assert (image.size, image.getpixel((0, 0))) == ((30, 20), (255, 0, 0))
    assert "immutable" in response.headers["cache-control"]
    assert (await client.get(f"{base}/2", params=params)).status_code == 404
    assert (await client.get(f"{base}/one", params=params)).status_code == 400


async def test_only_an_image_original_renders_its_pages(client, seeded):
    _, ids = seeded
    response = await client.get(
        f"/api/inspect/alpha/documents/{ids['alpha']['document_id']}/pages/1",
        params={"sha256": _sha256(letter_pdf(1))},
    )
    assert response.status_code == 404
