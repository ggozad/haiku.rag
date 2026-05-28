import io

import pypdfium2 as pdfium

from haiku.rag.client import HaikuRAG
from haiku.rag.client.documents import (
    MAX_ATTACHMENT_DEPTH,
    _reconcile_pdf_attachments,
    parent_uri_filter,
)
from haiku.rag.ingester.sources import FetchResult
from haiku.rag.store.models.document import Document


def build_pdf(attachments: list[tuple[str, bytes]]) -> bytes:
    """Build a minimal one-page PDF with the given (name, bytes) attachments."""
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(200, 200)
    for name, data in attachments:
        att = pdf.new_attachment(name)
        att.set_data(data)
    buf = io.BytesIO()
    pdf.save(buf)
    return buf.getvalue()


async def fake_ingest_fetch_result(
    client,
    result: FetchResult,
    *,
    title,
    user_metadata,
    stored_uri,
    existing_doc,
    depth=0,
):
    """A stand-in for ``_ingest_fetch_result`` that skips docling/embedder
    entirely: it writes the document with content_type/md5/parent_uri set
    correctly, then defers to the real ``_reconcile_pdf_attachments`` so
    recursive logic stays under test."""
    final_metadata = {
        **(user_metadata or {}),
        "content_type": result.content_type,
        "md5": result.content_hash,
        **result.extra_metadata,
    }
    if result.revision is not None:
        final_metadata["source_revision"] = result.revision

    if existing_doc:
        existing_doc.content = ""
        existing_doc.metadata = final_metadata
        if title is not None:
            existing_doc.title = title
        doc = await client.document_repository.update(existing_doc)
    else:
        doc = await client.document_repository.create(
            Document(
                content="",
                uri=stored_uri,
                title=title,
                metadata=final_metadata,
            )
        )
    await _reconcile_pdf_attachments(client, doc, result.body, depth=depth)
    return doc


async def _make_parent(
    client: HaikuRAG,
    uri: str,
    body: bytes,
    *,
    content_type: str = "application/pdf",
) -> Document:
    """Insert a parent Document directly + invoke reconciliation for its body."""
    import hashlib

    md5 = hashlib.md5(body, usedforsecurity=False).hexdigest()
    parent = await client.document_repository.create(
        Document(
            content="",
            uri=uri,
            metadata={"content_type": content_type, "md5": md5},
        )
    )
    return parent


async def test_first_ingest_creates_one_doc_per_attachment(temp_db_path, monkeypatch):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    leaf_pdf = build_pdf([])
    pdf_bytes = build_pdf([("a.pdf", leaf_pdf), ("notes.txt", b"plain text payload")])
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///fixtures/parent.pdf"
        parent = await _make_parent(client, parent_uri, pdf_bytes)

        await _reconcile_pdf_attachments(client, parent, pdf_bytes, depth=0)

        children = await client.list_documents(filter=parent_uri_filter(parent_uri))
        assert len(children) == 2
        by_uri = {c.uri: c for c in children}
        assert f"{parent_uri}#attachment=a.pdf" in by_uri
        assert f"{parent_uri}#attachment=notes.txt" in by_uri

        a = by_uri[f"{parent_uri}#attachment=a.pdf"]
        assert a.metadata["parent_uri"] == parent_uri
        assert a.metadata["content_type"] == "application/pdf"
        assert "md5" in a.metadata

        txt = by_uri[f"{parent_uri}#attachment=notes.txt"]
        assert txt.metadata["content_type"] == "text/plain"


async def test_attachment_with_spaces_in_name_is_percent_encoded(
    temp_db_path, monkeypatch
):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    pdf_bytes = build_pdf([("memo with spaces.pdf", b"payload")])
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///fixtures/parent.pdf"
        parent = await _make_parent(client, parent_uri, pdf_bytes)
        await _reconcile_pdf_attachments(client, parent, pdf_bytes, depth=0)

        children = await client.list_documents(filter=parent_uri_filter(parent_uri))
        assert len(children) == 1
        assert children[0].uri == f"{parent_uri}#attachment=memo%20with%20spaces.pdf"


async def test_reingest_removes_dropped_attachment(temp_db_path, monkeypatch):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///fixtures/parent.pdf"
        first = build_pdf([("a.txt", b"A"), ("b.txt", b"B")])
        parent = await _make_parent(client, parent_uri, first)
        await _reconcile_pdf_attachments(client, parent, first, depth=0)
        assert (
            len(await client.list_documents(filter=parent_uri_filter(parent_uri))) == 2
        )

        second = build_pdf([("a.txt", b"A")])
        await _reconcile_pdf_attachments(client, parent, second, depth=0)
        remaining = await client.list_documents(filter=parent_uri_filter(parent_uri))
        assert len(remaining) == 1
        assert remaining[0].uri == f"{parent_uri}#attachment=a.txt"


async def test_reingest_updates_changed_attachment_in_place(temp_db_path, monkeypatch):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///fixtures/parent.pdf"
        first = build_pdf([("a.txt", b"original")])
        parent = await _make_parent(client, parent_uri, first)
        await _reconcile_pdf_attachments(client, parent, first, depth=0)
        before = (await client.list_documents(filter=parent_uri_filter(parent_uri)))[0]

        second = build_pdf([("a.txt", b"different")])
        await _reconcile_pdf_attachments(client, parent, second, depth=0)
        after = (await client.list_documents(filter=parent_uri_filter(parent_uri)))[0]

        assert after.id == before.id
        assert after.metadata["md5"] != before.metadata["md5"]


async def test_reingest_adds_new_attachment(temp_db_path, monkeypatch):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///fixtures/parent.pdf"
        first = build_pdf([("a.txt", b"A")])
        parent = await _make_parent(client, parent_uri, first)
        await _reconcile_pdf_attachments(client, parent, first, depth=0)

        second = build_pdf([("a.txt", b"A"), ("c.txt", b"C")])
        await _reconcile_pdf_attachments(client, parent, second, depth=0)
        children = await client.list_documents(filter=parent_uri_filter(parent_uri))
        assert len(children) == 2
        names = {c.uri for c in children}
        assert f"{parent_uri}#attachment=a.txt" in names
        assert f"{parent_uri}#attachment=c.txt" in names


async def test_nested_pdf_attachments_recurse_up_to_cap(temp_db_path, monkeypatch):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    # Build a 4-deep chain: root -> L1 -> L2 -> L3. MAX_ATTACHMENT_DEPTH=3
    # means root + L1 + L2 ingested (3 PDFs total); L3 is skipped at the cap.
    assert MAX_ATTACHMENT_DEPTH == 3
    l3 = build_pdf([("leaf.txt", b"deepest")])
    l2 = build_pdf([("l3.pdf", l3)])
    l1 = build_pdf([("l2.pdf", l2)])
    root = build_pdf([("l1.pdf", l1)])

    async with HaikuRAG(temp_db_path, create=True) as client:
        root_uri = "file:///fixtures/root.pdf"
        parent = await _make_parent(client, root_uri, root)
        await _reconcile_pdf_attachments(client, parent, root, depth=0)

        l1_uri = f"{root_uri}#attachment=l1.pdf"
        l2_uri = f"{l1_uri}#attachment=l2.pdf"
        l3_uri = f"{l2_uri}#attachment=l3.pdf"

        assert await client.get_document_by_uri(l1_uri) is not None
        assert await client.get_document_by_uri(l2_uri) is not None
        assert await client.get_document_by_uri(l3_uri) is None


async def test_config_off_skips_extraction(temp_db_path, monkeypatch):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    async with HaikuRAG(temp_db_path, create=True) as client:
        monkeypatch.setattr(client._config.processing, "extract_pdf_attachments", False)
        parent_uri = "file:///fixtures/parent.pdf"
        pdf_bytes = build_pdf([("a.txt", b"A")])
        parent = await _make_parent(client, parent_uri, pdf_bytes)
        await _reconcile_pdf_attachments(client, parent, pdf_bytes, depth=0)

        assert await client.list_documents(filter=parent_uri_filter(parent_uri)) == []


async def test_non_pdf_parent_is_ignored(temp_db_path, monkeypatch):
    """A non-PDF document with a PDF blob would be a logic bug, but the helper
    must short-circuit on content_type alone — never call pypdfium2."""
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///fixtures/parent.txt"
        parent = await _make_parent(
            client,
            parent_uri,
            b"not a pdf",
            content_type="text/plain",
        )
        # Even with PDF bytes, content_type=text/plain blocks extraction.
        pdf_bytes = build_pdf([("a.txt", b"A")])
        await _reconcile_pdf_attachments(client, parent, pdf_bytes, depth=0)
        assert await client.list_documents(filter=parent_uri_filter(parent_uri)) == []


async def test_parent_without_uri_is_skipped(temp_db_path, monkeypatch):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent = Document(
            content="",
            uri=None,
            metadata={"content_type": "application/pdf", "md5": "abc"},
        )
        pdf_bytes = build_pdf([("a.txt", b"A")])
        await _reconcile_pdf_attachments(client, parent, pdf_bytes, depth=0)
        assert await client.list_documents() == []


async def test_malformed_pdf_logs_warning_and_skips(temp_db_path, monkeypatch, caplog):
    """A non-PDF body labelled as application/pdf must not crash the helper —
    pypdfium2's open raises PdfiumError, which we log and return from."""
    import logging

    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///fixtures/junk.pdf"
        garbage = b"this is not a pdf at all"
        parent = await _make_parent(client, parent_uri, garbage)
        with caplog.at_level(logging.WARNING, logger="haiku.rag.client.documents"):
            await _reconcile_pdf_attachments(client, parent, garbage, depth=0)
        assert await client.list_documents(filter=parent_uri_filter(parent_uri)) == []


async def test_unsupported_attachment_continues_loop(temp_db_path, monkeypatch):
    """One attachment whose ingest raises UnsupportedSourceError must not
    prevent siblings from being ingested. The unsupported attachment is
    skipped with a warning; the others land."""
    from haiku.rag.client.exceptions import UnsupportedSourceError

    async def picky_fake(
        client,
        result,
        *,
        title,
        user_metadata,
        stored_uri,
        existing_doc,
        depth=0,
    ):
        if stored_uri.endswith("unsupported.xyz"):
            raise UnsupportedSourceError("nope")
        return await fake_ingest_fetch_result(
            client,
            result,
            title=title,
            user_metadata=user_metadata,
            stored_uri=stored_uri,
            existing_doc=existing_doc,
            depth=depth,
        )

    monkeypatch.setattr("haiku.rag.client.documents._ingest_fetch_result", picky_fake)
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///fixtures/parent.pdf"
        pdf_bytes = build_pdf([("ok.txt", b"keep me"), ("unsupported.xyz", b"data")])
        parent = await _make_parent(client, parent_uri, pdf_bytes)
        await _reconcile_pdf_attachments(client, parent, pdf_bytes, depth=0)
        children = await client.list_documents(filter=parent_uri_filter(parent_uri))
        assert {c.uri for c in children} == {f"{parent_uri}#attachment=ok.txt"}


async def test_cascade_delete_removes_reconciled_children(temp_db_path, monkeypatch):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///fixtures/parent.pdf"
        pdf_bytes = build_pdf([("a.txt", b"A"), ("b.txt", b"B")])
        parent = await _make_parent(client, parent_uri, pdf_bytes)
        await _reconcile_pdf_attachments(client, parent, pdf_bytes, depth=0)
        assert len(await client.list_documents()) == 3

        await client.delete_document(parent.id)
        assert await client.list_documents() == []


async def test_create_document_from_source_extracts_attachments(
    tmp_path, temp_db_path, monkeypatch
):
    """The full create_document_from_source path — the same entry point the
    ingester worker uses for an UPSERT job — produces parent + child docs
    when the source is a PDF with embedded files on disk."""
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    pdf_path = tmp_path / "parent.pdf"
    pdf_path.write_bytes(
        build_pdf([("notes.txt", b"plain text"), ("data.txt", b"more data")])
    )

    async with HaikuRAG(temp_db_path, create=True) as client:
        parent = await client.create_document_from_source(pdf_path)

    async with HaikuRAG(temp_db_path) as client:
        assert isinstance(parent, Document)
        children = await client.list_documents(filter=parent_uri_filter(parent.uri))
        assert len(children) == 2
        assert {c.metadata["parent_uri"] for c in children} == {parent.uri}


async def test_create_document_from_source_reingest_after_attachment_edit(
    tmp_path, temp_db_path, monkeypatch
):
    """Mutate the parent PDF's attachments and re-ingest. The md5 short-circuit
    must NOT fire (parent bytes changed); reconciliation diffs children to add,
    update, and delete in one pass while leaving unrelated children untouched."""
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    pdf_path = tmp_path / "parent.pdf"
    pdf_path.write_bytes(
        build_pdf(
            [
                ("stable.txt", b"unchanged across runs"),
                ("changed.txt", b"old contents"),
                ("removed.txt", b"goes away"),
            ]
        )
    )

    async with HaikuRAG(temp_db_path, create=True) as client:
        parent = await client.create_document_from_source(pdf_path)
        assert isinstance(parent, Document)
        before_children = {
            c.uri: c
            for c in await client.list_documents(filter=parent_uri_filter(parent.uri))
        }
        assert set(before_children) == {
            f"{parent.uri}#attachment=stable.txt",
            f"{parent.uri}#attachment=changed.txt",
            f"{parent.uri}#attachment=removed.txt",
        }
        stable_id_before = before_children[f"{parent.uri}#attachment=stable.txt"].id
        changed_id_before = before_children[f"{parent.uri}#attachment=changed.txt"].id

    pdf_path.write_bytes(
        build_pdf(
            [
                ("stable.txt", b"unchanged across runs"),
                ("changed.txt", b"new contents"),
                ("added.txt", b"brand new"),
            ]
        )
    )

    async with HaikuRAG(temp_db_path) as client:
        await client.create_document_from_source(pdf_path)
        after_children = {
            c.uri: c
            for c in await client.list_documents(filter=parent_uri_filter(parent.uri))
        }

        assert set(after_children) == {
            f"{parent.uri}#attachment=stable.txt",
            f"{parent.uri}#attachment=changed.txt",
            f"{parent.uri}#attachment=added.txt",
        }
        stable = after_children[f"{parent.uri}#attachment=stable.txt"]
        changed = after_children[f"{parent.uri}#attachment=changed.txt"]
        assert stable.id == stable_id_before
        assert changed.id == changed_id_before
        assert (
            stable.metadata["md5"]
            == before_children[f"{parent.uri}#attachment=stable.txt"].metadata["md5"]
        )
        assert (
            changed.metadata["md5"]
            != before_children[f"{parent.uri}#attachment=changed.txt"].metadata["md5"]
        )


async def test_create_document_from_source_delete_cascades(
    tmp_path, temp_db_path, monkeypatch
):
    """The ingester worker's DELETE path is just client.delete_document(doc.id).
    A parent ingested via the full pipeline must cascade to its children when
    that path runs — mirrors what happens when a watched file is removed."""
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    pdf_path = tmp_path / "parent.pdf"
    pdf_path.write_bytes(build_pdf([("a.txt", b"A"), ("b.txt", b"B")]))

    async with HaikuRAG(temp_db_path, create=True) as client:
        parent = await client.create_document_from_source(pdf_path)
        assert isinstance(parent, Document)
        assert len(await client.list_documents()) == 3

        await client.delete_document(parent.id)
        assert await client.list_documents() == []
