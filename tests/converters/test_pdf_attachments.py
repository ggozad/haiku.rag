import io
import threading

import pypdfium2 as pdfium

from haiku.rag.client import HaikuRAG
from haiku.rag.client.documents import (
    MAX_ATTACHMENT_DEPTH,
    _extract_pdf_attachments,
    _reconcile_pdf_attachments,
    parent_uri_filter,
)
from haiku.rag.sources import FetchResult
from haiku.rag.store.models.document import Document
from tests.conftest import writing


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
    session,
    result: FetchResult,
    *,
    title,
    user_metadata,
    stored_uri,
    existing_doc,
    source_id=None,
    depth=0,
    filename=None,
    metadata_provider=None,
    provider_source_id=None,
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
    if source_id is not None:
        final_metadata["source_id"] = source_id

    if existing_doc:
        existing_doc.content = ""
        existing_doc.metadata = final_metadata
        if title is not None:
            existing_doc.title = title
        doc = await session.document_repository.update(existing_doc)
    else:
        doc = await session.document_repository.create(
            Document(
                content="",
                uri=stored_uri,
                title=title,
                metadata=final_metadata,
            )
        )
    await _reconcile_pdf_attachments(
        session,
        doc,
        result.body,
        depth=depth,
        metadata_provider=metadata_provider,
        provider_source_id=provider_source_id,
    )
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

        await _reconcile_pdf_attachments(writing(client), parent, pdf_bytes, depth=0)

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
        await _reconcile_pdf_attachments(writing(client), parent, pdf_bytes, depth=0)

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
        await _reconcile_pdf_attachments(writing(client), parent, first, depth=0)
        assert (
            len(await client.list_documents(filter=parent_uri_filter(parent_uri))) == 2
        )

        second = build_pdf([("a.txt", b"A")])
        await _reconcile_pdf_attachments(writing(client), parent, second, depth=0)
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
        await _reconcile_pdf_attachments(writing(client), parent, first, depth=0)
        before = (await client.list_documents(filter=parent_uri_filter(parent_uri)))[0]

        second = build_pdf([("a.txt", b"different")])
        await _reconcile_pdf_attachments(writing(client), parent, second, depth=0)
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
        await _reconcile_pdf_attachments(writing(client), parent, first, depth=0)

        second = build_pdf([("a.txt", b"A"), ("c.txt", b"C")])
        await _reconcile_pdf_attachments(writing(client), parent, second, depth=0)
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
        await _reconcile_pdf_attachments(writing(client), parent, root, depth=0)

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
        await _reconcile_pdf_attachments(writing(client), parent, pdf_bytes, depth=0)

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
        await _reconcile_pdf_attachments(writing(client), parent, pdf_bytes, depth=0)
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
        await _reconcile_pdf_attachments(writing(client), parent, pdf_bytes, depth=0)
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
            await _reconcile_pdf_attachments(writing(client), parent, garbage, depth=0)
        assert await client.list_documents(filter=parent_uri_filter(parent_uri)) == []


async def test_unsupported_attachment_continues_loop(temp_db_path, monkeypatch):
    """An attachment whose extension the converter does not support must not
    prevent siblings from being ingested. It is skipped with a warning before
    any ingest; the others land."""
    ingested: list[str] = []

    async def recording_fake(session, result, **kwargs):
        ingested.append(result.uri)
        return await fake_ingest_fetch_result(session, result, **kwargs)

    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result", recording_fake
    )
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///fixtures/parent.pdf"
        pdf_bytes = build_pdf([("ok.txt", b"keep me"), ("unsupported.xyz", b"data")])
        parent = await _make_parent(client, parent_uri, pdf_bytes)
        await _reconcile_pdf_attachments(writing(client), parent, pdf_bytes, depth=0)
        children = await client.list_documents(filter=parent_uri_filter(parent_uri))
        assert {c.uri for c in children} == {f"{parent_uri}#attachment=ok.txt"}
        assert ingested == [f"{parent_uri}#attachment=ok.txt"]


async def test_attachment_rejected_during_ingest_continues_loop(
    temp_db_path, monkeypatch
):
    """An attachment with a supported extension can still be rejected while it
    is ingested (a .pdf pypdfium2 cannot open for slicing). Its siblings must
    still be ingested."""
    from haiku.rag.client.exceptions import UnsupportedSourceError

    async def picky_fake(session, result, **kwargs):
        if result.uri.endswith("broken.pdf"):
            raise UnsupportedSourceError("pypdfium2 cannot open PDF")
        return await fake_ingest_fetch_result(session, result, **kwargs)

    monkeypatch.setattr("haiku.rag.client.documents._ingest_fetch_result", picky_fake)
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///fixtures/parent.pdf"
        pdf_bytes = build_pdf([("ok.txt", b"keep me"), ("broken.pdf", b"data")])
        parent = await _make_parent(client, parent_uri, pdf_bytes)
        await _reconcile_pdf_attachments(writing(client), parent, pdf_bytes, depth=0)
        children = await client.list_documents(filter=parent_uri_filter(parent_uri))
        assert {c.uri for c in children} == {f"{parent_uri}#attachment=ok.txt"}


async def test_joboptions_attachment_skipped_not_routed_as_pdf(temp_db_path, caplog):
    """An attachment's own name drives its extension, not the synthetic
    ``...#attachment=<name>`` URI (whose fragment the URL-suffix fallback drops,
    inheriting the parent's ``.pdf``). ``.joboptions`` is unsupported, so the
    child is skipped before any converter/embedder call (real ingest, no fake)."""
    import logging

    pdf_bytes = build_pdf([("Press Quality.joboptions", b"/CompressObjects /Tags\n")])
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///fixtures/brochure.pdf"
        parent = await _make_parent(client, parent_uri, pdf_bytes)

        with caplog.at_level(logging.WARNING, logger="haiku.rag.client.documents"):
            await _reconcile_pdf_attachments(
                writing(client), parent, pdf_bytes, depth=0
            )

        assert await client.list_documents(filter=parent_uri_filter(parent_uri)) == []


async def test_cascade_delete_removes_reconciled_children(temp_db_path, monkeypatch):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///fixtures/parent.pdf"
        pdf_bytes = build_pdf([("a.txt", b"A"), ("b.txt", b"B")])
        parent = await _make_parent(client, parent_uri, pdf_bytes)
        await _reconcile_pdf_attachments(writing(client), parent, pdf_bytes, depth=0)
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


async def test_attachment_children_carry_no_source_id(
    tmp_path, temp_db_path, monkeypatch
):
    """Only the fetched document is attributed; derived children are not."""
    from haiku.rag.sources.fs import FSSource

    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    pdf_path = tmp_path / "parent.pdf"
    pdf_path.write_bytes(build_pdf([("notes.txt", b"plain text")]))
    source = FSSource(root=tmp_path, source_id="fs:attachments")

    async with HaikuRAG(temp_db_path, create=True) as client:
        parent = await client.create_document_from_source(
            pdf_path, sources=[source], source_id=source.source_id
        )

    async with HaikuRAG(temp_db_path) as client:
        assert isinstance(parent, Document)
        assert parent.metadata["source_id"] == source.source_id
        children = await client.list_documents(filter=parent_uri_filter(parent.uri))
        assert len(children) == 1
        assert "source_id" not in children[0].metadata


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


async def test_extract_pdf_attachments_called_off_event_loop_thread(
    temp_db_path, monkeypatch
):
    """_extract_pdf_attachments must run in a thread-pool thread, not on the
    event-loop thread. A synchronous call would freeze the event loop for the
    duration of pdfium I/O, stalling every other concurrent worker.

    We verify this by capturing the thread identity inside a spy wrapper: if
    asyncio.to_thread is used correctly the spy runs off the event-loop thread."""
    event_loop_thread = threading.current_thread()
    called_from: list[threading.Thread] = []

    def spy(body, uri, *, depth):
        called_from.append(threading.current_thread())
        return _extract_pdf_attachments(body, uri, depth=depth)

    monkeypatch.setattr("haiku.rag.client.documents._extract_pdf_attachments", spy)
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )

    pdf_bytes = build_pdf([("a.txt", b"payload")])
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///fixtures/parent.pdf"
        parent = await _make_parent(client, parent_uri, pdf_bytes)
        await _reconcile_pdf_attachments(writing(client), parent, pdf_bytes, depth=0)

    assert called_from, "_extract_pdf_attachments was never called"
    assert called_from[0] is not event_loop_thread, (
        "_extract_pdf_attachments ran on the event-loop thread; "
        "it must be dispatched via asyncio.to_thread to avoid blocking the loop"
    )


async def _two_lookalike_parents(client: HaikuRAG, underscored_uri: str):
    """Parents at `underscored_uri` and its `_`-to-`-` lookalike, one attachment each."""
    pdf_bytes = build_pdf([("a.txt", b"A")])
    underscored = await _make_parent(client, underscored_uri, pdf_bytes)
    await _reconcile_pdf_attachments(writing(client), underscored, pdf_bytes, depth=0)
    hyphenated_uri = underscored_uri.replace("_", "-")
    hyphenated = await _make_parent(client, hyphenated_uri, pdf_bytes)
    await _reconcile_pdf_attachments(writing(client), hyphenated, pdf_bytes, depth=0)
    return underscored, f"{hyphenated_uri}#attachment=a.txt"


async def test_reingest_leaves_a_lookalike_parents_attachments(
    temp_db_path, monkeypatch
):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent, lookalike_child = await _two_lookalike_parents(
            client, "file:///fixtures/report_1.pdf"
        )

        second = build_pdf([("b.txt", b"B")])
        await _reconcile_pdf_attachments(writing(client), parent, second, depth=0)

        assert (
            await client.get_document_by_uri(f"{parent.uri}#attachment=a.txt") is None
        )
        assert await client.get_document_by_uri(lookalike_child) is not None


async def test_cascade_delete_leaves_a_lookalike_parents_attachments(
    temp_db_path, monkeypatch
):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent, lookalike_child = await _two_lookalike_parents(
            client, "file:///fixtures/report_1.pdf"
        )

        await client.delete_document(parent.id)

        assert (
            await client.get_document_by_uri(f"{parent.uri}#attachment=a.txt") is None
        )
        assert await client.get_document_by_uri(lookalike_child) is not None


async def test_cascade_delete_of_a_percent_uri_leaves_other_parents_attachments(
    temp_db_path, monkeypatch
):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    async with HaikuRAG(temp_db_path, create=True) as client:
        pdf_bytes = build_pdf([("a.txt", b"A")])
        other = await _make_parent(client, "file:///fixtures/other.pdf", pdf_bytes)
        await _reconcile_pdf_attachments(writing(client), other, pdf_bytes, depth=0)
        percent = await _make_parent(client, "file:///fixtures/%.pdf", pdf_bytes)
        await _reconcile_pdf_attachments(writing(client), percent, pdf_bytes, depth=0)

        await client.delete_document(percent.id)

        assert (
            await client.get_document_by_uri("file:///fixtures/%.pdf#attachment=a.txt")
            is None
        )
        assert (
            await client.get_document_by_uri(
                "file:///fixtures/other.pdf#attachment=a.txt"
            )
            is not None
        )


# --- metadata providers -------------------------------------------------------


class _RecordingProvider:
    """Provider double: records each call, returns ``metadata`` plus the call's
    URI, and raises for any URI ending in one of ``fail_for``."""

    def __init__(self, metadata: dict | None = None, *, fail_for: tuple = ()):
        self.metadata = metadata or {}
        self.fail_for = fail_for
        self.calls: list[tuple[str, str, FetchResult]] = []

    async def __call__(self, source_id: str, uri: str, result: FetchResult) -> dict:
        self.calls.append((source_id, uri, result))
        if self.fail_for and uri.endswith(self.fail_for):
            raise RuntimeError(f"provider failed for {uri}")
        return {**self.metadata, "seen_uri": uri}

    def attachment_calls(self) -> list[tuple[str, str, FetchResult]]:
        return [c for c in self.calls if "parent_uri" in c[2].extra_metadata]


async def _ingest_with_provider(tmp_path, client, pdf_path, provider) -> Document:
    from haiku.rag.sources.fs import FSSource

    source = FSSource(root=tmp_path, source_id="fs:attachments")
    parent = await client.create_document_from_source(
        pdf_path,
        sources=[source],
        source_id=source.source_id,
        metadata_provider=provider,
    )
    assert isinstance(parent, Document)
    return parent


async def _reconcile_with_provider(
    client: HaikuRAG, parent: Document, body: bytes, provider
) -> None:
    await _reconcile_pdf_attachments(
        writing(client),
        parent,
        body,
        depth=0,
        metadata_provider=provider,
        provider_source_id="src",
    )


async def test_provider_called_once_per_attachment_with_parent_source_id(
    tmp_path, temp_db_path, monkeypatch
):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    pdf_path = tmp_path / "parent.pdf"
    pdf_path.write_bytes(build_pdf([("a.txt", b"A"), ("b.txt", b"B")]))
    provider = _RecordingProvider({"tag": "x"})

    async with HaikuRAG(temp_db_path, create=True) as client:
        parent = await _ingest_with_provider(tmp_path, client, pdf_path, provider)
        children = {
            c.uri: c
            for c in await client.list_documents(filter=parent_uri_filter(parent.uri))
        }

    a_uri = f"{parent.uri}#attachment=a.txt"
    b_uri = f"{parent.uri}#attachment=b.txt"
    assert len(provider.calls) == 3
    assert [(s, u, r.body) for s, u, r in provider.attachment_calls()] == [
        ("fs:attachments", a_uri, b"A"),
        ("fs:attachments", b_uri, b"B"),
    ]
    for _, uri, result in provider.attachment_calls():
        assert result.uri == uri
        assert result.content_type == "text/plain"
        assert result.extra_metadata == {"parent_uri": parent.uri}
        assert result.disk_path is None
        assert result.revision is None
    assert children[a_uri].metadata["tag"] == "x"
    assert children[a_uri].metadata["seen_uri"] == a_uri
    assert children[b_uri].metadata["seen_uri"] == b_uri
    assert parent.metadata["seen_uri"] == str(pdf_path)


async def test_attachment_children_carry_no_source_id_under_a_provider(
    tmp_path, temp_db_path, monkeypatch
):
    """The provider runs on the parent's behalf; ownership is not passed down,
    so orphan reconciliation never sees a child (see test_reconcile)."""
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    pdf_path = tmp_path / "parent.pdf"
    pdf_path.write_bytes(build_pdf([("notes.txt", b"plain text")]))

    async with HaikuRAG(temp_db_path, create=True) as client:
        parent = await _ingest_with_provider(
            tmp_path, client, pdf_path, _RecordingProvider()
        )
        (child,) = await client.list_documents(filter=parent_uri_filter(parent.uri))

    assert parent.metadata["source_id"] == "fs:attachments"
    assert "source_id" not in child.metadata
    assert "seen_uri" in child.metadata


async def test_provider_reserved_keys_do_not_reach_a_child(temp_db_path, monkeypatch):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    provider = _RecordingProvider(
        {
            "source_id": "forged",
            "md5": "forged",
            "content_type": "forged",
            "source_revision": "forged",
            "parent_uri": "file:///elsewhere.pdf",
            "kept": "yes",
        }
    )
    pdf_bytes = build_pdf([("a.txt", b"A")])
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///fixtures/parent.pdf"
        parent = await _make_parent(client, parent_uri, pdf_bytes)
        await _reconcile_with_provider(client, parent, pdf_bytes, provider)
        (child,) = await client.list_documents(filter=parent_uri_filter(parent_uri))

    assert child.metadata["kept"] == "yes"
    assert child.metadata["parent_uri"] == parent_uri
    assert child.metadata["content_type"] == "text/plain"
    assert child.metadata["md5"] != "forged"
    assert "source_id" not in child.metadata
    assert "source_revision" not in child.metadata


async def test_provider_called_for_nested_attachments_up_to_cap(
    temp_db_path, monkeypatch
):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    l3 = build_pdf([("leaf.txt", b"deepest")])
    l2 = build_pdf([("l3.pdf", l3)])
    l1 = build_pdf([("l2.pdf", l2)])
    root = build_pdf([("l1.pdf", l1)])
    provider = _RecordingProvider()

    async with HaikuRAG(temp_db_path, create=True) as client:
        root_uri = "file:///fixtures/root.pdf"
        parent = await _make_parent(client, root_uri, root)
        await _reconcile_with_provider(client, parent, root, provider)
        l1_uri = f"{root_uri}#attachment=l1.pdf"
        l2_uri = f"{l1_uri}#attachment=l2.pdf"
        l2_doc = await client.get_document_by_uri(l2_uri)

    assert [(s, u) for s, u, _ in provider.calls] == [("src", l1_uri), ("src", l2_uri)]
    assert provider.calls[1][2].extra_metadata == {"parent_uri": l1_uri}
    assert l2_doc is not None
    assert l2_doc.metadata["seen_uri"] == l2_uri


async def test_provider_skipped_for_unchanged_child_and_rerun_for_changed(
    temp_db_path, monkeypatch
):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///fixtures/parent.pdf"
        first = build_pdf([("stable.txt", b"same"), ("changed.txt", b"old")])
        parent = await _make_parent(client, parent_uri, first)
        await _reconcile_with_provider(
            client, parent, first, _RecordingProvider({"run": "1", "only_first": "y"})
        )

        second_provider = _RecordingProvider({"run": "2"})
        second = build_pdf([("stable.txt", b"same"), ("changed.txt", b"new")])
        await _reconcile_with_provider(client, parent, second, second_provider)
        stable = await client.get_document_by_uri(f"{parent_uri}#attachment=stable.txt")
        changed = await client.get_document_by_uri(
            f"{parent_uri}#attachment=changed.txt"
        )

    assert [u for _, u, _ in second_provider.calls] == [
        f"{parent_uri}#attachment=changed.txt"
    ]
    assert stable is not None and changed is not None
    assert stable.metadata["run"] == "1"
    assert changed.metadata["run"] == "2"
    assert "only_first" not in changed.metadata


async def test_provider_failure_for_an_attachment_is_logged_not_raised(
    tmp_path, temp_db_path, monkeypatch
):
    """The parent is stored before its children, so a provider raising for a
    child must neither fail the parent nor cost the child or its siblings."""
    import logging

    from tests.conftest import capture_logs

    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    pdf_path = tmp_path / "parent.pdf"
    pdf_path.write_bytes(build_pdf([("bad.txt", b"B"), ("good.txt", b"G")]))
    provider = _RecordingProvider({"tag": "x"}, fail_for=("bad.txt",))

    logger = logging.getLogger("haiku.rag.client.documents")
    async with HaikuRAG(temp_db_path, create=True) as client:
        with capture_logs(logger, logging.WARNING) as records:
            parent = await _ingest_with_provider(tmp_path, client, pdf_path, provider)
        bad = await client.get_document_by_uri(f"{parent.uri}#attachment=bad.txt")
        good = await client.get_document_by_uri(f"{parent.uri}#attachment=good.txt")

    assert parent.metadata["tag"] == "x"
    assert bad is not None
    assert "tag" not in bad.metadata
    assert bad.metadata["parent_uri"] == parent.uri
    assert good is not None
    assert good.metadata["tag"] == "x"
    (warning,) = records
    assert warning.args[:2] == (f"{parent.uri}#attachment=bad.txt", parent.uri)
    assert warning.exc_info is not None


async def test_provider_mutating_its_result_cannot_change_a_child(
    temp_db_path, monkeypatch
):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )

    async def mutating(source_id: str, uri: str, result: FetchResult) -> dict:
        result.content_hash = "tampered"
        result.extra_metadata["parent_uri"] = "file:///elsewhere.pdf"
        return {}

    pdf_bytes = build_pdf([("a.txt", b"A")])
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///fixtures/parent.pdf"
        parent = await _make_parent(client, parent_uri, pdf_bytes)
        await _reconcile_with_provider(client, parent, pdf_bytes, mutating)
        (child,) = await client.list_documents(filter=parent_uri_filter(parent_uri))

    assert child.metadata["parent_uri"] == parent_uri
    assert child.metadata["md5"] != "tampered"


async def test_provider_without_a_source_id_is_not_called(temp_db_path, monkeypatch):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    provider = _RecordingProvider()
    pdf_bytes = build_pdf([("a.txt", b"A")])
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent = await _make_parent(client, "file:///fixtures/p.pdf", pdf_bytes)
        await _reconcile_pdf_attachments(
            writing(client), parent, pdf_bytes, depth=0, metadata_provider=provider
        )
        (child,) = await client.list_documents(filter=parent_uri_filter(parent.uri))

    assert provider.calls == []
    assert "seen_uri" not in child.metadata


async def test_provider_not_called_for_an_unsupported_attachment(
    temp_db_path, monkeypatch
):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    provider = _RecordingProvider()
    pdf_bytes = build_pdf([("Press Quality.joboptions", b"/Tags\n")])
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent = await _make_parent(client, "file:///fixtures/p.pdf", pdf_bytes)
        await _reconcile_with_provider(client, parent, pdf_bytes, provider)
        assert await client.list_documents(filter=parent_uri_filter(parent.uri)) == []

    assert provider.calls == []


async def test_ingest_takes_the_extension_from_the_attachment_name(temp_db_path):
    """The synthetic URI's fragment is dropped by the URL-suffix fallback, which
    would inherit the parent's .pdf; `filename` makes the name authoritative,
    so an unsupported name is rejected before any converter call."""
    import pytest

    from haiku.rag.client.documents import _ingest_fetch_result
    from haiku.rag.client.exceptions import UnsupportedSourceError

    child_uri = "file:///fixtures/brochure.pdf#attachment=Press%20Quality.joboptions"
    result = FetchResult(
        uri=child_uri,
        body=b"/CompressObjects /Tags\n",
        content_type="application/pdf",
        content_hash="x",
    )
    async with HaikuRAG(temp_db_path, create=True) as client:
        with pytest.raises(UnsupportedSourceError, match=".joboptions"):
            await _ingest_fetch_result(
                writing(client),
                result,
                title=None,
                user_metadata={},
                stored_uri=child_uri,
                existing_doc=None,
                filename="Press Quality.joboptions",
            )
