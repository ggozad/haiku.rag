import logging
import threading
from collections import Counter
from functools import partial

import pytest

from haiku.rag.client import HaikuRAG, RebuildMode
from haiku.rag.client import documents as client_documents
from haiku.rag.client import rebuild as rebuild_module
from haiku.rag.client.documents import (
    MAX_ATTACHMENT_DEPTH,
    _extract_pdf_attachments,
    _reconcile_pdf_attachments,
    parent_uri_filter,
)
from haiku.rag.sources import FetchResult
from haiku.rag.store.models.document import Document
from tests.conftest import (
    build_pdf,
    capture_logs,
    fake_ingest_fetch_result,
    writing,
)


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
        filename=None,
        metadata_provider=None,
        force=False,
        written_ids=None,
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
            force=force,
            written_ids=written_ids,
        )

    monkeypatch.setattr("haiku.rag.client.documents._ingest_fetch_result", picky_fake)
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///fixtures/parent.pdf"
        pdf_bytes = build_pdf([("ok.txt", b"keep me"), ("unsupported.xyz", b"data")])
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
    """Provider double: records each call and returns ``metadata`` plus the
    call's URI. Raises the first time it sees a URI ending in ``fail_once``."""

    def __init__(self, metadata: dict | None = None, *, fail_once: str | None = None):
        self.metadata = metadata or {}
        self.fail_once = fail_once
        self.calls: list[tuple[str, str, FetchResult]] = []

    async def __call__(self, source_id: str, uri: str, result: FetchResult) -> dict:
        self.calls.append((source_id, uri, result))
        if self.fail_once is not None and uri.endswith(self.fail_once):
            self.fail_once = None
            raise RuntimeError(f"provider failed for {uri}")
        return {**self.metadata, "seen_uri": uri}

    def attachment_uris(self) -> list[str]:
        return [u for _, u, r in self.calls if "parent_uri" in r.extra_metadata]


async def _ingest_with_provider(tmp_path, client, pdf_path, provider) -> Document:
    """Ingest ``pdf_path`` through an FS source under ``provider``."""
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
        metadata_provider=partial(provider, "src"),
    )


def _md5(data: bytes) -> str:
    import hashlib

    return hashlib.md5(data, usedforsecurity=False).hexdigest()


@pytest.mark.vcr()
@pytest.mark.usefixtures("docling_local_models")
async def test_provider_called_for_each_attachment_with_parent_source_id(
    tmp_path, temp_db_path
):
    pdf_path = tmp_path / "parent.pdf"
    pdf_path.write_bytes(
        build_pdf([("a.txt", b"alpha notes"), ("b.txt", b"beta notes")])
    )
    parent_uri = pdf_path.absolute().as_uri()
    provider = _RecordingProvider({"tag": "x"})

    async with HaikuRAG(temp_db_path, create=True) as client:
        parent = await _ingest_with_provider(tmp_path, client, pdf_path, provider)
        children = {
            c.uri: c
            for c in await client.list_documents(filter=parent_uri_filter(parent_uri))
        }

    a_uri = f"{parent_uri}#attachment=a.txt"
    b_uri = f"{parent_uri}#attachment=b.txt"
    assert [(s, u, r.body) for s, u, r in provider.calls] == [
        ("fs:attachments", str(pdf_path), pdf_path.read_bytes()),
        ("fs:attachments", a_uri, b"alpha notes"),
        ("fs:attachments", b_uri, b"beta notes"),
    ]
    for _, _, result in provider.calls[1:]:
        assert result.content_type == "text/plain"
        assert result.extra_metadata == {"parent_uri": parent_uri}
        assert result.disk_path is None
        assert result.revision is None
    assert parent.metadata["source_id"] == "fs:attachments"
    assert parent.metadata["md5"] == _md5(pdf_path.read_bytes())
    assert "source_revision" in parent.metadata
    for uri in (a_uri, b_uri):
        assert children[uri].metadata["tag"] == "x"
        assert children[uri].metadata["seen_uri"] == uri
        assert "source_id" not in children[uri].metadata


@pytest.mark.vcr()
async def test_provider_reserved_keys_do_not_reach_a_child(temp_db_path):
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
    pdf_bytes = build_pdf([("a.txt", b"alpha notes")])
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///fixtures/parent.pdf"
        parent = await _make_parent(client, parent_uri, pdf_bytes)
        await _reconcile_with_provider(client, parent, pdf_bytes, provider)
        (child,) = await client.list_documents(filter=parent_uri_filter(parent_uri))

    assert child.metadata["kept"] == "yes"
    assert child.metadata["parent_uri"] == parent_uri
    assert child.metadata["content_type"] == "text/plain"
    assert child.metadata["md5"] == _md5(b"alpha notes")
    assert "source_id" not in child.metadata
    assert "source_revision" not in child.metadata


@pytest.mark.usefixtures("docling_local_models")
async def test_provider_called_for_nested_attachments_up_to_cap(temp_db_path):
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


@pytest.mark.vcr()
async def test_provider_skipped_for_unchanged_child_and_rerun_for_changed(
    temp_db_path,
):
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///fixtures/parent.pdf"
        first = build_pdf([("stable.txt", b"same text"), ("changed.txt", b"old text")])
        parent = await _make_parent(client, parent_uri, first)
        await _reconcile_with_provider(
            client, parent, first, _RecordingProvider({"run": "1", "only_first": "y"})
        )

        second_provider = _RecordingProvider({"run": "2"})
        second = build_pdf([("stable.txt", b"same text"), ("changed.txt", b"new text")])
        await _reconcile_with_provider(client, parent, second, second_provider)
        stable = await client.get_document_by_uri(f"{parent_uri}#attachment=stable.txt")
        changed = await client.get_document_by_uri(
            f"{parent_uri}#attachment=changed.txt"
        )

    assert second_provider.attachment_uris() == [f"{parent_uri}#attachment=changed.txt"]
    assert stable is not None and changed is not None
    assert stable.metadata["run"] == "1"
    assert changed.metadata["run"] == "2"
    assert "only_first" not in changed.metadata


async def test_provider_not_called_for_an_unsupported_attachment(temp_db_path):
    provider = _RecordingProvider()
    pdf_bytes = build_pdf([("Press Quality.joboptions", b"/Tags\n")])
    async with HaikuRAG(temp_db_path, create=True) as client:
        parent_uri = "file:///fixtures/p.pdf"
        parent = await _make_parent(client, parent_uri, pdf_bytes)
        await _reconcile_with_provider(client, parent, pdf_bytes, provider)
        assert await client.list_documents(filter=parent_uri_filter(parent_uri)) == []

    assert provider.calls == []


@pytest.mark.vcr()
@pytest.mark.usefixtures("docling_local_models")
async def test_retry_ingests_an_attachment_whose_provider_failed(
    tmp_path, temp_db_path
):
    pdf_path = tmp_path / "parent.pdf"
    pdf_path.write_bytes(
        build_pdf([("a.txt", b"alpha notes"), ("b.txt", b"beta notes")])
    )
    revision = str(pdf_path.stat().st_mtime_ns)
    parent_uri = pdf_path.absolute().as_uri()
    a_uri = f"{parent_uri}#attachment=a.txt"
    b_uri = f"{parent_uri}#attachment=b.txt"
    provider = _RecordingProvider(fail_once="b.txt")

    async with HaikuRAG(temp_db_path, create=True) as client:
        with pytest.raises(RuntimeError, match="provider failed"):
            await _ingest_with_provider(tmp_path, client, pdf_path, provider)
        failed = await client.get_document_by_uri(parent_uri)
        assert failed is not None
        assert "md5" not in failed.metadata
        assert "source_revision" not in failed.metadata
        assert await client.get_document_by_uri(a_uri) is not None
        assert await client.get_document_by_uri(b_uri) is None

        provider.calls.clear()
        retried = await _ingest_with_provider(tmp_path, client, pdf_path, provider)
        b = await client.get_document_by_uri(b_uri)

    assert str(pdf_path.stat().st_mtime_ns) == revision
    assert provider.attachment_uris() == [b_uri]
    assert b is not None
    assert b.metadata["seen_uri"] == b_uri
    assert retried.metadata["source_revision"] == revision
    assert retried.metadata["md5"] == _md5(pdf_path.read_bytes())


@pytest.mark.vcr()
@pytest.mark.usefixtures("docling_local_models")
async def test_retry_ingests_a_grandchild_whose_provider_failed(tmp_path, temp_db_path):
    pdf_path = tmp_path / "root.pdf"
    mid = build_pdf([("leaf.txt", b"leaf notes")])
    pdf_path.write_bytes(build_pdf([("mid.pdf", mid)]))
    root_uri = pdf_path.absolute().as_uri()
    mid_uri = f"{root_uri}#attachment=mid.pdf"
    leaf_uri = f"{mid_uri}#attachment=leaf.txt"
    provider = _RecordingProvider(fail_once="leaf.txt")

    async with HaikuRAG(temp_db_path, create=True) as client:
        with pytest.raises(RuntimeError, match="provider failed"):
            await _ingest_with_provider(tmp_path, client, pdf_path, provider)
        failed_mid = await client.get_document_by_uri(mid_uri)
        assert failed_mid is not None
        assert "md5" not in failed_mid.metadata
        assert "source_revision" not in failed_mid.metadata
        assert await client.get_document_by_uri(leaf_uri) is None

        retried = await _ingest_with_provider(tmp_path, client, pdf_path, provider)
        mid_doc = await client.get_document_by_uri(mid_uri)
        leaf = await client.get_document_by_uri(leaf_uri)

    assert leaf is not None
    assert leaf.metadata["seen_uri"] == leaf_uri
    assert mid_doc is not None
    assert mid_doc.metadata["md5"] == _md5(mid)
    assert retried.metadata["md5"] == _md5(pdf_path.read_bytes())


async def test_full_rebuild_keeps_an_unchanged_attachment_tree(
    tmp_path, temp_db_path, monkeypatch
):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    pdf_path = tmp_path / "parent.pdf"
    pdf_path.write_bytes(
        build_pdf(
            [
                ("direct.txt", b"direct"),
                ("embedded.pdf", build_pdf([("leaf.txt", b"leaf")])),
            ]
        )
    )

    async with HaikuRAG(temp_db_path, create=True) as client:
        await client.create_document_from_source(pdf_path)
        before = {d.uri: d.id for d in await client.list_documents()}

        async for _ in client.rebuild_database(mode=RebuildMode.FULL):
            pass

        after = {d.uri: d.id for d in await client.list_documents()}

    assert len(before) == 4
    assert after == before


async def _full_rebuild(client: HaikuRAG) -> Counter[str]:
    return Counter(
        [doc_id async for doc_id in client.rebuild_database(mode=RebuildMode.FULL)]
    )


def _record_fallbacks(monkeypatch) -> list[str]:
    """Ids of the documents a FULL rebuild rebuilds from stored content."""
    fallbacks: list[str] = []
    flush = rebuild_module._flush_rebuild_batch

    async def recording(session, documents, chunks, **kwargs):
        fallbacks.extend(d.id for d in documents)
        await flush(session, documents, chunks, **kwargs)

    monkeypatch.setattr(rebuild_module, "_flush_rebuild_batch", recording)
    return fallbacks


def _nested_tree() -> bytes:
    return build_pdf(
        [
            ("direct.txt", b"direct"),
            ("embedded.pdf", build_pdf([("leaf.txt", b"leaf")])),
        ]
    )


@pytest.mark.slow
@pytest.mark.vcr()
async def test_full_rebuild_reconverts_each_attachment_from_its_own_payload(
    tmp_path, temp_db_path
):
    pdf_path = tmp_path / "parent.pdf"
    pdf_path.write_bytes(
        build_pdf(
            [
                ("direct.txt", b"The direct attachment is about apples."),
                (
                    "embedded.pdf",
                    build_pdf([("leaf.txt", b"The nested leaf is about pears.")]),
                ),
            ]
        )
    )

    async with HaikuRAG(temp_db_path, create=True) as client:
        parent = await client.create_document_from_source(pdf_path)
        assert isinstance(parent, Document)
        direct_uri = f"{parent.uri}#attachment=direct.txt"
        leaf_uri = f"{parent.uri}#attachment=embedded.pdf#attachment=leaf.txt"
        payloads = {direct_uri: "apples", leaf_uri: "pears"}

        for uri in payloads:
            listed = await client.get_document_by_uri(uri)
            assert listed is not None and listed.id is not None
            doc = await client.document_repository.get_by_id(
                listed.id, include_blobs=True
            )
            assert doc is not None
            doc.content = "Stale conversion."
            doc.metadata = {**doc.metadata, "department": "legal"}
            await client.document_repository.update(doc)

        async def tree() -> dict:
            return {
                d.uri: (
                    d.id,
                    d.metadata.get("parent_uri"),
                    d.metadata.get("department"),
                )
                for d in await client.list_documents()
            }

        async def chunk_ids(uri: str) -> set[str]:
            doc = await client.get_document_by_uri(uri)
            assert doc is not None and doc.id is not None
            chunks = await client.chunk_repository.get_by_document_id(doc.id)
            return {c.id for c in chunks if c.id is not None}

        before = await tree()
        assert len(before) == 4

        for _ in range(2):
            old_chunks = {uri: await chunk_ids(uri) for uri in payloads}
            await _full_rebuild(client)

            assert await tree() == before
            for uri, payload in payloads.items():
                doc = await client.get_document_by_uri(uri)
                assert doc is not None
                assert payload in doc.content
                new_chunks = await chunk_ids(uri)
                assert new_chunks
                assert new_chunks.isdisjoint(old_chunks[uri])


async def test_full_rebuild_rebuilds_attachments_listed_before_their_parent(
    tmp_path, temp_db_path, monkeypatch
):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    pdf_path = tmp_path / "parent.pdf"
    pdf_path.write_bytes(_nested_tree())

    async with HaikuRAG(temp_db_path, create=True) as client:
        await client.create_document_from_source(pdf_path)
        ids = {d.id for d in await client.list_documents()}

        session = writing(client)
        list_documents = session.list_documents

        async def children_first(*args, **kwargs):
            docs = await list_documents(*args, **kwargs)
            return sorted(docs, key=lambda d: "parent_uri" not in d.metadata)

        monkeypatch.setattr(session, "list_documents", children_first)
        fallbacks = _record_fallbacks(monkeypatch)

        yielded = await _full_rebuild(client)

        assert yielded == Counter(ids)
        assert fallbacks == []


async def test_full_rebuild_falls_back_for_attachments_a_failed_parent_left(
    tmp_path, temp_db_path, monkeypatch
):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    pdf_path = tmp_path / "parent.pdf"
    pdf_path.write_bytes(build_pdf([("a.txt", b"a"), ("b.txt", b"b"), ("c.txt", b"c")]))

    async def failing_on_b(session, result, **kwargs):
        if kwargs["stored_uri"].endswith("#attachment=b.txt"):
            raise RuntimeError("conversion failed")
        return await fake_ingest_fetch_result(session, result, **kwargs)

    async with HaikuRAG(temp_db_path, create=True) as client:
        await client.create_document_from_source(pdf_path)
        by_name = {
            d.uri.rsplit("=", 1)[-1] if d.uri and "#" in d.uri else "parent": d.id
            for d in await client.list_documents()
        }

        monkeypatch.setattr(
            "haiku.rag.client.documents._ingest_fetch_result", failing_on_b
        )
        fallbacks = _record_fallbacks(monkeypatch)

        with capture_logs(rebuild_module.logger, logging.WARNING) as records:
            yielded = await _full_rebuild(client)

        assert yielded == Counter(by_name.values())
        assert sorted(fallbacks) == sorted([by_name["b.txt"], by_name["c.txt"]])
        assert [r.getMessage() for r in records] == [
            f"Rebuilding {pdf_path.as_uri()} from source failed (conversion failed)"
        ]


async def test_full_rebuild_reports_an_attachment_added_since_ingestion(
    tmp_path, temp_db_path, monkeypatch
):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    pdf_path = tmp_path / "parent.pdf"
    pdf_path.write_bytes(build_pdf([("kept.txt", b"kept"), ("dropped.txt", b"gone")]))

    async with HaikuRAG(temp_db_path, create=True) as client:
        parent = await client.create_document_from_source(pdf_path)
        assert isinstance(parent, Document)
        kept = await client.get_document_by_uri(f"{parent.uri}#attachment=kept.txt")
        assert kept is not None

        pdf_path.write_bytes(build_pdf([("kept.txt", b"kept"), ("added.txt", b"new")]))
        yielded = await _full_rebuild(client)

        added = await client.get_document_by_uri(f"{parent.uri}#attachment=added.txt")
        assert added is not None
        assert yielded == Counter([parent.id, kept.id, added.id])
        assert (
            await client.get_document_by_uri(f"{parent.uri}#attachment=dropped.txt")
            is None
        )


async def test_full_rebuild_without_the_parent_source_falls_back_for_the_tree(
    tmp_path, temp_db_path, monkeypatch
):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    pdf_path = tmp_path / "parent.pdf"
    pdf_path.write_bytes(_nested_tree())

    async with HaikuRAG(temp_db_path, create=True) as client:
        await client.create_document_from_source(pdf_path)
        before = {d.uri: d.id for d in await client.list_documents()}
        pdf_path.unlink()
        fallbacks = _record_fallbacks(monkeypatch)

        with capture_logs(rebuild_module.logger, logging.WARNING) as records:
            yielded = await _full_rebuild(client)

        assert {d.uri: d.id for d in await client.list_documents()} == before
        assert yielded == Counter(before.values())
        assert sorted(fallbacks) == sorted(before.values())
        assert [r.getMessage() for r in records] == [
            f"Source missing for {pdf_path.as_uri()}, re-embedding from content"
        ]


async def test_written_ids_leaves_out_a_document_whose_write_failed(
    tmp_path, temp_db_path, monkeypatch
):
    async def failing_store(*args, **kwargs):
        raise RuntimeError("write failed")

    monkeypatch.setattr(
        "haiku.rag.client.documents._store_document_with_chunks", failing_store
    )
    path = tmp_path / "note.txt"
    path.write_text("note")
    written_ids: set[str] = set()

    async with HaikuRAG(temp_db_path, create=True) as client:
        with pytest.raises(RuntimeError, match="write failed"):
            await client_documents.create_document_from_source(
                writing(client), path, written_ids=written_ids
            )

    assert written_ids == set()


async def test_written_ids_collects_every_file_of_a_directory(
    tmp_path, temp_db_path, monkeypatch
):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "a.txt").write_text("a")
    (folder / "b.txt").write_text("b")
    written_ids: set[str] = set()

    async with HaikuRAG(temp_db_path, create=True) as client:
        docs = await client_documents.create_document_from_source(
            writing(client), folder, written_ids=written_ids
        )

    assert isinstance(docs, list) and len(docs) == 2
    assert written_ids == {d.id for d in docs}


async def test_full_rebuild_falls_back_for_attachments_when_extraction_is_off(
    tmp_path, temp_db_path, monkeypatch
):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result",
        fake_ingest_fetch_result,
    )
    pdf_path = tmp_path / "parent.pdf"
    pdf_path.write_bytes(_nested_tree())

    async with HaikuRAG(temp_db_path, create=True) as client:
        parent = await client.create_document_from_source(pdf_path)
        assert isinstance(parent, Document)
        before = {d.uri: d.id for d in await client.list_documents()}
        monkeypatch.setattr(client._config.processing, "extract_pdf_attachments", False)
        fallbacks = _record_fallbacks(monkeypatch)

        yielded = await _full_rebuild(client)

        assert {d.uri: d.id for d in await client.list_documents()} == before
        assert yielded == Counter(before.values())
        assert sorted(fallbacks) == sorted(set(before.values()) - {parent.id})
