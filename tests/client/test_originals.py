import hashlib
import logging
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc.base import BoundingBox, Size
from docling_core.types.doc.document import DoclingDocument, ProvenanceItem
from docling_core.types.doc.labels import DocItemLabel
from PIL import Image
from pydantic_ai.embeddings import TestEmbeddingModel

from haiku.rag.client import DocumentImport, HaikuRAG, RebuildMode, documents
from haiku.rag.client.rebuild import (
    _STAGING_MARKER_TABLE_NAME,
    _STAGING_TABLE_NAME,
    IncompleteRebuildRecord,
    _incomplete_rebuild,
    _mark_phase1_complete,
    _populate_staging_table,
    _record_incomplete_rebuild,
)
from haiku.rag.config import AppConfig
from haiku.rag.hooks import Hook, IngestRequest
from haiku.rag.sources.base import FetchResult
from haiku.rag.sources.fs import FSSource
from haiku.rag.store.models.chunk import Chunk
from tests.conftest import build_pdf, capture_logs, serving_url, writing


def _md5(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _png(width: int = 300, height: int = 200) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _paged_doc(width: float = 200.0, height: float = 200.0) -> DoclingDocument:
    doc = DoclingDocument(name="paged")
    doc.add_page(page_no=1, size=Size(width=width, height=height))
    doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text="A paragraph with a box.",
        prov=ProvenanceItem(
            page_no=1,
            bbox=BoundingBox(l=20, t=180, r=180, b=150),
            charspan=(0, 23),
        ),
    )
    return doc


def _chunks(dim: int) -> list[Chunk]:
    return [
        Chunk(
            content="A paragraph with a box.",
            metadata={"doc_item_refs": ["#/texts/0"], "page_numbers": [1]},
            embedding=[0.1] * dim,
            order=0,
        )
    ]


@asynccontextmanager
async def _client(path: Path, config: AppConfig | None = None):
    async with HaikuRAG(path, config=config or AppConfig(), create=True) as rag:
        model = TestEmbeddingModel(dimensions=rag.embedder.vector_dim)
        assert rag.embedder._embedder is not None
        with rag.embedder._embedder.override(model=model):
            yield rag


async def _stored(
    client: HaikuRAG, document_id: str
) -> tuple[bytes | None, str | None]:
    [row] = await (
        client.store.documents_table.query()
        .select(["original", "original_md5"])
        .where(f"id = '{document_id}'")
        .to_list()
    )
    return row["original"], row["original_md5"]


@pytest.fixture
def converted(monkeypatch) -> list[bytes]:
    """Replaces conversion, recording the bytes each conversion read."""
    seen: list[bytes] = []

    async def fake_convert(config, source, *, format="md", source_uri=None):
        seen.append(Path(source).read_bytes())
        return _paged_doc()

    monkeypatch.setattr(documents, "convert", fake_convert)
    return seen


@pytest.mark.parametrize(
    ("name", "body"),
    [("scan.pdf", build_pdf([])), ("scan.png", _png())],
)
async def test_a_source_ingest_stores_the_converted_bytes(
    temp_db_path, tmp_path, converted, name, body
):
    path = tmp_path / name
    path.write_bytes(body)
    async with _client(temp_db_path) as client:
        doc = await client.create_document_from_source(path)
        assert not isinstance(doc, list) and doc.id is not None

        assert await _stored(client, doc.id) == (body, _md5(body))
        assert converted == [body]


async def test_a_text_ingest_stores_no_original(temp_db_path, tmp_path, converted):
    path = tmp_path / "notes.txt"
    path.write_text("Plain notes.")
    async with _client(temp_db_path) as client:
        doc = await client.create_document_from_source(path)
        assert not isinstance(doc, list) and doc.id is not None

        assert await _stored(client, doc.id) == (None, None)


async def test_no_original_is_stored_when_storing_is_off(
    temp_db_path, tmp_path, converted
):
    path = tmp_path / "scan.pdf"
    path.write_bytes(build_pdf([]))
    config = AppConfig()
    config.storage.store_originals = False
    async with _client(temp_db_path, config) as client:
        doc = await client.create_document_from_source(path)
        assert not isinstance(doc, list) and doc.id is not None
        imported = await client.import_document(
            _paged_doc(),
            _chunks(client.embedder.vector_dim),
            uri="test://imported",
            original=build_pdf([]),
        )
        assert imported.id is not None

        assert await _stored(client, doc.id) == (None, None)
        assert await _stored(client, imported.id) == (None, None)


async def test_a_file_converts_in_place_when_originals_are_not_stored(
    temp_db_path, tmp_path, converted, monkeypatch
):
    async def no_copy(body, extension):
        raise AssertionError("the fetched body was written to a temp file")

    monkeypatch.setattr(documents, "_write_fetch_body", no_copy)
    path = tmp_path / "scan.pdf"
    path.write_bytes(build_pdf([]))
    config = AppConfig()
    config.storage.store_originals = False
    async with _client(temp_db_path, config) as client:
        await client.create_document_from_source(path)

    assert len(converted) == 1


async def test_a_pdf_without_a_pdf_content_type_stores_its_original(
    temp_db_path, converted
):
    body = build_pdf([])
    result = FetchResult(
        uri="https://example.com/report.pdf",
        body=body,
        content_type="application/octet-stream",
        content_hash=_md5(body),
    )
    async with _client(temp_db_path) as client:
        doc = await documents._ingest_fetch_result(
            writing(client),
            client,
            result,
            title=None,
            user_metadata={},
            stored_uri=result.uri,
            existing_doc=None,
        )
        assert doc.id is not None

        assert await _stored(client, doc.id) == (body, _md5(body))


async def test_a_before_ingest_body_is_the_stored_original(
    temp_db_path, tmp_path, converted
):
    rewritten = build_pdf([("note.txt", b"added by the hook")])

    class Rewrite(Hook):
        async def before_ingest(self, client, request: IngestRequest):
            request.body = rewritten
            return request

    path = tmp_path / "scan.pdf"
    path.write_bytes(build_pdf([]))
    config = AppConfig()
    config.processing.extract_pdf_attachments = False
    async with _client(temp_db_path, config) as client:
        client._hooks = [Rewrite()]
        doc = await client.create_document_from_source(path)
        assert not isinstance(doc, list) and doc.id is not None

        assert await _stored(client, doc.id) == (rewritten, _md5(rewritten))
        assert converted == [rewritten]


async def test_a_file_replaced_after_its_fetch_is_stored_as_fetched(
    temp_db_path, tmp_path, converted, monkeypatch
):
    path = tmp_path / "scan.pdf"
    fetched = build_pdf([])
    path.write_bytes(fetched)
    fetch = FSSource.fetch

    async def fetch_then_replace(self, uri):
        result = await fetch(self, uri)
        path.write_bytes(build_pdf([("late.txt", b"written after the fetch")]))
        return result

    monkeypatch.setattr(FSSource, "fetch", fetch_then_replace)
    async with _client(temp_db_path) as client:
        doc = await client.create_document_from_source(path)
        assert not isinstance(doc, list) and doc.id is not None

        assert await _stored(client, doc.id) == (fetched, _md5(fetched))
        assert converted == [fetched]


async def test_an_import_with_an_original_stores_it_and_renders_from_it(
    temp_db_path,
):
    original = build_pdf([])
    async with _client(temp_db_path) as client:
        doc = await client.import_document(
            _paged_doc(),
            _chunks(client.embedder.vector_dim),
            uri="test://imported",
            original=original,
        )
        assert doc.id is not None
        [chunk] = await client.chunk_repository.get_by_document_id(doc.id)

        assert await _stored(client, doc.id) == (original, _md5(original))
        [image] = await client.visualize_chunk(chunk)
        assert image.size == (400, 400)


async def test_an_original_that_is_neither_a_pdf_nor_an_image_is_refused(
    temp_db_path,
):
    junk = b"not a document"
    async with _client(temp_db_path) as client:
        chunks = _chunks(client.embedder.vector_dim)
        with pytest.raises(ValueError, match="neither a PDF nor an image"):
            await client.import_document(_paged_doc(), chunks, original=junk)
        with pytest.raises(ValueError, match="neither a PDF nor an image"):
            await client.import_documents(
                [DocumentImport(_paged_doc(), chunks, original=junk)]
            )
        document_id = await _imported(client)
        kept = await _stored(client, document_id)
        with pytest.raises(ValueError, match="neither a PDF nor an image"):
            await client.update_document(
                document_id, docling_document=_paged_doc(), original=junk
            )

        assert await client.count_documents() == 1
        assert await _stored(client, document_id) == kept


async def test_a_batch_import_stores_each_original(temp_db_path):
    first, second = build_pdf([]), _png()
    async with _client(temp_db_path) as client:
        dim = client.embedder.vector_dim
        stored = await client.import_documents(
            [
                DocumentImport(
                    _paged_doc(), _chunks(dim), uri="test://a", original=first
                ),
                DocumentImport(_paged_doc(), _chunks(dim), uri="test://b"),
                DocumentImport(
                    _paged_doc(), _chunks(dim), uri="test://c", original=second
                ),
            ]
        )

        assert [await _stored(client, d.id) for d in stored if d.id] == [
            (first, _md5(first)),
            (None, None),
            (second, _md5(second)),
        ]


async def _imported(client: HaikuRAG) -> str:
    doc = await client.import_document(
        _paged_doc(),
        _chunks(client.embedder.vector_dim),
        uri="test://kept",
        original=build_pdf([]),
    )
    assert doc.id is not None
    return doc.id


async def test_a_new_structure_without_an_original_clears_it(temp_db_path):
    async with _client(temp_db_path) as client:
        replaced_by_content = await _imported(client)
        await client.update_document(replaced_by_content, content="# New text")
        replaced_by_structure = await _imported(client)
        await client.update_document(
            replaced_by_structure, docling_document=_paged_doc()
        )
        recreated = await _imported(client)
        await client.create_document("# Same URI", uri="test://kept")

        for document_id in (replaced_by_content, replaced_by_structure, recreated):
            assert await _stored(client, document_id) == (None, None)


async def test_a_new_structure_with_an_original_replaces_it(temp_db_path):
    replacement = _png()
    async with _client(temp_db_path) as client:
        document_id = await _imported(client)

        await client.update_document(
            document_id, docling_document=_paged_doc(), original=replacement
        )

        assert await _stored(client, document_id) == (
            replacement,
            _md5(replacement),
        )


async def test_updates_that_keep_the_structure_keep_the_original(
    temp_db_path, monkeypatch
):
    async with _client(temp_db_path) as client:
        document_id = await _imported(client)
        kept = await _stored(client, document_id)
        loaded: list[str] = []
        record_by_id = client.document_repository._record_by_id

        async def recording(doc_id, include_blobs):
            record = await record_by_id(doc_id, include_blobs)
            if record is not None and record.original is not None:
                loaded.append(doc_id)
            return record

        monkeypatch.setattr(client.document_repository, "_record_by_id", recording)

        await client.update_document(document_id, metadata={"topic": "boxes"})
        await client.update_document(document_id, title="Renamed")
        await client.update_document(
            document_id, chunks=_chunks(client.embedder.vector_dim)
        )
        await client.update_document(
            document_id,
            content="Edited text",
            chunks=_chunks(client.embedder.vector_dim),
        )

        assert await _stored(client, document_id) == kept
        assert loaded == []


@pytest.mark.parametrize(
    "mode",
    [
        RebuildMode.RECHUNK,
        RebuildMode.EMBED_ONLY,
        RebuildMode.DESCRIPTIONS,
        RebuildMode.TITLE_ONLY,
        RebuildMode.SET_EMBEDDER,
        RebuildMode.REINDEX,
    ],
)
async def test_rebuilds_that_keep_the_structure_keep_the_original(temp_db_path, mode):
    config = AppConfig()
    config.processing.pictures = "description"
    async with _client(temp_db_path, config) as client:
        document_id = await _imported(client)
        kept = await _stored(client, document_id)

        async for _ in client.rebuild_database(mode=mode):
            pass

        assert await _stored(client, document_id) == kept


async def test_a_full_rebuild_from_stored_content_clears_the_original(
    temp_db_path,
):
    async with _client(temp_db_path) as client:
        document_id = await _imported(client)

        async for _ in client.rebuild_database(mode=RebuildMode.FULL):
            pass

        assert await _stored(client, document_id) == (None, None)


async def test_an_original_without_a_new_structure_is_refused(temp_db_path):
    async with _client(temp_db_path) as client:
        document_id = await _imported(client)
        kept = await _stored(client, document_id)

        with pytest.raises(ValueError, match="original needs the docling_document"):
            await client.update_document(document_id, original=_png())

        assert await _stored(client, document_id) == kept


def _config(store: bool) -> AppConfig:
    config = AppConfig()
    config.storage.store_originals = store
    return config


async def _ingest(db: Path, *paths: Path) -> list[str]:
    """Ingest without storing originals, as a database from before them."""
    async with _client(db, _config(False)) as client:
        ids = []
        for path in paths:
            doc = await client.create_document_from_source(path)
            assert not isinstance(doc, list) and doc.id is not None
            ids.append(doc.id)
        return ids


async def _backfill(client: HaikuRAG) -> list[str]:
    return [i async for i in client.rebuild_database(mode=RebuildMode.ORIGINALS)]


async def _by_uri(client: HaikuRAG) -> dict[str, str]:
    return {d.uri: d.id for d in await client.list_documents() if d.uri and d.id}


@pytest.mark.parametrize(
    ("name", "body"), [("scan.pdf", build_pdf([])), ("scan.png", _png())]
)
async def test_a_document_regains_its_original_and_renders(
    temp_db_path, tmp_path, converted, name, body
):
    path = tmp_path / name
    path.write_bytes(body)
    [doc_id] = await _ingest(temp_db_path, path)

    async with _client(temp_db_path) as client:
        assert await _backfill(client) == [doc_id]

        assert await _stored(client, doc_id) == (body, _md5(body))
        [chunk] = await client.chunk_repository.get_by_document_id(doc_id)
        assert len(await client.visualize_chunk(chunk)) == 1


async def test_a_structure_without_pages_renders_with_its_original(
    temp_db_path, tmp_path, monkeypatch
):
    from haiku.rag.client import documents

    async def without_pages(config, source, *, format="md", source_uri=None):
        doc = _paged_doc()
        doc.pages = {}
        return doc

    monkeypatch.setattr(documents, "convert", without_pages)
    path = tmp_path / "scan.pdf"
    path.write_bytes(build_pdf([]))
    [doc_id] = await _ingest(temp_db_path, path)

    async with _client(temp_db_path) as client:
        await _backfill(client)

        [chunk] = await client.chunk_repository.get_by_document_id(doc_id)
        assert len(await client.visualize_chunk(chunk)) == 1


async def test_attachments_two_levels_deep_regain_their_originals(
    temp_db_path, tmp_path, converted
):
    leaf = _png()
    inner = build_pdf([("leaf.png", leaf)])
    path = tmp_path / "outer.pdf"
    path.write_bytes(build_pdf([("inner.pdf", inner)]))
    await _ingest(temp_db_path, path)

    async with _client(temp_db_path) as client:
        ids = await _by_uri(client)
        assert len(ids) == 3
        stored = set(await _backfill(client))

        assert stored == set(ids.values())
        inner_uri = f"{path.as_uri()}#attachment=inner.pdf"
        leaf_uri = f"{inner_uri}#attachment=leaf.png"
        assert await _stored(client, ids[inner_uri]) == (inner, _md5(inner))
        assert await _stored(client, ids[leaf_uri]) == (leaf, _md5(leaf))


async def test_a_sibling_sorting_between_a_parent_and_its_attachment(
    temp_db_path, converted
):
    """`!` sorts before `#`, so in plain URI order the sibling comes between."""
    body = build_pdf([("leaf.png", _png())])
    parent, sibling = "https://example.com/a.pdf", "https://example.com/a.pdf!b.pdf"
    with serving_url(body, "application/pdf"):
        async with _client(temp_db_path, _config(False)) as client:
            await client.create_document_from_source(parent)
            await client.create_document_from_source(sibling)
            ids = await _by_uri(client)
        assert f"{parent}#attachment=leaf.png" in ids

        async with _client(temp_db_path) as client:
            assert set(await _backfill(client)) == set(ids.values())


async def test_an_attachment_is_filled_from_its_parents_stored_original(
    temp_db_path, tmp_path, converted
):
    leaf = _png()
    path = tmp_path / "outer.pdf"
    path.write_bytes(build_pdf([("leaf.png", leaf)]))
    async with _client(temp_db_path) as client:
        await client.create_document_from_source(path)
        leaf_id = (await _by_uri(client))[f"{path.as_uri()}#attachment=leaf.png"]
        await client.store.documents_table.update(
            {"original": None, "original_md5": None}, where=f"id = '{leaf_id}'"
        )
    path.unlink()

    async with _client(temp_db_path) as client:
        assert await _backfill(client) == [leaf_id]

        assert await _stored(client, leaf_id) == (leaf, _md5(leaf))


@pytest.mark.parametrize("change", ["edited", "deleted"])
async def test_a_source_that_is_not_the_ingested_file_is_skipped(
    temp_db_path, tmp_path, converted, change
):
    path = tmp_path / "scan.pdf"
    path.write_bytes(build_pdf([]))
    [doc_id] = await _ingest(temp_db_path, path)
    if change == "edited":
        path.write_bytes(build_pdf([("note.png", _png())]))
    else:
        path.unlink()

    logger = logging.getLogger("haiku.rag")
    level = logger.level
    logger.setLevel(logging.INFO)
    try:
        with capture_logs(logger, logging.INFO) as records:
            async with _client(temp_db_path) as client:
                assert await _backfill(client) == []
                assert await _stored(client, doc_id) == (None, None)
    finally:
        logger.setLevel(level)

    reason = "MD5 differs" if change == "edited" else "source unavailable"
    messages = [r.getMessage() for r in records]
    assert any(path.as_uri() in m and reason in m for m in messages)
    assert any(f"{reason} 1" in m for m in messages)


async def test_only_documents_without_an_original_of_a_grounded_format_are_fetched(
    temp_db_path, tmp_path, converted, monkeypatch
):
    pdf = tmp_path / "kept.pdf"
    pdf.write_bytes(build_pdf([]))
    slides = tmp_path / "slides.pptx"
    slides.write_bytes(b"not really a deck")
    async with _client(temp_db_path) as client:
        await client.create_document_from_source(pdf)
    await _ingest(temp_db_path, slides)
    fetched = []
    fetch = FSSource.fetch

    async def spy(self, uri):
        fetched.append(uri)
        return await fetch(self, uri)

    monkeypatch.setattr(FSSource, "fetch", spy)

    async with _client(temp_db_path) as client:
        assert await _backfill(client) == []

    assert fetched == []


async def test_a_document_updated_during_the_fetch_is_left_alone(
    temp_db_path, tmp_path, converted, monkeypatch
):
    path = tmp_path / "scan.pdf"
    path.write_bytes(build_pdf([]))
    [doc_id] = await _ingest(temp_db_path, path)

    async with _client(temp_db_path) as client:
        fetch = FSSource.fetch

        async def update_meanwhile(self, uri):
            doc = await client.get_document_by_id(doc_id)
            assert doc is not None
            await writing(client).document_repository.update_meta(doc)
            return await fetch(self, uri)

        monkeypatch.setattr(FSSource, "fetch", update_meanwhile)

        assert await _backfill(client) == []
        assert await _stored(client, doc_id) == (None, None)


async def test_a_rerun_stores_nothing_more(temp_db_path, tmp_path, converted):
    path = tmp_path / "scan.pdf"
    path.write_bytes(build_pdf([]))
    [doc_id] = await _ingest(temp_db_path, path)

    async with _client(temp_db_path) as client:
        assert await _backfill(client) == [doc_id]
        assert await _backfill(client) == []


async def test_the_backfill_leaves_an_interrupted_rebuild_alone(
    temp_db_path, tmp_path, converted
):
    path = tmp_path / "scan.pdf"
    path.write_bytes(build_pdf([]))
    await _ingest(temp_db_path, path)
    record = IncompleteRebuildRecord(mode="embed_only", vector_index=False)
    config = AppConfig()
    config.storage.auto_vacuum = False

    async with _client(temp_db_path, config) as client:
        session = writing(client)
        await _record_incomplete_rebuild(session, record)
        await _populate_staging_table(session)
        await _mark_phase1_complete(session)
        staging = await session.store.db.open_table(_STAGING_TABLE_NAME)
        staged = await staging.count_rows()
        settings = session.store.stored_settings
        versions = await session.store.current_table_versions()

        await _backfill(client)

        assert await _incomplete_rebuild(session) == record
        tables = (await session.store.db.list_tables()).tables
        assert {_STAGING_TABLE_NAME, _STAGING_MARKER_TABLE_NAME} <= set(tables)
        assert await staging.count_rows() == staged > 0
        assert session.store.stored_settings == settings
        after = await session.store.current_table_versions()
        assert {n: v for n, v in after.items() if n != "documents"} == {
            n: v for n, v in versions.items() if n != "documents"
        }


@pytest.mark.parametrize("failing_step", ["adapter", "fetch"])
async def test_a_source_that_fails_is_skipped(
    temp_db_path, tmp_path, converted, monkeypatch, failing_step
):
    from haiku.rag import sources

    failing, working = tmp_path / "a.pdf", tmp_path / "b.pdf"
    failing.write_bytes(build_pdf([]))
    working.write_bytes(build_pdf([("x.png", _png())]))
    failing_id, working_id = await _ingest(temp_db_path, failing, working)
    resolve, fetch = sources.resolve_adhoc_fetcher, FSSource.fetch

    def resolve_or_fail(uri, **kwargs):
        if failing_step == "adapter" and uri == failing.as_uri():
            raise ImportError("no adapter")
        return resolve(uri, **kwargs)

    async def fetch_or_fail(self, uri):
        if failing_step == "fetch" and uri == failing.as_uri():
            raise OSError("unreadable")
        return await fetch(self, uri)

    monkeypatch.setattr(sources, "resolve_adhoc_fetcher", resolve_or_fail)
    monkeypatch.setattr(FSSource, "fetch", fetch_or_fail)

    async with _client(temp_db_path) as client:
        stored = await _backfill(client)

    assert working_id in stored and failing_id not in stored


async def test_an_attachment_of_an_unavailable_parent_is_skipped(
    temp_db_path, tmp_path, converted
):
    path = tmp_path / "outer.pdf"
    path.write_bytes(build_pdf([("leaf.png", _png())]))
    await _ingest(temp_db_path, path)
    path.unlink()

    async with _client(temp_db_path) as client:
        assert await _backfill(client) == []


async def test_an_attachment_whose_parent_is_not_stored_is_skipped(
    temp_db_path, tmp_path, converted
):
    path, other = tmp_path / "outer.pdf", tmp_path / "other.png"
    path.write_bytes(build_pdf([("leaf.png", _png())]))
    other.write_bytes(_png())
    parent_id, other_id = await _ingest(temp_db_path, path, other)
    async with _client(temp_db_path) as client:
        await client.store.documents_table.delete(f"id = '{parent_id}'")
        await client.store.document_meta_table.delete(f"id = '{parent_id}'")

        assert await _backfill(client) == [other_id]


async def test_originals_are_written_in_batches(
    temp_db_path, tmp_path, converted, monkeypatch
):
    from haiku.rag.client import rebuild
    from haiku.rag.store.repositories.document import DocumentRepository

    paths = [tmp_path / "a.pdf", tmp_path / "b.png"]
    paths[0].write_bytes(build_pdf([]))
    paths[1].write_bytes(_png())
    ids = await _ingest(temp_db_path, *paths)
    monkeypatch.setattr(rebuild, "_ORIGINALS_WRITE_BYTES", 1)
    writes: list[list[str]] = []
    store_originals = DocumentRepository.store_originals

    async def record(self, originals):
        writes.append(list(originals))
        await store_originals(self, originals)

    monkeypatch.setattr(DocumentRepository, "store_originals", record)

    async with _client(temp_db_path) as client:
        assert await _backfill(client) == ids

    assert writes == [[ids[0]], [ids[1]]]


async def test_the_backfill_refuses_when_originals_are_not_stored(temp_db_path):
    async with _client(temp_db_path, _config(False)) as client:
        with pytest.raises(ValueError, match="store_originals"):
            await _backfill(client)
