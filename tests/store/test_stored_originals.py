import hashlib

from haiku.rag.store.engine import Store
from haiku.rag.store.info import gather_database_info
from haiku.rag.store.models import Document
from haiku.rag.store.repositories.document import DocumentRepository

LARGE = b"\x00" * 4_000_000
SMALL = b"\x00\x01video bytes" * 1000


async def _with_original(store: Store, original: bytes) -> str:
    """The id of a new document whose original is `original`."""
    doc = await DocumentRepository(store).create(Document(content="a recording"))
    assert doc.id is not None
    await DocumentRepository(store).store_originals({doc.id: original})
    return doc.id


def _documents_bytes(info) -> int:
    return next(t.total_bytes for t in info.tables if t.name == "documents")


async def test_info_counts_the_stored_files_in_the_documents_size(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        await _with_original(store, LARGE)
        config = store._config

    info = await gather_database_info(temp_db_path, config)

    assert _documents_bytes(info) >= len(LARGE)


async def test_info_at_a_tag_counts_the_files_the_tag_holds(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        doc_id = await _with_original(store, LARGE)
        await store.create_tag("kept")
        await DocumentRepository(store).delete(doc_id)
        config = store._config

    live = await gather_database_info(temp_db_path, config)
    tagged = await gather_database_info(temp_db_path, config, tag="kept")

    assert _documents_bytes(live) < len(LARGE) <= _documents_bytes(tagged)


async def test_info_counts_an_inline_original_once(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        await _with_original(store, SMALL)
        [row] = await store.documents_table.query().select(["original"]).to_list()
        assert row["original"]["kind"] == 0
        stats: dict = await store.documents_table.stats()  # type: ignore[assignment]
        config = store._config

    info = await gather_database_info(temp_db_path, config)

    assert _documents_bytes(info) == stats["total_bytes"]


async def test_vacuum_keeps_the_original_a_tag_holds(temp_db_path, monkeypatch):
    async with Store(temp_db_path, create=True) as store:
        monkeypatch.setattr(store._config.storage, "auto_vacuum", False)
        doc_id = await _with_original(store, LARGE)
        await store.create_tag("kept")
        await DocumentRepository(store).delete(doc_id)

        assert await store.vacuum(retention_seconds=0)
        await store.restore_tag("kept")

        stored = await DocumentRepository(store).open_original(doc_id)
        assert stored is not None
        assert await stored.read(0, 32) == LARGE[:32]


async def test_only_a_recording_named_by_its_structure_opens_as_one(temp_db_path):
    from docling_core.types.doc.document import DoclingDocument, DocumentOrigin

    def structure(mimetype: str, binary_hash: int) -> DoclingDocument:
        doc = DoclingDocument(name="d")
        doc.origin = DocumentOrigin(
            filename="d.mp4", mimetype=mimetype, binary_hash=binary_hash
        )
        return doc

    sha256 = hashlib.sha256(SMALL).hexdigest()
    named = int(sha256, 16) & (2**64 - 1)
    async with Store(temp_db_path, create=True) as store:
        repo = DocumentRepository(store)
        opened = {}
        for case, mimetype, binary_hash in (
            ("recording", "video/mp4", named),
            ("pdf", "application/pdf", named),
            ("unnamed", "video/mp4", named + 1),
        ):
            doc = Document(content=case)
            doc.set_docling(structure(mimetype, binary_hash), SMALL)
            created = await repo.create(doc)
            assert created.id is not None
            opened[case] = await repo.open_recording(created.id)

    assert opened["recording"] is not None
    assert (opened["recording"].suffix, opened["recording"].video) == (".mp4", True)
    assert opened["pdf"] is None and opened["unnamed"] is None
