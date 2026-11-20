import os
import sys
import time
from datetime import UTC, datetime

import pytest

from haiku.rag.store.engine import Store
from haiku.rag.store.models.document import Document
from haiku.rag.store.models.document_item import DocumentItem
from haiku.rag.store.repositories import document as document_repository
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.store.repositories.document_item import DocumentItemRepository
from tests.conftest import run_with_lance_memory_pool


@pytest.mark.parametrize("include_content", [False, True])
async def test_document_list_all(
    qa_corpus: list[dict[str, str]], temp_db_path, include_content
):
    """list_all excludes content and docling_document unless include_content=True."""
    async with Store(temp_db_path, create=True) as store:
        doc_repo = DocumentRepository(store)

        content = qa_corpus[0]["document_extracted"]
        doc = Document(
            content=content,
            uri="https://example.com/doc.txt",
            title="Test Document",
            metadata={"key": "value"},
        )
        created = await doc_repo.create(doc)

        docs = await doc_repo.list_all(include_content=include_content)
        assert len(docs) == 1
        assert docs[0].id == created.id
        assert docs[0].title == "Test Document"
        assert docs[0].uri == "https://example.com/doc.txt"
        assert docs[0].metadata == {"key": "value"}
        assert docs[0].content == (content if include_content else "")
        assert docs[0].docling_document is None


async def test_document_list_all_content_skips_docling_blobs(temp_db_path):
    """include_content loads content only; the docling blob and the original
    (hundreds of MB per document) are never materialized."""
    async with Store(temp_db_path, create=True) as store:
        doc_repo = DocumentRepository(store)

        for content in ("first", "second"):
            await doc_repo.create(
                Document(
                    content=content,
                    docling_document=b"structure-blob",
                    original=b"original-file",
                )
            )

        docs = await doc_repo.list_all(include_content=True)

        assert {d.content for d in docs} == {"first", "second"}
        assert all(d.docling_document is None for d in docs)
        assert all(d.original is None for d in docs)


async def test_document_list_all_content_spans_batches(temp_db_path, monkeypatch):
    """Every document gets its content when the lookup spans several batches."""
    monkeypatch.setattr(document_repository, "_CONTENT_BATCH", 2)
    async with Store(temp_db_path, create=True) as store:
        doc_repo = DocumentRepository(store)
        contents = {f"document {i}" for i in range(5)}
        for content in contents:
            await doc_repo.create(Document(content=content))

        docs = await doc_repo.list_all(include_content=True)

        assert {d.content for d in docs} == contents


async def test_document_list_with_filter(qa_corpus: list[dict[str, str]], temp_db_path):
    """Test listing documents with filter clause."""
    async with Store(temp_db_path, create=True) as store:
        doc_repo = DocumentRepository(store)

        first_doc = qa_corpus[0]
        document_text = first_doc["document_extracted"]

        doc1 = Document(
            content=document_text,
            uri="https://example.com/doc1.txt",
            metadata={"source": "test", "category": "A"},
        )
        doc2 = Document(
            content=document_text,
            uri="https://arxiv.org/paper.pdf",
            metadata={"source": "test", "category": "B"},
        )
        doc3 = Document(
            content=document_text,
            uri="https://example.com/doc3.txt",
            metadata={"source": "test", "category": "A"},
        )

        created_doc1 = await doc_repo.create(doc1)
        created_doc2 = await doc_repo.create(doc2)
        created_doc3 = await doc_repo.create(doc3)

        all_documents = await doc_repo.list_all()
        assert len(all_documents) == 3

        arxiv_documents = await doc_repo.list_all(filter="uri LIKE '%arxiv%'")
        assert len(arxiv_documents) == 1
        assert arxiv_documents[0].id == created_doc2.id

        example_documents = await doc_repo.list_all(filter="uri LIKE '%example.com%'")
        assert len(example_documents) == 2
        assert {doc.id for doc in example_documents} == {
            created_doc1.id,
            created_doc3.id,
        }


async def test_document_create_batch(qa_corpus: list[dict[str, str]], temp_db_path):
    """`create` writes a list of documents in one version, and `create([])` writes
    nothing."""
    async with Store(temp_db_path, create=True) as store:
        doc_repo = DocumentRepository(store)

        before = await store.documents_table.version()
        assert await doc_repo.create([]) == []
        assert await store.documents_table.version() == before

        content = qa_corpus[0]["document_extracted"]
        doc_a = Document(content=content, uri="https://example.com/a.txt", title="A")
        doc_b = Document(content=content, uri="https://example.com/b.txt", title="B")

        before = await store.documents_table.version()
        created = await doc_repo.create([doc_a, doc_b])
        after = await store.documents_table.version()

        assert isinstance(created, list)
        assert len(created) == 2
        assert created[0].id is not None
        assert created[1].id is not None
        assert created[0].id != created[1].id
        assert after - before == 1

        round_a = await doc_repo.get_by_id(created[0].id)
        round_b = await doc_repo.get_by_id(created[1].id)
        assert round_a is not None and round_a.title == "A"
        assert round_b is not None and round_b.title == "B"


async def test_document_item_create_all(temp_db_path):
    """`create_all` writes several documents' items in one version. `create_all([])`
    and `create_items(id, [])` write nothing."""
    async with Store(temp_db_path, create=True) as store:
        item_repo = DocumentItemRepository(store)

        before = await store.document_items_table.version()
        await item_repo.create_all([])
        assert await store.document_items_table.version() == before

        before = await store.document_items_table.version()
        await item_repo.create_items("doc-1", [])
        assert await store.document_items_table.version() == before

        items = [
            DocumentItem(
                document_id="doc-1", position=0, self_ref="#/texts/0", text="a"
            ),
            DocumentItem(
                document_id="doc-1", position=1, self_ref="#/texts/1", text="b"
            ),
            DocumentItem(
                document_id="doc-2", position=0, self_ref="#/texts/0", text="c"
            ),
        ]

        before = await store.document_items_table.version()
        await item_repo.create_all(items)
        after = await store.document_items_table.version()

        assert after - before == 1

        doc1_items = await item_repo.get_all_items("doc-1")
        doc2_items = await item_repo.get_all_items("doc-2")
        assert [i.text for i in doc1_items] == ["a", "b"]
        assert [i.text for i in doc2_items] == ["c"]


async def test_document_item_delete_by_document_ids(temp_db_path):
    """Deletes the listed documents' items in one version; [] writes nothing."""
    async with Store(temp_db_path, create=True) as store:
        item_repo = DocumentItemRepository(store)
        await item_repo.create_all(
            [
                DocumentItem(document_id=doc_id, position=0, self_ref="#/texts/0")
                for doc_id in ("doc-1", "doc-2", "doc-3")
            ]
        )

        before = await store.document_items_table.version()
        await item_repo.delete_by_document_ids([])
        assert await store.document_items_table.version() == before

        await item_repo.delete_by_document_ids(["doc-1", "doc-3"])

        assert await store.document_items_table.version() == before + 1
        assert await item_repo.get_all_items("doc-1") == []
        assert await item_repo.get_all_items("doc-3") == []
        assert len(await item_repo.get_all_items("doc-2")) == 1


def test_document_get_docling_document():
    """Test parsing stored DoclingDocument JSON."""
    doc_json = {
        "name": "test_doc",
        "texts": [
            {
                "self_ref": "#/texts/0",
                "text": "Test text",
                "orig": "Test text",
                "label": "paragraph",
            },
        ],
        "tables": [],
        "pictures": [],
        "groups": [],
        "body": {"self_ref": "#/body", "children": []},
        "furniture": {"self_ref": "#/furniture", "children": []},
    }

    import json

    from haiku.rag.store.compression import compress_json

    document = Document(
        content="Test content",
        docling_document=compress_json(json.dumps(doc_json)),
        docling_version="1.3.0",
    )

    docling_doc = document.get_docling_document()

    assert docling_doc is not None
    assert docling_doc.name == "test_doc"
    assert len(docling_doc.texts) == 1
    assert docling_doc.texts[0].text == "Test text"


def test_document_get_docling_document_none():
    """Test get_docling_document returns None when not stored."""
    document = Document(content="Test content")

    assert document.docling_document is None
    assert document.get_docling_document() is None


def test_set_docling_keeps_page_sizes_without_images():
    """set_docling stores the pages' sizes in the structure, without rasters."""
    import json

    from docling_core.types.doc.base import Size
    from docling_core.types.doc.document import DoclingDocument, ImageRef
    from docling_core.types.doc.labels import DocItemLabel
    from PIL import Image

    from haiku.rag.store.compression import decompress_json

    docling_doc = DoclingDocument(name="pages_test")
    docling_doc.add_text(label=DocItemLabel.PARAGRAPH, text="Hello world")
    docling_doc.add_page(
        page_no=1,
        size=Size(width=612, height=792),
        image=ImageRef.from_pil(Image.new("RGB", (612, 792), "white"), dpi=72),
    )

    document = Document(content="test")
    document.set_docling(docling_doc)

    assert document.docling_version == docling_doc.version
    assert document.docling_document is not None
    structure = json.loads(decompress_json(document.docling_document))
    assert structure["pages"]["1"]["size"] == {"width": 612.0, "height": 792.0}
    assert structure["pages"]["1"]["image"] is None

    parsed = document.get_docling_document()
    assert parsed is not None
    assert parsed.name == "pages_test"
    assert parsed.pages[1].size.width == 612
    assert len(list(parsed.iterate_items())) > 0


def test_compress_docling_dict_sources_match():
    """Both ways of building the compression input dict must yield identical
    stored bytes: ``model_dump(mode="json")`` (used at ingest) and
    ``json.loads(model_dump_json())`` (used when migrating a stored string).
    In particular int-keyed ``pages`` must serialize to string keys either way.
    """
    import json

    from docling_core.types.doc.base import Size
    from docling_core.types.doc.document import DoclingDocument, PageItem
    from docling_core.types.doc.labels import DocItemLabel

    from haiku.rag.store.compression import compress_docling

    docling_doc = DoclingDocument(name="equivalence_test")
    docling_doc.add_text(label=DocItemLabel.PARAGRAPH, text="Hello world")
    docling_doc.pages[1] = PageItem(size=Size(width=612, height=792), page_no=1)

    assert compress_docling(docling_doc.model_dump(mode="json")) == compress_docling(
        json.loads(docling_doc.model_dump_json())
    )


async def test_get_docling_data_loads_only_docling_columns(
    qa_corpus: list[dict[str, str]], temp_db_path
):
    """get_docling_data returns docling blob without loading content."""
    import json

    from haiku.rag.store.compression import compress_json

    async with Store(temp_db_path, create=True) as store:
        doc_repo = DocumentRepository(store)

        doc_json = {
            "name": "test_doc",
            "texts": [],
            "tables": [],
            "pictures": [],
            "groups": [],
            "body": {"self_ref": "#/body", "children": []},
            "furniture": {"self_ref": "#/furniture", "children": []},
        }
        compressed = compress_json(json.dumps(doc_json))

        doc = Document(
            content=qa_corpus[0]["document_extracted"],
            uri="https://example.com/doc.txt",
            docling_document=compressed,
            docling_version="2.1.0",
        )
        created = await doc_repo.create(doc)
        assert created.id is not None

        result = await doc_repo.get_docling_data(created.id)
        assert result is not None
        assert result.id == created.id
        assert result.content == ""
        assert result.docling_document == compressed
        assert result.docling_version == "2.1.0"

        # Verify docling document can be parsed
        docling_doc = result.get_docling_document()
        assert docling_doc is not None
        assert docling_doc.name == "test_doc"

        # Non-existent ID returns None
        assert await doc_repo.get_docling_data("nonexistent-id") is None


async def test_get_original_loads_only_the_original(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        doc_repo = DocumentRepository(store)
        stored = await doc_repo.create(Document(content="a", original=b"%PDF-"))
        bare = await doc_repo.create(Document(content="b"))
        assert stored.id is not None and bare.id is not None

        assert await doc_repo.get_original(stored.id) == b"%PDF-"
        assert await doc_repo.get_original(bare.id) is None
        assert await doc_repo.get_original("nonexistent-id") is None


@pytest.mark.parametrize("include_blobs", [False, True])
async def test_document_get_by_id_docling_blobs(temp_db_path, include_blobs):
    """get_by_id leaves the docling blob out unless asked for it, and never
    loads the original, which runs to hundreds of MB."""
    async with Store(temp_db_path, create=True) as store:
        doc_repo = DocumentRepository(store)

        created = await doc_repo.create(
            Document(
                content="the text",
                uri="https://example.com/doc.pdf",
                title="Test Document",
                metadata={"key": "value"},
                docling_document=b"structure-blob",
                original=b"original-file",
                docling_version="2.1.0",
            )
        )
        assert created.id is not None

        doc = await doc_repo.get_by_id(created.id, include_blobs=include_blobs)

        assert doc is not None
        assert doc.id == created.id
        assert doc.content == "the text"
        assert doc.uri == "https://example.com/doc.pdf"
        assert doc.title == "Test Document"
        assert doc.metadata == {"key": "value"}
        assert doc.original is None
        if include_blobs:
            assert doc.docling_document == b"structure-blob"
            assert doc.docling_version == "2.1.0"
        else:
            assert doc.docling_document is None


@pytest.mark.parametrize("include_blobs", [False, True])
async def test_document_get_by_uri_docling_blobs(temp_db_path, include_blobs):
    """get_by_uri has the same projection as get_by_id."""
    async with Store(temp_db_path, create=True) as store:
        doc_repo = DocumentRepository(store)

        created = await doc_repo.create(
            Document(
                content="the text",
                uri="https://example.com/doc.pdf",
                docling_document=b"structure-blob",
                original=b"original-file",
            )
        )

        doc = await doc_repo.get_by_uri(
            "https://example.com/doc.pdf", include_blobs=include_blobs
        )

        assert doc is not None
        assert doc.id == created.id
        assert doc.content == "the text"
        assert doc.original is None
        if include_blobs:
            assert doc.docling_document == b"structure-blob"
        else:
            assert doc.docling_document is None


async def test_document_get_by_uri_with_special_characters(
    qa_corpus: list[dict[str, str]], temp_db_path
):
    """Test get_by_uri handles URIs with special characters like single quotes."""
    async with Store(temp_db_path, create=True) as store:
        doc_repo = DocumentRepository(store)

        first_doc = qa_corpus[0]
        document_text = first_doc["document_extracted"]

        doc_with_quote = Document(
            content=document_text,
            uri="Hamish and Andy's Gap Year",
            metadata={"source": "test"},
        )

        created_doc = await doc_repo.create(doc_with_quote)

        retrieved = await doc_repo.get_by_uri("Hamish and Andy's Gap Year")
        assert retrieved is not None
        assert retrieved.id == created_doc.id
        assert retrieved.uri == "Hamish and Andy's Gap Year"


def test_naive_timestamp_is_converted_from_local_time():
    if sys.platform == "win32":
        pytest.skip("time.tzset is POSIX-only")
    original_tz = os.environ.get("TZ")
    os.environ["TZ"] = "Europe/Athens"
    time.tzset()
    try:
        created = datetime(2026, 1, 1, 14, 0, 0)  # January is EET, UTC+2
        updated = datetime(2026, 1, 1, 14, 30, 0)
        doc = Document(content="x", created_at=created, updated_at=updated)
        assert doc.created_at == datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        assert doc.updated_at == datetime(2026, 1, 1, 12, 30, 0, tzinfo=UTC)
    finally:
        if original_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original_tz
        time.tzset()


def test_created_at_serializes_with_timezone_offset():
    doc = Document(content="x")
    value = doc.model_dump(mode="json")["created_at"]
    assert datetime.fromisoformat(value).utcoffset() is not None


_UPDATE_OVER_POOL = """
import asyncio, os, sys
from pathlib import Path

from haiku.rag.config.models import AppConfig
from haiku.rag.store.engine import Store
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.document import DocumentRepository


async def main():
    config = AppConfig()
    config.storage.auto_vacuum = False
    async with Store(Path(sys.argv[1]), config=config, create=True) as store:
        repo = DocumentRepository(store)
        target = await repo.create(Document(content="target", uri="test://target"))
        other = await repo.create(
            Document(content="other", uri="test://other", docling_document=b"kept")
        )
        await store.rebuild_indexes()
        stats = await store.documents_table.index_stats("id_idx")
        assert stats is not None and stats.num_indexed_rows == 2
        version = await store.documents_table.version()

        blob = os.urandom(int(sys.argv[2]))
        target.docling_document = blob
        await repo.update(target)

        assert await store.documents_table.version() == version + 1
        stored = await repo.get_by_id(target.id, include_blobs=True)
        assert stored is not None and stored.docling_document == blob
        untouched = await repo.get_by_id(other.id, include_blobs=True)
        assert untouched is not None
        assert untouched.content == "other"
        assert untouched.docling_document == b"kept"


asyncio.run(main())
"""


def test_update_writes_a_row_larger_than_the_lance_memory_pool(tmp_path):
    """Updating an indexed row with blobs larger than lance's query memory pool."""
    result = run_with_lance_memory_pool(
        _UPDATE_OVER_POOL, 1 << 20, str(tmp_path / "db.lancedb"), str(4 << 20)
    )
    assert result.returncode == 0, result.stderr
