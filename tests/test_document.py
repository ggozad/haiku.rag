import pytest
from datasets import Dataset

from haiku.rag.store.engine import Store
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.document import DocumentRepository


@pytest.mark.asyncio
async def test_document_list_excludes_content_by_default(
    qa_corpus: Dataset, temp_db_path
):
    """list_all excludes content and docling_document by default."""
    store = Store(temp_db_path, create=True)
    doc_repo = DocumentRepository(store)

    doc = Document(
        content=qa_corpus[0]["document_extracted"],
        uri="https://example.com/doc.txt",
        title="Test Document",
        metadata={"key": "value"},
    )
    created = await doc_repo.create(doc)

    docs = await doc_repo.list_all()
    assert len(docs) == 1
    assert docs[0].id == created.id
    assert docs[0].title == "Test Document"
    assert docs[0].uri == "https://example.com/doc.txt"
    assert docs[0].metadata == {"key": "value"}
    assert docs[0].content == ""
    assert docs[0].docling_document is None

    store.close()


@pytest.mark.asyncio
async def test_document_list_includes_content_when_requested(
    qa_corpus: Dataset, temp_db_path
):
    """list_all returns content when include_content=True."""
    store = Store(temp_db_path, create=True)
    doc_repo = DocumentRepository(store)

    content = qa_corpus[0]["document_extracted"]
    doc = Document(content=content, uri="https://example.com/doc.txt")
    created = await doc_repo.create(doc)

    docs = await doc_repo.list_all(include_content=True)
    assert len(docs) == 1
    assert docs[0].id == created.id
    assert docs[0].content == content

    store.close()


@pytest.mark.asyncio
async def test_document_list_with_filter(qa_corpus: Dataset, temp_db_path):
    """Test listing documents with filter clause."""
    store = Store(temp_db_path, create=True)
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
    assert {doc.id for doc in example_documents} == {created_doc1.id, created_doc3.id}

    store.close()


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


def test_set_docling_splits_structure_and_pages():
    """set_docling stores structure and pages separately."""
    import json

    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.labels import DocItemLabel

    from haiku.rag.store.compression import decompress_json

    docling_doc = DoclingDocument(name="split_test")
    docling_doc.add_text(label=DocItemLabel.PARAGRAPH, text="Hello world")

    document = Document(content="test")
    document.set_docling(docling_doc)

    assert document.docling_document is not None
    assert document.docling_version == docling_doc.version

    # Structure should not contain pages
    structure = json.loads(decompress_json(document.docling_document))
    assert "pages" not in structure
    assert structure["name"] == "split_test"

    # get_docling_document should work from the split structure
    parsed = document.get_docling_document()
    assert parsed is not None
    assert parsed.name == "split_test"
    assert len(list(parsed.iterate_items())) > 0


def test_set_docling_with_page_images():
    """set_docling stores page images in docling_pages."""
    import json

    from docling_core.types.doc.base import Size
    from docling_core.types.doc.document import DoclingDocument, PageItem
    from docling_core.types.doc.labels import DocItemLabel

    from haiku.rag.store.compression import decompress_json

    docling_doc = DoclingDocument(name="pages_test")
    docling_doc.add_text(label=DocItemLabel.PARAGRAPH, text="Content")
    docling_doc.pages[1] = PageItem(
        size=Size(width=612, height=792),
        page_no=1,
    )

    document = Document(content="test")
    document.set_docling(docling_doc)

    assert document.docling_pages is not None

    # Pages blob should contain page data
    pages = json.loads(decompress_json(document.docling_pages))
    assert "1" in pages


def test_get_page_images():
    """get_page_images returns requested pages from docling_pages blob."""
    import json

    from haiku.rag.store.compression import compress_json

    pages_data = {
        "1": {"size": {"width": 612, "height": 792}, "page_no": 1},
        "2": {"size": {"width": 612, "height": 792}, "page_no": 2},
        "3": {"size": {"width": 612, "height": 792}, "page_no": 3},
    }
    document = Document(
        content="test",
        docling_pages=compress_json(json.dumps(pages_data)),
    )

    result = document.get_page_images([1, 3])
    assert len(result) == 2
    assert 1 in result
    assert 3 in result
    assert 2 not in result

    # Missing pages are skipped
    result = document.get_page_images([99])
    assert len(result) == 0

    # None docling_pages returns empty
    doc_no_pages = Document(content="test")
    assert doc_no_pages.get_page_images([1]) == {}


@pytest.mark.asyncio
async def test_get_docling_data_loads_only_docling_columns(
    qa_corpus: Dataset, temp_db_path
):
    """get_docling_data returns docling blob without loading content."""
    import json

    from haiku.rag.store.compression import compress_json

    store = Store(temp_db_path, create=True)
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

    store.close()


@pytest.mark.asyncio
async def test_get_pages_data_loads_only_pages_column(qa_corpus: Dataset, temp_db_path):
    """get_pages_data returns only page image data for a document."""
    import json

    from haiku.rag.store.compression import compress_json

    pages_blob = compress_json(
        json.dumps({"1": {"size": {"width": 612, "height": 792}, "page_no": 1}})
    )

    store = Store(temp_db_path, create=True)
    doc_repo = DocumentRepository(store)

    doc = Document(
        content=qa_corpus[0]["document_extracted"],
        uri="https://example.com/doc.txt",
        docling_pages=pages_blob,
    )
    created = await doc_repo.create(doc)
    assert created.id is not None

    result = await doc_repo.get_pages_data(created.id)
    assert result is not None
    assert result.id == created.id
    assert result.content == ""
    assert result.docling_pages == pages_blob

    # Non-existent ID returns None
    assert await doc_repo.get_pages_data("nonexistent-id") is None

    store.close()


@pytest.mark.asyncio
async def test_get_pages_data_none_for_markdown_document(
    qa_corpus: Dataset, temp_db_path
):
    """Markdown documents have no page images — get_pages_data returns None pages."""
    store = Store(temp_db_path, create=True)
    doc_repo = DocumentRepository(store)

    doc = Document(
        content=qa_corpus[0]["document_extracted"],
        uri="https://example.com/doc.md",
    )
    created = await doc_repo.create(doc)
    assert created.id is not None

    result = await doc_repo.get_pages_data(created.id)
    assert result is not None
    assert result.id == created.id
    assert result.docling_pages is None

    store.close()


@pytest.mark.asyncio
async def test_document_get_by_uri_with_special_characters(
    qa_corpus: Dataset, temp_db_path
):
    """Test get_by_uri handles URIs with special characters like single quotes."""
    store = Store(temp_db_path, create=True)
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

    store.close()
