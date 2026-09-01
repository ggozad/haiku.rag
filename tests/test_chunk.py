from unittest.mock import patch

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.config import get_config
from haiku.rag.store.models.chunk import Chunk, ChunkMetadata, SearchResult
from tests.conftest import capture_logs


@pytest.mark.vcr()
async def test_chunk_repository_operations(
    qa_corpus: list[dict[str, str]], temp_db_path
):
    """Test ChunkRepository operations."""
    async with HaikuRAG(
        db_path=temp_db_path, config=get_config(), create=True
    ) as client:
        # Get the first document from the corpus
        first_doc = qa_corpus[0]
        document_text = first_doc["document_extracted"]

        # Create a document first with chunks
        created_document = await client.create_document(
            content=document_text, metadata={"source": "test"}
        )
        assert created_document.id is not None

        # Test getting chunks by document ID
        chunks = await client.chunk_repository.get_by_document_id(created_document.id)
        assert len(chunks) > 0
        assert all(chunk.document_id == created_document.id for chunk in chunks)

        # Test chunk search
        results = await client.chunk_repository.search(
            "election", limit=2, search_type="vector"
        )
        assert len(results) <= 2
        assert all(hasattr(chunk, "content") for chunk, _ in results)

        # Test deleting chunks by document ID
        deleted = await client.chunk_repository.delete_by_document_id(
            created_document.id
        )
        assert deleted is True

        # Verify chunks are gone
        chunks_after_delete = await client.chunk_repository.get_by_document_id(
            created_document.id
        )
        assert len(chunks_after_delete) == 0


@pytest.mark.vcr()
async def test_chunk_repository_pagination(
    qa_corpus: list[dict[str, str]], temp_db_path
):
    """Test ChunkRepository pagination with get_by_document_id and count_by_document_id."""
    async with HaikuRAG(
        db_path=temp_db_path, config=get_config(), create=True
    ) as client:
        # Get the first document from the corpus (should produce multiple chunks)
        first_doc = qa_corpus[0]
        document_text = first_doc["document_extracted"]

        # Create a document with chunks
        created_document = await client.create_document(
            content=document_text, metadata={"source": "test"}
        )
        assert created_document.id is not None

        # Get total chunk count
        total_count = await client.chunk_repository.count_by_document_id(
            created_document.id
        )
        assert total_count > 0

        # Get all chunks without pagination
        all_chunks = await client.chunk_repository.get_by_document_id(
            created_document.id
        )
        assert len(all_chunks) == total_count

        # Test pagination with limit
        limit = min(2, total_count)
        first_batch = await client.chunk_repository.get_by_document_id(
            created_document.id, limit=limit
        )
        assert len(first_batch) == limit
        assert first_batch[0].id == all_chunks[0].id

        # Test pagination with offset
        if total_count > limit:
            second_batch = await client.chunk_repository.get_by_document_id(
                created_document.id, limit=limit, offset=limit
            )
            assert len(second_batch) <= limit
            assert second_batch[0].id == all_chunks[limit].id

        # Test offset beyond available chunks
        empty_batch = await client.chunk_repository.get_by_document_id(
            created_document.id, limit=10, offset=total_count + 100
        )
        assert len(empty_batch) == 0


@pytest.mark.vcr()
async def test_chunking_pipeline(qa_corpus: list[dict[str, str]], temp_db_path):
    """Test document chunking using client primitives."""
    from haiku.rag.client import HaikuRAG
    from haiku.rag.embeddings import embed_chunks

    async with HaikuRAG(db_path=temp_db_path, create=True) as client:
        # Get the first document from the corpus
        first_doc = qa_corpus[0]
        document_text = first_doc["document_extracted"]

        # Use client primitives: convert → chunk → embed
        docling_document = await client.convert(document_text)
        chunks = await client.chunk(docling_document)
        embedded_chunks = await embed_chunks(chunks, client.embedder)

        # Verify chunks were created with embeddings
        assert len(chunks) > 0
        assert all(chunk.embedding is None for chunk in chunks)  # Before embedding
        assert all(chunk.embedding is not None for chunk in embedded_chunks)  # After

        # Verify chunk order
        for i, chunk in enumerate(chunks):
            assert chunk.order == i


@pytest.mark.parametrize(
    "metadata,refs,headings,labels,page_numbers",
    [
        (
            {
                "doc_item_refs": ["#/texts/0", "#/texts/1", "#/tables/0"],
                "headings": ["Chapter 1", "Section 1.1"],
                "labels": ["paragraph", "paragraph", "table"],
                "page_numbers": [1, 1, 2],
            },
            ["#/texts/0", "#/texts/1", "#/tables/0"],
            ["Chapter 1", "Section 1.1"],
            ["paragraph", "paragraph", "table"],
            [1, 1, 2],
        ),
        ({}, [], None, [], []),
    ],
    ids=["populated", "defaults"],
)
def test_chunk_metadata_parsing(metadata, refs, headings, labels, page_numbers):
    """Test ChunkMetadata parsing from chunk metadata dict."""
    chunk = Chunk(content="Test content", metadata=metadata)

    chunk_meta = chunk.get_chunk_metadata()

    assert isinstance(chunk_meta, ChunkMetadata)
    assert chunk_meta.doc_item_refs == refs
    assert chunk_meta.headings == headings
    assert chunk_meta.labels == labels
    assert chunk_meta.page_numbers == page_numbers


@pytest.fixture
def two_text_docling_doc():
    """Minimal DoclingDocument with two resolvable text items."""
    from docling_core.types.doc.document import DoclingDocument

    return DoclingDocument.model_validate(
        {
            "name": "test_doc",
            "texts": [
                {
                    "self_ref": "#/texts/0",
                    "text": "First text",
                    "orig": "First text",
                    "label": "paragraph",
                },
                {
                    "self_ref": "#/texts/1",
                    "text": "Second text",
                    "orig": "Second text",
                    "label": "title",
                },
            ],
            "tables": [],
            "pictures": [],
            "groups": [],
            "body": {"self_ref": "#/body", "children": []},
            "furniture": {"self_ref": "#/furniture", "children": []},
        }
    )


@pytest.mark.parametrize(
    "refs,expected_texts",
    [
        (["#/texts/0", "#/texts/1"], ["First text", "Second text"]),
        # Out-of-range and malformed refs are skipped rather than raising.
        (["#/texts/0", "#/texts/999", "#/invalid/path"], ["First text"]),
        ([], []),
    ],
    ids=["all_valid", "graceful_degradation", "empty_refs"],
)
def test_chunk_metadata_resolve_doc_items(two_text_docling_doc, refs, expected_texts):
    """Test resolving doc_item_refs to actual DocItem objects."""
    chunk_meta = ChunkMetadata(doc_item_refs=refs)

    doc_items = chunk_meta.resolve_doc_items(two_text_docling_doc)

    assert [getattr(item, "text") for item in doc_items] == expected_texts


def test_search_result_from_chunk_preserves_document_meta():
    """Document metadata flows from Chunk to SearchResult for citation
    consumers (UIs)."""
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        content="Some content.",
        document_uri="file:///docs/report.pdf",
        document_meta={"source_url": "https://example.org/report/view"},
    )

    result = SearchResult.from_chunk(chunk, score=0.9)

    assert result.document_meta == {"source_url": "https://example.org/report/view"}


def test_search_result_from_chunk_preserves_chunk_meta():
    """Test flow through of unparsed chunk metadata from Chunk to SearchResult"""
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        content="Some content.",
        metadata={
            "headings": ["Chapter 1"],
            "para_no": "12",
            "speaker": "MR SMITH",
        },
    )

    result = SearchResult.from_chunk(chunk, score=0.9)

    assert result.chunk_meta == {
        "headings": ["Chapter 1"],
        "para_no": "12",
        "speaker": "MR SMITH",
    }


def test_search_result_format_for_agent_omits_chunk_meta():
    """Test that chunk_meta is never shown to the model"""
    result = SearchResult(
        content="Some content.",
        score=0.9,
        chunk_id="chunk-1",
        chunk_meta={"para_no": "12"},
    )

    formatted = result.format_for_agent(rank=1, total=1)

    assert "para_no" not in formatted


def test_search_result_format_for_agent_omits_document_meta():
    """Document metadata is UI plumbing, never shown to the model."""
    result = SearchResult(
        content="Some content.",
        score=0.9,
        chunk_id="chunk-1",
        document_meta={"source_url": "https://example.org/report/view"},
    )

    formatted = result.format_for_agent(rank=1, total=1)

    assert "source_url" not in formatted
    assert "https://example.org/report/view" not in formatted


@pytest.fixture
def rich_search_result():
    """SearchResult with every optional field populated."""
    return SearchResult(
        content="This is the chunk content about elections.",
        score=0.85,
        chunk_id="chunk-123",
        document_id="doc-456",
        document_uri="file:///docs/report.pdf",
        document_title="Annual Report 2024",
        headings=["Chapter 1", "Section 1.1", "Elections"],
        labels=["paragraph", "table"],
        page_numbers=[1, 2],
    )


@pytest.mark.parametrize(
    "kwargs,present,absent",
    [
        # A rank is supplied, so the raw RRF score is withheld from the agent.
        ({"rank": 1, "total": 5}, "[rank 1 of 5]", "score:"),
        ({}, "(score: 0.85)", "[rank"),
    ],
    ids=["with_rank", "score_fallback"],
)
def test_search_result_format_for_agent_rank_vs_score(
    rich_search_result, kwargs, present, absent
):
    """format_for_agent shows a rank when given one, else falls back to score."""
    formatted = rich_search_result.format_for_agent(**kwargs)

    assert present in formatted
    assert absent not in formatted
    assert "[chunk-123]" in formatted
    assert (
        'Source: "Annual Report 2024" > Chapter 1 > Section 1.1 > Elections'
        in formatted
    )
    assert "Type: table" in formatted  # table has higher priority than paragraph
    assert "Content:\nThis is the chunk content about elections." in formatted


def test_search_result_format_for_agent_picture_captions():
    """Picture captions render as labelled lines so the model can correlate them
    with binary parts (BinaryContent.identifier doesn't survive serialization
    to the OpenAI vision API; insertion order is the only reliable signal)."""
    result = SearchResult(
        content="...surrounding text...",
        score=0.5,
        chunk_id="chunk-xyz",
        labels=["picture", "text"],
        picture_captions={
            "#/pictures/0": "Figure 1. Results from each model.",
            "#/pictures/1": "Figure 2. Projected annual emissions.",
        },
    )

    formatted = result.format_for_agent(rank=1, total=2)

    lines = formatted.splitlines()
    cap0 = next(
        i for i, line in enumerate(lines) if "Figure caption (#/pictures/0)" in line
    )
    cap1 = next(
        i for i, line in enumerate(lines) if "Figure caption (#/pictures/1)" in line
    )
    content_line = next(
        i for i, line in enumerate(lines) if line.startswith("Content:")
    )
    assert cap0 < cap1 < content_line
    assert "Figure 1. Results from each model." in formatted
    assert "Figure 2. Projected annual emissions." in formatted


def test_search_result_format_for_agent_no_captions_no_line():
    """Without picture_captions, no caption lines appear (zero-overhead for text chunks)."""
    result = SearchResult(
        content="prose",
        score=0.5,
        chunk_id="chunk-abc",
        labels=["text"],
    )
    formatted = result.format_for_agent(rank=1, total=1)
    assert "Figure caption" not in formatted


@pytest.mark.parametrize(
    "kwargs,present,absent",
    [
        ({"rank": 2}, "[rank 2]", ["score:"]),
        # No structural metadata at all, so no Source:/Type: lines are emitted.
        ({}, "(score: 0.72)", ["[rank", "Source:", "Type:"]),
    ],
    ids=["rank_only", "minimal"],
)
def test_search_result_format_for_agent_minimal(kwargs, present, absent):
    """A result carrying only content/score/chunk_id formats without metadata lines."""
    result = SearchResult(
        content="Some content here.",
        score=0.72,
        chunk_id="chunk-abc",
    )

    formatted = result.format_for_agent(**kwargs)

    assert "[chunk-abc]" in formatted
    assert present in formatted
    for token in absent:
        assert token not in formatted
    assert "Content:\nSome content here." in formatted


@pytest.mark.parametrize(
    "fields,expected_source",
    [
        ({"document_title": "My Document"}, 'Source: "My Document"'),
        (
            {"headings": ["Introduction", "Background"]},
            "Source: Introduction > Background",
        ),
    ],
    ids=["title_only", "headings_only"],
)
def test_search_result_format_for_agent_source_line(fields, expected_source):
    """The Source: line is built from the title, the headings, or both."""
    result = SearchResult(
        content="Content text.",
        score=0.60,
        chunk_id="chunk-xyz",
        **fields,
    )

    assert expected_source in result.format_for_agent()


@pytest.mark.parametrize(
    "labels,expected",
    [
        (["paragraph", "table", "text"], "table"),
        (["paragraph", "code"], "code"),
        (["list_item", "code"], "code"),
        (["text", "list_item"], "list_item"),
        # No structural label: falls through to the first label.
        (["paragraph", "text"], "paragraph"),
        ([], None),
    ],
)
def test_search_result_get_primary_label(labels, expected):
    """Test _get_primary_label prioritization."""
    result = SearchResult(content="x", score=0.5, labels=labels)
    assert result._get_primary_label() == expected


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "metadata,content,expected_content_fts",
    [
        (
            {"headings": ["Chapter 1", "Section 1.1"]},
            "This is the raw chunk content.",
            "Chapter 1\nSection 1.1\nThis is the raw chunk content.",
        ),
        ({}, "Plain content without headings.", "Plain content without headings."),
    ],
    ids=["populated", "without_headings"],
)
async def test_chunk_content_fts(temp_db_path, metadata, content, expected_content_fts):
    """content_fts holds the contextualized content while content stays raw."""
    from haiku.rag.embeddings import get_embedder

    async with HaikuRAG(
        db_path=temp_db_path, config=get_config(), create=True
    ) as client:
        chunk = Chunk(
            document_id="test-doc",
            content=content,
            metadata=metadata,
            order=0,
        )

        embedder = get_embedder(get_config())
        embedding = (await embedder.embed_documents([chunk.content]))[0]
        chunk.embedding = embedding

        await client.chunk_repository.create(chunk)

        records = (
            await client.store.chunks_table.query()
            .where(f"id = '{chunk.id}'")
            .limit(1)
            .to_arrow()
        ).to_pylist()

        assert len(records) == 1
        record = records[0]

        assert record["content"] == content
        assert record["content_fts"] == expected_content_fts


async def _import_one(client) -> None:
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.labels import DocItemLabel

    doc = DoclingDocument(name="one")
    doc.add_text(label=DocItemLabel.TEXT, text="a document about gardens")
    await client.import_document(
        doc,
        [
            Chunk(
                content="a document about gardens",
                embedding=[0.1] * get_config().embeddings.model.vector_dim,
                order=0,
            )
        ],
        uri="test://one",
    )


async def _add_legacy_row(client) -> None:
    """A row written without index maintenance, as older releases wrote them."""
    await client.store.chunks_table.add(
        [
            client.store.ChunkRecord(
                document_id="doc-legacy",
                content="a document about gardens",
                content_fts="a document about gardens",
                metadata="{}",
                order=0,
                vector=[0.1] * get_config().embeddings.model.vector_dim,
            )
        ]
    )


async def test_fts_search_warns_when_index_covers_no_rows(temp_db_path):
    """A database whose FTS index predates its rows covers none of them;
    searching that state warns, once."""
    import logging

    from lancedb.index import FTS

    from haiku.rag.store.repositories import chunk as chunk_module

    async with HaikuRAG(
        db_path=temp_db_path, config=get_config(), create=True
    ) as client:
        await client.store.chunks_table.create_index(
            "content_fts", config=FTS(with_position=True, remove_stop_words=False)
        )
        await _add_legacy_row(client)

        with capture_logs(chunk_module.logger, logging.WARNING) as records:
            await client.chunk_repository.search("gardens", search_type="fts")
            await client.chunk_repository.search("gardens", search_type="fts")

        warned = [r for r in records if "covers 0 rows" in r.getMessage()]
        assert len(warned) == 1


async def test_fts_search_warns_when_index_is_missing(temp_db_path, monkeypatch):
    import logging

    from lancedb.table import AsyncTable

    from haiku.rag.store.repositories import chunk as chunk_module

    original = AsyncTable.list_indices

    async def no_chunk_indices(self):
        if self.name == "chunks":
            return []
        return await original(self)

    async with HaikuRAG(
        db_path=temp_db_path, config=get_config(), create=True
    ) as client:
        await _import_one(client)
        monkeypatch.setattr(AsyncTable, "list_indices", no_chunk_indices)

        with capture_logs(chunk_module.logger, logging.WARNING) as records:
            await client.chunk_repository.search("gardens", search_type="fts")

        warned = [
            r.getMessage()
            for r in records
            if "No full-text search index" in r.getMessage()
        ]
        assert len(warned) == 1
        assert "rebuild --embed-only" in warned[0]
        assert "vacuum" not in warned[0]


async def test_fts_search_does_not_warn_when_index_covers_rows(temp_db_path):
    import logging

    from haiku.rag.store.repositories import chunk as chunk_module

    async with HaikuRAG(
        db_path=temp_db_path, config=get_config(), create=True
    ) as client:
        await _import_one(client)
        await client.store.vacuum(retention_seconds=0)

        with capture_logs(chunk_module.logger, logging.WARNING) as records:
            results = await client.chunk_repository.search("gardens", search_type="fts")

        assert results
        assert not records


async def test_fts_coverage_check_failure_does_not_break_search(temp_db_path):
    """The coverage check is a diagnostic: a metadata failure must not take
    the search down with it."""
    from lancedb.table import AsyncTable

    async def boom(self):
        raise RuntimeError("metadata unavailable")

    async with HaikuRAG(
        db_path=temp_db_path, config=get_config(), create=True
    ) as client:
        await _import_one(client)

        with patch.object(AsyncTable, "list_indices", boom):
            results = await client.chunk_repository.search("gardens", search_type="fts")

        assert results


async def test_create_assigns_the_id_when_index_maintenance_fails(temp_db_path):
    """The row is committed before the index is ensured, so the chunk carries
    the id it was written with even when that ensure fails."""
    from haiku.rag.store.repositories import chunk as chunk_module

    async with HaikuRAG(
        db_path=temp_db_path, config=get_config(), create=True
    ) as client:
        repo = client.chunk_repository

        async def boom(*_args, **_kwargs):
            raise RuntimeError("index build failed")

        chunk = Chunk(
            document_id="doc-1",
            content="a chunk about gardens",
            embedding=[0.1] * get_config().embeddings.model.vector_dim,
            order=0,
        )
        with patch.object(chunk_module, "ensure_indexes", boom):
            with pytest.raises(RuntimeError, match="index build failed"):
                await repo.create(chunk)

        assert chunk.id is not None
        assert await client.store.chunks_table.count_rows() == 1


async def test_fts_search_on_an_empty_table_does_not_suppress_later_warnings(
    temp_db_path,
):
    """An empty table proves nothing about coverage, so searching it must not
    spend the once-per-repository check."""
    import logging

    from lancedb.index import FTS

    from haiku.rag.store.repositories import chunk as chunk_module

    async with HaikuRAG(
        db_path=temp_db_path, config=get_config(), create=True
    ) as client:
        await client.store.chunks_table.create_index(
            "content_fts", config=FTS(with_position=True, remove_stop_words=False)
        )
        await client.chunk_repository.search("gardens", search_type="fts")
        await _add_legacy_row(client)

        with capture_logs(chunk_module.logger, logging.WARNING) as records:
            await client.chunk_repository.search("gardens", search_type="fts")

        assert any("covers 0 rows" in r.getMessage() for r in records)


async def test_fts_search_does_not_warn_on_an_empty_table(temp_db_path):
    import logging

    from haiku.rag.store.repositories import chunk as chunk_module

    async with HaikuRAG(
        db_path=temp_db_path, config=get_config(), create=True
    ) as client:
        with capture_logs(chunk_module.logger, logging.WARNING) as records:
            await client.chunk_repository.search("gardens", search_type="fts")

        assert not records


async def test_search_populates_embeddings_only_when_asked(temp_db_path):
    """Vectors ride the result frame either way; the per-chunk lists are
    materialized only for the caller that reads them (federated fusion)."""
    async with HaikuRAG(
        db_path=temp_db_path, config=get_config(), create=True
    ) as client:
        await _import_one(client)
        await client.store.vacuum(retention_seconds=0)

        plain = await client.chunk_repository.search("gardens", search_type="fts")
        with_vectors = await client.chunk_repository.search(
            "gardens", search_type="fts", with_vectors=True
        )

    assert plain and all(chunk.embedding is None for chunk, _ in plain)
    assert with_vectors and all(
        chunk.embedding is not None for chunk, _ in with_vectors
    )


@pytest.mark.vcr()
async def test_chunk_repository_get_by_id_and_list_all_pagination(
    qa_corpus: list[dict[str, str]], temp_db_path
):
    """get_by_id resolves a stored chunk; list_all honours limit and offset."""
    async with HaikuRAG(
        db_path=temp_db_path, config=get_config(), create=True
    ) as client:
        # A corpus document is long enough to chunk more than once, which is
        # what makes the offset assertion below meaningful.
        doc = await client.create_document(content=qa_corpus[0]["document_extracted"])
        assert doc.id is not None

        stored = await client.chunk_repository.get_by_document_id(doc.id)
        assert stored

        fetched = await client.get_chunk_by_id(stored[0].id)
        assert fetched is not None
        assert fetched.id == stored[0].id
        assert fetched.content == stored[0].content

        assert await client.get_chunk_by_id("no-such-chunk") is None

        everything = await client.chunk_repository.list_all()
        assert len(everything) == len(stored)

        first = await client.chunk_repository.list_all(limit=1)
        assert len(first) == 1
        assert first[0].id == everything[0].id

        # Fail loudly if the fixture stops producing enough chunks to page.
        assert len(everything) >= 2

        second = await client.chunk_repository.list_all(limit=1, offset=1)
        assert len(second) == 1
        assert second[0].id == everything[1].id


@pytest.mark.vcr()
async def test_chunk_search_returns_empty_for_blank_query(temp_db_path):
    """A blank query with no precomputed vector short-circuits before searching."""
    async with HaikuRAG(
        db_path=temp_db_path, config=get_config(), create=True
    ) as client:
        await client.create_document(content="Searchable body about elections.")

        # Positive control: the corpus is non-empty, so [] is a real decision
        # rather than the answer to every query.
        assert await client.chunk_repository.search("elections")
        assert await client.chunk_repository.search("   ") == []


@pytest.mark.vcr()
async def test_chunk_search_with_precomputed_vector_skips_text_query(temp_db_path):
    """The image-as-query path searches vector-only using a stored embedding."""
    async with HaikuRAG(
        db_path=temp_db_path, config=get_config(), create=True
    ) as client:
        doc = await client.create_document(content="Vector-only search target.")
        assert doc.id is not None

        rows = (await client.store.chunks_table.query().limit(1).to_arrow()).to_pylist()
        stored_vector = list(rows[0]["vector"])

        results = await client.chunk_repository.search("", query_vector=stored_vector)

        assert results
        assert any(c.document_id == doc.id for c, _ in results)


@pytest.mark.parametrize(
    "metric,expected_ids",
    [
        ("cosine", ["cosine-best", "l2-best"]),
        ("l2", ["l2-best", "cosine-best"]),
    ],
)
async def test_vector_metric_ranking_matches_before_and_after_index(
    temp_db_path, metric, expected_ids
):
    """Flat and indexed searches honor the configured metric.

    The two leading vectors are deliberately not unit-normalized: cosine and
    L2 must rank them in opposite orders, so LanceDB's flat-search L2 default
    cannot accidentally satisfy the cosine case.
    """
    from datetime import timedelta

    from lancedb.index import IvfPq

    config = get_config().model_copy(deep=True)
    config.embeddings.model.vector_dim = 8
    config.search.vector_index_metric = metric

    def row(chunk_id, vector, order):
        return {
            "id": chunk_id,
            "document_id": "doc-1",
            "content": chunk_id,
            "content_fts": chunk_id,
            "metadata": "{}",
            "order": order,
            "vector": vector,
        }

    rows = [
        row("cosine-best", [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0),
        row("l2-best", [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1),
    ]
    # IVF-PQ needs 256 training rows. These are far from the query under both
    # metrics and vary enough to train the quantizer without entering Top-K.
    rows.extend(
        row(
            f"filler-{i}",
            [0.0, *[100.0 + i + j for j in range(7)]],
            i + 2,
        )
        for i in range(254)
    )

    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        await client.store.chunks_table.add(rows)
        query_vector = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        async def top_ids():
            results = await client.chunk_repository.search(
                "", limit=2, search_type="vector", query_vector=query_vector
            )
            return [chunk.id for chunk, _score in results]

        flat_ids = await top_ids()

        await client.store.chunks_table.create_index(
            "vector",
            config=IvfPq(distance_type=metric, num_partitions=1, num_sub_vectors=1),
            replace=True,
        )
        await client.store.chunks_table.wait_for_index(
            ["vector_idx"], timeout=timedelta(minutes=1)
        )
        indexed_ids = await top_ids()

    assert flat_ids == expected_ids
    assert indexed_ids == expected_ids


async def test_get_chunk_ids_by_self_ref_grouped_without_documents(temp_db_path):
    async with HaikuRAG(
        db_path=temp_db_path, config=get_config(), create=True
    ) as client:
        assert await client.chunk_repository.get_chunk_ids_by_self_ref_grouped([]) == {}


async def test_process_search_results_rejects_unknown_score_column(temp_db_path):
    """A result frame with no recognised score column is a programming error."""
    import pandas as pd

    async with HaikuRAG(
        db_path=temp_db_path, config=get_config(), create=True
    ) as client:

        class _Frame:
            async def to_pandas(self):
                return pd.DataFrame([{"id": "c1", "content": "x", "metadata": "{}"}])

        with pytest.raises(ValueError, match="Unknown search result format"):
            await client.chunk_repository._process_search_results(_Frame())


@pytest.mark.parametrize("search_type", ["vector", "hybrid"])
async def test_search_applies_configured_nprobes(
    temp_db_path, monkeypatch, search_type
):
    """The configured probe count reaches the vector query."""
    from lancedb.query import AsyncVectorQueryBase

    probed: list[int] = []
    original = AsyncVectorQueryBase.nprobes

    def record(self, nprobes):
        probed.append(nprobes)
        return original(self, nprobes)

    monkeypatch.setattr(AsyncVectorQueryBase, "nprobes", record)

    config = get_config()
    config.search.vector_nprobes = 7
    async with HaikuRAG(db_path=temp_db_path, config=config, create=True) as client:
        await _import_one(client)
        await client.chunk_repository.search(
            "gardens",
            search_type=search_type,
            query_vector=[0.1] * config.embeddings.model.vector_dim,
        )

    assert probed == [7]
