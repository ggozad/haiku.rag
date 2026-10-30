import sys
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from haiku.rag.chunkers import get_chunker
from haiku.rag.chunkers.docling_local import DoclingLocalChunker, _get_tokenizer
from haiku.rag.chunkers.docling_serve import DoclingServeChunker
from haiku.rag.config import AppConfig, get_config
from haiku.rag.converters import get_converter


async def test_local_chunker(qa_corpus: list[dict[str, str]]):
    """Test DoclingLocalChunker with real document."""
    chunker = DoclingLocalChunker()
    doc_text = qa_corpus[0]["document_extracted"]

    # Convert text to DoclingDocument
    converter = get_converter(get_config())
    doc = await converter.convert_text(doc_text, name="test.md")

    chunks = await chunker.chunk(doc)

    # Ensure that the text is split into multiple chunks
    assert len(chunks) > 1

    # Load tokenizer for verification
    tokenizer = _get_tokenizer(chunker.tokenizer_name)

    # Ensure that chunks are reasonably sized (allowing more flexibility for structure-aware chunking)
    total_tokens = 0
    for chunk in chunks:
        encoded_tokens = tokenizer.encode(chunk.content, add_special_tokens=False)
        token_count = len(encoded_tokens)
        total_tokens += token_count

        # Each chunk should be reasonably sized (allowing more flexibility than the old strict limits)
        assert (
            token_count <= chunker.chunk_size * 1.2
        )  # Allow some flexibility for semantic boundaries
        assert token_count > 5  # Ensure chunks aren't too small

    # Ensure that all chunks together contain roughly the same content as original
    original_tokens = len(tokenizer.encode(doc_text, add_special_tokens=False))

    # Due to structure-aware chunking, we might have some variation in token count
    # but it should be reasonable
    assert abs(total_tokens - original_tokens) <= original_tokens * 0.1


async def test_local_chunker_none_document():
    """Test DoclingLocalChunker returns empty list for None document."""
    chunker = DoclingLocalChunker()
    assert await chunker.chunk(None) == []


async def test_local_chunker_runs_off_event_loop_thread():
    """Chunking is CPU-bound; verify it runs in a worker thread."""
    import threading
    from unittest.mock import patch

    chunker = DoclingLocalChunker()
    event_loop_thread = threading.current_thread()
    called_from: list[threading.Thread] = []

    original = chunker._chunk_sync

    def recording_chunk_sync(_self, document):
        called_from.append(threading.current_thread())
        return original(document)

    converter = get_converter(get_config())
    doc = await converter.convert_text("# Hello\n\nWorld", name="test.md")

    with patch.object(DoclingLocalChunker, "_chunk_sync", recording_chunk_sync):
        await chunker.chunk(doc)

    assert called_from, "_chunk_sync was never called"
    assert called_from[0] is not event_loop_thread, (
        "_chunk_sync ran on the event-loop thread; "
        "it must be dispatched via asyncio.to_thread"
    )


@pytest.mark.parametrize(
    ("name", "chunker_class"),
    [("docling-local", DoclingLocalChunker), ("docling-serve", DoclingServeChunker)],
)
def test_get_chunker_selects_the_configured_chunker(name, chunker_class):
    config = AppConfig()
    config.processing.chunker = name
    assert isinstance(get_chunker(config), chunker_class)


def test_get_chunker_invalid():
    """Test factory raises error for invalid chunker."""
    config = AppConfig()
    # Deliberately past validation: pins the factory's defensive raise.
    config.processing.chunker = "invalid-chunker"  # ty: ignore[invalid-assignment]
    with pytest.raises(ValueError, match="Unsupported chunker"):
        get_chunker(config)


def test_tokenizer_cached_across_chunker_instances():
    """Repeated DoclingLocalChunker instantiations share one loaded tokenizer.

    Each `AutoTokenizer.from_pretrained` call triggers an `HfApi.model_info`
    HTTP request to check for revision drift. Batch ingest creates one
    chunker per document, which without caching hits HF Hub's 1000-per-5min
    limit and crashes with HTTP 429.
    """
    _get_tokenizer.cache_clear()

    DoclingLocalChunker()
    DoclingLocalChunker()
    DoclingLocalChunker()

    info = _get_tokenizer.cache_info()
    assert info.misses == 1
    assert info.hits == 2


async def test_local_chunker_retains_no_token_counters():
    """Splitting an oversized paragraph leaves no semchunk token counter behind."""
    from semchunk import semchunk

    paragraph = " ".join(
        f"Sentence {i} about the quarterly report." for i in range(300)
    )
    converter = get_converter(get_config())
    doc = await converter.convert_text(f"# Title\n\n{paragraph}", name="test.md")

    chunks = await DoclingLocalChunker().chunk(doc)

    assert len(chunks) > 1
    assert semchunk._memoized_token_counters == {}


async def test_local_chunker_hierarchical_without_semchunk(monkeypatch):
    """Hierarchical chunking works when semchunk is not installed."""
    monkeypatch.setitem(sys.modules, "semchunk", None)
    monkeypatch.delitem(sys.modules, "semchunk.semchunk", raising=False)
    config = AppConfig()
    config.processing.chunker_type = "hierarchical"
    converter = get_converter(config)
    doc = await converter.convert_text("# Title\n\nSome text.", name="test.md")

    chunks = await DoclingLocalChunker(config).chunk(doc)

    assert [chunk.content for chunk in chunks] == ["Some text."]


def test_local_chunker_invalid_type():
    """Test DoclingLocalChunker raises error for invalid chunker_type."""
    config = AppConfig()
    config.processing.chunker_type = "invalid-type"  # ty: ignore[invalid-assignment]
    with pytest.raises(ValueError, match="Unsupported chunker_type"):
        DoclingLocalChunker(config)


async def test_local_chunker_markdown_tables():
    """Test DoclingLocalChunker with markdown table serialization."""
    markdown_with_table = """# Test Document

| Column 1 | Column 2 |
|----------|----------|
| Value A  | Value B  |
| Value D  | Value E  |
"""

    converter = get_converter(get_config())
    doc = await converter.convert_text(markdown_with_table, name="test.md")

    # Test with markdown tables enabled
    config_md = AppConfig()
    config_md.processing.chunking_use_markdown_tables = True
    chunker_md = DoclingLocalChunker(config_md)
    chunks_md = await chunker_md.chunk(doc)

    # Should contain markdown table format
    assert any("|" in chunk.content for chunk in chunks_md)
    assert any("Column 1" in chunk.content for chunk in chunks_md)

    # Test with markdown tables disabled (narrative format)
    config_narrative = AppConfig()
    config_narrative.processing.chunking_use_markdown_tables = False
    chunker_narrative = DoclingLocalChunker(config_narrative)
    chunks_narrative = await chunker_narrative.chunk(doc)

    # Should contain narrative format (no pipe characters in table)
    table_content = [
        chunk.content for chunk in chunks_narrative if "Value" in chunk.content
    ][0]
    # Narrative format uses commas, not pipes for table structure
    assert "," in table_content and "|" not in table_content


async def test_local_chunker_settings_order_and_metadata():
    """The chunker takes its size and tokenizer from the config, numbers chunks
    in order and extracts their metadata."""
    sample_md = """# Chapter 1: Introduction

This is the first paragraph of the introduction.

## Section 1.1: Background

Here is some background information.

| Header 1 | Header 2 |
|----------|----------|
| Value 1  | Value 2  |
"""
    converter = get_converter(get_config())
    doc = await converter.convert_text(sample_md, name="test.md")
    config = AppConfig()
    config.processing.chunk_size = 128
    config.processing.chunking_tokenizer = "Qwen/Qwen3-Embedding-0.6B"

    chunker = DoclingLocalChunker(config)
    chunks = await chunker.chunk(doc)

    assert chunker.chunk_size == 128
    assert chunker.tokenizer_name == "Qwen/Qwen3-Embedding-0.6B"
    assert len(chunks) > 1
    assert [chunk.order for chunk in chunks] == list(range(len(chunks)))

    # Check that at least one chunk has doc_item_refs
    all_refs = []
    all_labels = []
    all_headings = []
    for chunk in chunks:
        meta = chunk.get_chunk_metadata()
        all_refs.extend(meta.doc_item_refs)
        all_labels.extend(meta.labels)
        if meta.headings:
            all_headings.extend(meta.headings)

    # Should have JSON pointer refs like #/texts/0, #/tables/0
    assert len(all_refs) > 0
    assert any(ref.startswith("#/") for ref in all_refs)

    # Should have labels
    assert len(all_labels) > 0
    assert "text" in all_labels or "table" in all_labels

    # Should have headings
    assert len(all_headings) > 0
    assert any("Chapter" in h or "Section" in h for h in all_headings)


async def test_inline_markup_survives_chunking():
    """Inline code stays inline in the chunk text, and a heading carrying a
    link names the breadcrumb."""
    sample_md = (
        "# Title\n\n"
        "## [0.68.0] - 2026-07-24\n\n"
        "By default, `haiku.rag` uses the configured embedder.\n\n"
        "[0.68.0]: https://github.com/ggozad/haiku.rag/releases\n"
    )
    converter = get_converter(get_config())
    doc = await converter.convert_text(sample_md, name="test.md")

    chunks = await DoclingLocalChunker().chunk(doc)

    assert any(
        "By default, `haiku.rag` uses the configured embedder." in chunk.content
        for chunk in chunks
    )
    assert all("```" not in chunk.content for chunk in chunks)
    headings = [
        h for chunk in chunks for h in (chunk.get_chunk_metadata().headings or [])
    ]
    assert "0.68.0 - 2026-07-24" in headings


def create_async_workflow_mocks(
    result_data: dict, task_id: str = "test-task-123"
) -> tuple[Mock, Mock, Mock]:
    """Create mock responses for docling-serve async workflow."""
    submit_response = Mock()
    submit_response.status_code = 200
    submit_response.json.return_value = {"task_id": task_id, "task_status": "pending"}
    submit_response.raise_for_status = Mock()

    poll_response = Mock()
    poll_response.status_code = 200
    poll_response.json.return_value = {"task_id": task_id, "task_status": "success"}
    poll_response.raise_for_status = Mock()

    result_response = Mock()
    result_response.status_code = 200
    result_response.json.return_value = result_data
    result_response.raise_for_status = Mock()

    return submit_response, poll_response, result_response


@pytest.fixture
def serve_chunks(monkeypatch):
    """Patch httpx.AsyncClient so docling-serve answers one chunking workflow
    with the given result and return the client mock."""

    def serve(result_data: dict) -> AsyncMock:
        submit_resp, poll_resp, result_resp = create_async_workflow_mocks(result_data)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=submit_resp)
        mock_client.get = AsyncMock(side_effect=[poll_resp, result_resp])
        client_class = MagicMock()
        client_class.return_value.__aenter__.return_value = mock_client
        monkeypatch.setattr(
            "haiku.rag.providers.docling_serve.httpx.AsyncClient", client_class
        )
        return mock_client

    return serve


class TestDoclingServeChunker:
    """Tests for DoclingServeChunker (mocked)."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = AppConfig()
        config.providers.docling_serve.base_url = "http://localhost:5001"
        config.providers.docling_serve.api_key = ""
        config.processing.chunk_size = 256
        config.processing.chunking_tokenizer = "Qwen/Qwen3-Embedding-0.6B"
        return config

    @pytest.fixture
    def chunker(self, config):
        """Create DoclingServeChunker instance."""
        return DoclingServeChunker(config)

    async def test_chunk_success(self, chunker, serve_chunks):
        """Test successful chunking via docling-serve async workflow."""
        result_data = {
            "chunks": [
                {"text": "Chunk 1", "chunk_index": 0},
                {"text": "Chunk 2", "chunk_index": 1},
            ]
        }
        mock_client = serve_chunks(result_data)

        # Create a simple document
        converter = get_converter(get_config())
        doc = await converter.convert_text("# Test\n\nContent", name="test.md")

        chunks = await chunker.chunk(doc)
        assert len(chunks) == 2
        assert chunks[0].content == "Chunk 1"
        assert chunks[1].content == "Chunk 2"
        mock_client.post.assert_called_once()

    async def test_chunk_hierarchical_endpoint(self, config, serve_chunks):
        """Test that hierarchical chunker uses correct endpoint."""
        config.processing.chunker_type = "hierarchical"
        chunker = DoclingServeChunker(config)

        result_data = {"chunks": [{"text": "Chunk 1", "chunk_index": 0}]}
        mock_client = serve_chunks(result_data)

        converter = get_converter(get_config())
        doc = await converter.convert_text("# Test", name="test.md")
        await chunker.chunk(doc)

        call_args = mock_client.post.call_args
        assert "/v1/chunk/hierarchical/file/async" in call_args[0][0]

    def test_chunking_request_sends_the_chunking_options(self, config):
        config.processing.chunk_size = 512
        config.processing.chunking_merge_peers = False
        config.processing.chunking_use_markdown_tables = True

        data = DoclingServeChunker(config)._build_chunking_data()

        assert data["chunking_max_tokens"] == "512"
        assert data["chunking_merge_peers"] == "false"
        assert data["chunking_use_markdown_tables"] == "true"

    @pytest.mark.parametrize(
        ("pictures", "ocr_lang"),
        [
            ("none", ["en"]),
            ("image", ["en"]),
            ("description", ["en"]),
            ("image", []),
        ],
    )
    def test_chunking_request_sends_the_conversion_options(
        self, config, pictures, ocr_lang
    ):
        """Chunk requests carry the conversion request's pipeline options and
        omit an empty OCR language list."""
        from haiku.rag.converters.docling_serve import DoclingServeConverter

        config.processing.pictures = pictures
        config.processing.conversion_options.table_mode = "fast"
        config.processing.conversion_options.ocr_lang = ocr_lang
        conversion = DoclingServeConverter(config)._build_conversion_data()
        chunking = DoclingServeChunker(config)._build_chunking_data()

        output_only = {"to_formats", "target_type"}
        sent_for_chunking = {
            k.removeprefix("convert_"): v
            for k, v in chunking.items()
            if k.startswith("convert_")
        }
        assert sent_for_chunking == {
            k: v for k, v in conversion.items() if k not in output_only
        }
        assert chunking["convert_do_ocr"] == "true"
        assert chunking["convert_force_ocr"] == "false"
        assert chunking["convert_ocr_engine"] == "auto"
        assert ("convert_ocr_lang" in chunking) == bool(ocr_lang)

    async def test_chunk_document_failure_status(self, chunker, serve_chunks):
        """Test that document-level failure status raises ValueError."""
        result_data = {
            "chunks": [],
            "documents": [
                {
                    "kind": "ExportResult",
                    "status": "failure",
                    "errors": ["Schema version mismatch"],
                }
            ],
        }
        serve_chunks(result_data)

        converter = get_converter(get_config())
        doc = await converter.convert_text("# Test", name="test.md")

        with pytest.raises(ValueError, match="Chunking failed"):
            await chunker.chunk(doc)

    async def test_chunk_document_success_empty_chunks(self, chunker, serve_chunks):
        """Test that successful status with empty chunks returns empty list."""
        result_data = {
            "chunks": [],
            "documents": [
                {
                    "kind": "ExportResult",
                    "status": "success",
                    "errors": [],
                }
            ],
        }
        serve_chunks(result_data)

        converter = get_converter(get_config())
        doc = await converter.convert_text("# Test", name="test.md")

        chunks = await chunker.chunk(doc)
        assert chunks == []

    async def test_chunk_metadata_extraction(self, chunker, serve_chunks):
        """Test that metadata is correctly extracted from API response.

        Labels are resolved from the DoclingDocument using the refs, so we need
        to create a document with matching structure for the mocked API response.
        """
        result_data = {
            "chunks": [
                {
                    "text": "Chapter 1\nThis is content.",
                    "doc_items": ["#/texts/0", "#/texts/1"],
                    "headings": ["Chapter 1"],
                    "page_numbers": [1],
                },
                {
                    "text": "Table content here.",
                    "doc_items": ["#/tables/0"],
                    "headings": ["Chapter 1", "Section 1.1"],
                    "page_numbers": [1, 2],
                },
            ]
        }
        serve_chunks(result_data)

        # Create a document with texts and tables that match the mocked refs
        converter = get_converter(get_config())
        doc = await converter.convert_text(
            """# Chapter 1

This is content.

| Col1 | Col2 |
|------|------|
| A    | B    |
""",
            name="test.md",
        )

        chunks = await chunker.chunk(doc)

        assert len(chunks) == 2

        # First chunk - labels resolved from document
        assert chunks[0].content == "Chapter 1\nThis is content."
        meta0 = chunks[0].get_chunk_metadata()
        assert meta0.doc_item_refs == ["#/texts/0", "#/texts/1"]
        # texts[0] is title (# heading), texts[1] is text (paragraph)
        assert meta0.labels == ["title", "text"]
        assert meta0.headings == ["Chapter 1"]
        assert meta0.page_numbers == [1]

        # Second chunk - label resolved from document
        assert chunks[1].content == "Table content here."
        meta1 = chunks[1].get_chunk_metadata()
        assert meta1.doc_item_refs == ["#/tables/0"]
        assert meta1.labels == ["table"]
        assert meta1.headings == ["Chapter 1", "Section 1.1"]
        assert meta1.page_numbers == [1, 2]

    async def test_chunk_serializes_document_off_event_loop_thread(
        self, chunker, serve_chunks
    ):
        """model_dump_json over a document carrying inlined base64 page/picture
        images is CPU-heavy and proportional to document size; it must run off
        the event-loop thread or it stalls every other worker's coroutine.

        A minimal fake document records the thread its model_dump_json runs on;
        the API response carries no doc_items so the document is touched only
        for serialization."""
        import threading

        result_data = {"chunks": [{"text": "Chunk 1", "chunk_index": 0}]}
        serve_chunks(result_data)

        event_loop_thread = threading.current_thread()
        called_from: list[threading.Thread] = []

        class FakeDoc:
            def model_dump_json(self):
                called_from.append(threading.current_thread())
                return "{}"

        chunks = await chunker.chunk(FakeDoc())

        assert len(chunks) == 1
        assert called_from, "model_dump_json was never called"
        assert called_from[0] is not event_loop_thread, (
            "DoclingDocument.model_dump_json ran on the event-loop thread; it "
            "must be dispatched via asyncio.to_thread"
        )


@pytest.mark.slow
@pytest.mark.usefixtures("docling_local_models")
@pytest.mark.vcr()
async def test_local_and_serve_chunkers_produce_same_output(doclaynet_first_page_pdf):
    """Pin converter and chunker parity across local and serve backends.

    Converter equality covers stable text and page fields; whole-document refs
    and classifier-threshold labels vary across platforms.
    """

    from haiku.rag.chunkers.docling_local import DoclingLocalChunker
    from haiku.rag.chunkers.docling_serve import DoclingServeChunker
    from haiku.rag.converters.docling_local import DoclingLocalConverter
    from haiku.rag.converters.docling_serve import DoclingServeConverter
    from haiku.rag.store.models.document_item import extract_items

    config = get_config().model_copy(deep=True)
    config.processing.conversion_options.do_ocr = False
    config.processing.chunk_size = 256
    config.processing.chunker_type = "hybrid"
    config.processing.chunking_merge_peers = True
    config.processing.chunking_use_markdown_tables = True

    local_doc = await DoclingLocalConverter(config).convert_file(
        doclaynet_first_page_pdf
    )
    serve_doc = await DoclingServeConverter(config).convert_file(
        doclaynet_first_page_pdf
    )

    local_items = extract_items("d", local_doc)
    serve_items = extract_items("d", serve_doc)
    assert len(local_items) == len(serve_items)
    assert [item.text for item in local_items] == [item.text for item in serve_items]
    assert [item.page_numbers for item in local_items] == [
        item.page_numbers for item in serve_items
    ]

    local_chunker = DoclingLocalChunker(config)
    serve_chunker = DoclingServeChunker(config)
    local_chunks = await local_chunker.chunk(local_doc)
    local_serve_doc_chunks = await local_chunker.chunk(serve_doc)
    serve_chunks = await serve_chunker.chunk(serve_doc)

    assert [chunk.content for chunk in local_chunks] == [
        chunk.content for chunk in serve_chunks
    ]
    assert len(local_serve_doc_chunks) == len(serve_chunks)

    for local, serve in zip(local_serve_doc_chunks, serve_chunks, strict=True):
        assert local.content == serve.content
        local_meta = local.get_chunk_metadata()
        serve_meta = serve.get_chunk_metadata()
        assert local_meta.doc_item_refs == serve_meta.doc_item_refs
        assert local_meta.labels == serve_meta.labels
        assert local_meta.headings == serve_meta.headings
        assert local_meta.page_numbers == serve_meta.page_numbers


@pytest.mark.slow
@pytest.mark.vcr()
async def test_serve_chunker_accepts_picture_laden_docling(doclaynet_first_page_pdf):
    """Round-trip a picture-bearing PDF through docling-serve's chunker.

    Catches the schema-shape failure we hit in production: when the
    converter routes through ``image_export_mode=referenced`` + ``target_type=zip``
    and rehydrates picture URIs as inline base64 ``data:`` URIs, the chunker
    must still accept the resulting docling JSON. A version-drift between
    the docling-serve container's docling-core and the local one would
    surface as ``Input document document.json is not valid``.
    """
    from haiku.rag.chunkers.docling_serve import DoclingServeChunker
    from haiku.rag.converters.docling_serve import DoclingServeConverter

    config = AppConfig()
    config.processing.conversion_options.do_ocr = False
    config.processing.chunk_size = 256
    config.processing.chunker_type = "hybrid"

    converter = DoclingServeConverter(config)
    pdf_path = doclaynet_first_page_pdf
    doc = await converter.convert_file(pdf_path)

    # Sanity: at least one picture has bytes inlined as a data URI — that's
    # the payload shape that broke us in production.
    pictures_with_uris = [
        p for p in doc.pictures if p.image is not None and p.image.uri
    ]
    assert pictures_with_uris, (
        "doclaynet.pdf is expected to yield at least one picture with bytes "
        "under pictures='image'"
    )
    sample_image = pictures_with_uris[0].image
    assert sample_image is not None  # narrow for ty
    assert str(sample_image.uri).startswith("data:image/"), (
        "Expected inline data: URI from referenced+zip rehydration"
    )

    serve_chunker = DoclingServeChunker(config)
    chunks = await serve_chunker.chunk(doc)

    assert len(chunks) > 0, "docling-serve chunker returned 0 chunks"


class TestDoclingServeChunkerRefResolution:
    """Label resolution of docling-serve doc_item refs."""

    @pytest.fixture
    def chunker(self):
        config = AppConfig()
        config.providers.docling_serve.base_url = "http://localhost:5001"
        config.processing.chunk_size = 256
        config.processing.chunking_tokenizer = "Qwen/Qwen3-Embedding-0.6B"
        return DoclingServeChunker(config)

    @pytest.fixture
    def document(self):
        from docling_core.types.doc.document import DoclingDocument

        return DoclingDocument.model_validate(
            {
                "name": "doc",
                "texts": [
                    {
                        "self_ref": "#/texts/0",
                        "text": "body",
                        "orig": "body",
                        "label": "paragraph",
                    }
                ],
                "tables": [],
                "pictures": [],
                "groups": [],
                "body": {"self_ref": "#/body", "children": []},
                "furniture": {"self_ref": "#/furniture", "children": []},
            }
        )

    @pytest.mark.parametrize(
        "ref",
        ["not-a-ref", "#/texts/999", "#/nope/0"],
        ids=["unparseable", "index_out_of_range", "unknown_collection"],
    )
    def test_unresolvable_ref_yields_no_label(self, document, ref):
        from haiku.rag.chunkers.docling_serve import _resolve_label_from_document

        assert _resolve_label_from_document(ref, document) is None

    def test_resolvable_ref_yields_label(self, document):
        from haiku.rag.chunkers.docling_serve import _resolve_label_from_document

        assert _resolve_label_from_document("#/texts/0", document) == "paragraph"

    async def test_chunk_of_none_returns_empty(self, chunker):
        assert await chunker.chunk(None) == []

    async def test_unresolvable_ref_is_kept_without_a_label(self, chunker, document):
        async def fake_chunk_api(_document):
            return [{"raw_text": "body", "doc_items": ["#/texts/0", "#/texts/999"]}]

        chunker._call_chunk_api = fake_chunk_api  # type: ignore[method-assign]

        chunks = await chunker.chunk(document)

        assert chunks[0].metadata["doc_item_refs"] == ["#/texts/0", "#/texts/999"]
        assert chunks[0].metadata["labels"] == ["paragraph"]
