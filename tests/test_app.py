"""Tests for the CLI application layer.

They stub the client and record the console: each test pins what the app asks
the client for and what it renders, without a database or a model.
"""

from unittest.mock import AsyncMock

import pytest
from rich.console import Console

from haiku.rag.app import HaikuRAGApp
from haiku.rag.client import RebuildMode
from haiku.rag.client.scope import DatabaseScope
from haiku.rag.config.models import (
    AppConfig,
    LanceDBConfig,
    PromptsConfig,
    StorageConfig,
)
from haiku.rag.store.models.chunk import Chunk, SearchResult
from haiku.rag.store.models.document import Document
from tests.conftest import for_path


@pytest.fixture
def client():
    return AsyncMock()


@pytest.fixture
def app(tmp_path, client, monkeypatch):
    class StubHaikuRAG:
        # run_mcp passes db_path positionally; every other caller uses kwargs.
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def _covering(cls, *args, **kwargs):
            return cls()

        async def __aenter__(self):
            return client

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr("haiku.rag.app.HaikuRAG", StubHaikuRAG)
    db = tmp_path / "db.lancedb"
    db.mkdir()
    application = HaikuRAGApp(scope=for_path(db), config=AppConfig())
    application.console = Console(record=True, width=200)
    return application


def out(app) -> str:
    # clear=False: export_text() empties the buffer by default, so a second
    # assertion in the same test would see nothing.
    return app.console.export_text(clear=False)


def _doc(content: str = "body text", **kwargs) -> Document:
    doc = Document(content=content, uri=kwargs.pop("uri", "test://doc"), **kwargs)
    doc.id = kwargs.get("id", "doc-1")
    return doc


async def test_a_listing_omits_the_fields_it_did_not_fetch(app, client):
    """`list` does not load content, and a document need not carry a uri, a
    title or metadata. Printing a header for each regardless announced fields
    the command declined to fetch or the document never had."""
    bare = Document(content="", uri=None)
    bare.id = "doc-bare"
    client.list_documents.return_value = [bare]

    await app.list_documents()

    printed = out(app)
    assert "doc-bare" in printed
    for absent in ("uri:", "title:", "meta:", "content:"):
        assert absent not in printed, absent


async def test_a_listing_prints_the_fields_it_has(app, client):
    client.list_documents.return_value = [
        _doc("body text", title="A title", metadata={"k": "v"})
    ]

    await app.list_documents()

    printed = out(app)
    assert "uri: test://doc" in printed
    assert "title: A title" in printed
    assert "meta:" in printed
    assert "content:" in printed


async def test_a_document_renders_fields_that_look_like_markup(app):
    """Uris, titles and metadata render as text, not markup."""
    doc = _doc(
        "body",
        uri="test://doc [/blue]",
        title="The [/bold] Title",
        metadata={"k": "[/red]"},
    )

    app._rich_print_document(doc)

    printed = out(app)
    assert "test://doc [/blue]" in printed
    assert "The [/bold] Title" in printed
    assert "[/red]" in printed


async def test_list_documents_prints_each_document(app, client):
    client.list_documents.return_value = [_doc("first"), _doc("second")]

    await app.list_documents()

    client.list_documents.assert_awaited_once_with(filter=None)
    assert "first" in out(app)
    assert "second" in out(app)


async def test_add_document_from_text_reports_the_new_id(app, client):
    client.create_document.return_value = _doc("added body")

    await app.add_document_from_text("added body", title="T", metadata={"k": "v"})

    client.create_document.assert_awaited_once_with(
        "added body", title="T", metadata={"k": "v"}
    )
    assert "doc-1 added successfully" in out(app)


async def test_add_document_from_source_reports_one_document(app, client):
    client.create_document_from_source.return_value = _doc("from file")

    await app.add_document_from_source("/tmp/x.md")

    assert "doc-1 added successfully" in out(app)


async def test_add_document_from_source_reports_a_directory_count(app, client):
    client.create_document_from_source.return_value = [_doc("a"), _doc("b")]

    await app.add_document_from_source("/tmp/dir")

    assert "2 documents added successfully" in out(app)


async def test_get_document_reports_a_missing_id(app, client):
    client.get_document_by_id.return_value = None

    await app.get_document("nope")

    assert "not found" in out(app)


async def test_get_document_prints_it_untruncated(app, client):
    client.get_document_by_id.return_value = _doc("line1\nline2\nline3\nline4")

    await app.get_document("doc-1")

    assert "line4" in out(app)


async def test_delete_document_confirms(app, client):
    client.delete_document.return_value = True

    await app.delete_document("doc-1")

    assert "deleted successfully" in out(app)


async def test_delete_document_reports_a_missing_id(app, client):
    client.delete_document.return_value = False

    await app.delete_document("nope")

    assert "not found" in out(app)


async def test_search_requires_a_query_or_an_image(app):
    await app.search()

    assert "Provide either a query argument or --image" in out(app)


async def test_search_refuses_both_a_query_and_an_image(app, tmp_path):
    await app.search(query="q", image=tmp_path / "i.png")

    assert "not both" in out(app)


async def test_search_type_needs_a_text_query(app, tmp_path):
    await app.search(image=tmp_path / "i.png", search_type="vector")

    assert "only for text queries" in out(app)


async def test_search_reports_no_results(app, client):
    client.search.return_value = []

    await app.search(query="q")

    assert "No results found" in out(app)


async def test_search_prints_results(app, client):
    client.search.return_value = [
        SearchResult(content="hit one", score=0.9, chunk_id="c1")
    ]

    await app.search(query="q", limit=3)

    client.search.assert_awaited_once_with("q", limit=3, filter=None, search_type=None)
    assert "hit one" in out(app)


async def test_a_result_names_its_database_only_across_several(tmp_path):
    """The label answers what this operation covers, not what is configured:
    after narrowing to one, the caller has already named it."""
    config = AppConfig(
        lancedb=LanceDBConfig(
            databases={
                "alpha": str(tmp_path / "a.lancedb"),
                "beta": str(tmp_path / "b.lancedb"),
            }
        )
    )
    hit = SearchResult(content="hit", score=0.9, chunk_id="c1", source="alpha")

    covering = HaikuRAGApp(scope=DatabaseScope.resolve(config), config=config)
    covering.console = Console(record=True, width=200)
    covering._rich_print_search_result(hit)
    assert "database: alpha" in covering.console.export_text()

    narrowed = HaikuRAGApp(
        scope=DatabaseScope.resolve(config, database_name="alpha"), config=config
    )
    narrowed.console = Console(record=True, width=200)
    narrowed._rich_print_search_result(hit)
    printed = narrowed.console.export_text()
    assert "hit" in printed
    assert "database:" not in printed


async def test_a_result_renders_names_that_look_like_markup(tmp_path):
    """Database names, titles, uris and headings render as text, not markup."""
    config = AppConfig(
        lancedb=LanceDBConfig(
            databases={
                "alpha [/red]": str(tmp_path / "a.lancedb"),
                "beta": str(tmp_path / "b.lancedb"),
            }
        )
    )
    hit = SearchResult(
        content="hit",
        score=0.9,
        chunk_id="c1",
        source="alpha [/red]",
        document_uri="test://doc [/blue]",
        document_title="The [/bold] Title",
        headings=["Chapter [/dim]"],
    )

    app = HaikuRAGApp(scope=DatabaseScope.resolve(config), config=config)
    app.console = Console(record=True, width=200)
    app._rich_print_search_result(hit)

    printed = app.console.export_text()
    assert "database: alpha [/red]" in printed
    assert "test://doc [/blue]" in printed
    assert "The [/bold] Title" in printed
    assert "Chapter [/dim]" in printed


async def test_search_by_image_reads_the_bytes(app, client, tmp_path):
    image = tmp_path / "query.png"
    image.write_bytes(b"pixels")
    client.search.return_value = []

    await app.search(image=image)

    assert client.search.await_args.args[0] == b"pixels"


async def test_visualize_reports_a_missing_chunk(app, client):
    client.get_chunk_by_id.return_value = None

    await app.visualize_chunk("nope")

    assert "not found" in out(app)


async def test_visualize_reports_no_grounding(app, client):
    client.get_chunk_by_id.return_value = Chunk(content="c", order=0)
    client.visualize_chunk.return_value = []

    await app.visualize_chunk("c1")

    assert "No visual grounding available" in out(app)


async def test_ask_prints_question_answer_and_citations(app, client, monkeypatch):
    client.ask.return_value = ("The answer.", [])

    async def no_citations(citations, client=None, full=False):
        return ["citation block"]

    monkeypatch.setattr("haiku.rag.app.format_citations_rich", no_citations)

    await app.ask("why?", filter="uri LIKE 'x%'")

    client.ask.assert_awaited_once_with("why?", filter="uri LIKE 'x%'", images=None)
    printed = out(app)
    assert "why?" in printed
    assert "The answer." in printed
    assert "citation block" in printed


async def test_ask_attaches_image_bytes(app, client, monkeypatch, tmp_path):
    image = tmp_path / "a.png"
    image.write_bytes(b"img")
    client.ask.return_value = ("answer", [])

    async def no_citations(citations, client=None, full=False):
        return []

    monkeypatch.setattr("haiku.rag.app.format_citations_rich", no_citations)

    await app.ask("why?", images=[image])

    assert client.ask.await_args.kwargs["images"] == [b"img"]


@pytest.mark.parametrize("verb", ["ask", "analyze"])
async def test_full_citations_reaches_the_formatter(app, client, monkeypatch, verb):
    """The flag has to survive the app layer, or the CLI switch does nothing."""
    client.ask.return_value = ("answer", [])
    result = AsyncMock()
    result.answer = "answer"
    result.citations = []
    client.analyze.return_value = result

    seen = []

    async def record(citations, client=None, full=False):
        seen.append(full)
        return []

    monkeypatch.setattr("haiku.rag.app.format_citations_rich", record)

    await getattr(app, verb)("why?", full_citations=True)
    await getattr(app, verb)("why?")

    assert seen == [True, False]


async def test_analyze_prints_the_answer(app, client, monkeypatch):
    result = AsyncMock()
    result.answer = "computed answer"
    result.citations = []
    client.analyze.return_value = result

    async def no_citations(citations, client=None, full=False):
        return []

    monkeypatch.setattr("haiku.rag.app.format_citations_rich", no_citations)

    await app.analyze("how many?")

    printed = out(app)
    assert "how many?" in printed
    assert "computed answer" in printed


async def test_rebuild_set_embedder_reports_settings_updated(app, client):
    async def one_document(mode):
        yield "doc-1"

    client.rebuild_database = one_document

    await app.rebuild(mode=RebuildMode.SET_EMBEDDER)

    assert "Stored embedder settings updated" in out(app)


async def test_rebuild_reports_an_empty_database(app, client):
    client.list_documents.return_value = []

    await app.rebuild(mode=RebuildMode.FULL)

    assert "No documents found" in out(app)


@pytest.mark.parametrize(
    "mode, description",
    [
        (RebuildMode.FULL, "full rebuild"),
        (RebuildMode.RECHUNK, "rechunk"),
        (RebuildMode.EMBED_ONLY, "embed only"),
        (RebuildMode.TITLE_ONLY, "title only"),
        (RebuildMode.DESCRIPTIONS, "picture descriptions"),
    ],
)
async def test_rebuild_names_the_mode_and_completes(app, client, mode, description):
    client.list_documents.return_value = [_doc()]

    async def one_document(mode):
        yield "doc-1"

    client.rebuild_database = one_document

    await app.rebuild(mode=mode)

    printed = out(app)
    assert description in printed
    assert "rebuild completed successfully" in printed


async def test_vacuum_confirms(app, client):
    await app.vacuum()

    client.vacuum.assert_awaited_once()
    assert "Vacuum completed successfully" in out(app)


async def test_create_index_refuses_a_small_table(app, client):
    client.store.chunks_table.count_rows = AsyncMock(return_value=10)

    await app.create_index()

    assert "Need at least 256 chunks" in out(app)


async def test_create_index_creates_one(app, client):
    client.store.chunks_table.count_rows = AsyncMock(return_value=512)
    client.store.chunks_table.list_indices = AsyncMock(return_value=[])

    await app.create_index()

    client.store._ensure_vector_index.assert_awaited_once()
    assert "Vector index created successfully" in out(app)


async def test_create_index_rebuilds_an_existing_one(app, client):
    client.store.chunks_table.count_rows = AsyncMock(return_value=512)
    client.store.chunks_table.list_indices = AsyncMock(return_value=["vector_idx"])

    await app.create_index()

    assert "Rebuilding existing vector index" in out(app)


def test_show_settings_hides_secrets(tmp_path):
    config = AppConfig(
        lancedb=LanceDBConfig(databases={"x": "db://x"}, api_key="secret-value")
    )
    app = HaikuRAGApp(scope=DatabaseScope.resolve(config), config=config)
    app.console = Console(record=True, width=200)

    app.show_settings()

    printed = app.console.export_text()
    assert "haiku.rag configuration" in printed
    assert "secret-value" not in printed


def test_show_settings_renders_the_shape_a_config_file_has(tmp_path):
    """Nesting is indented and a path is its string: what is read here is what
    `haiku.rag.yaml` holds."""
    import yaml

    config = AppConfig(
        lancedb=LanceDBConfig(databases={"alpha": "/tmp/a.lancedb"}),
        storage=StorageConfig(data_dir=tmp_path),
    )
    app = HaikuRAGApp(scope=DatabaseScope.resolve(config), config=config)
    app.console = Console(record=True, width=200)

    app.show_settings()

    printed = app.console.export_text()
    assert "PosixPath" not in printed
    # A dict repr parses as YAML flow style; the brace check tells the shapes
    # apart.
    assert "{'" not in printed
    assert "\n    alpha: /tmp/a.lancedb" in printed
    body = printed.split("haiku.rag configuration", 1)[1]
    parsed = yaml.safe_load(body)
    assert parsed["lancedb"]["databases"] == {"alpha": "/tmp/a.lancedb"}
    assert parsed["storage"]["data_dir"] == str(tmp_path)


def test_show_settings_survives_a_narrow_console_and_bracketed_values(tmp_path):
    """Values in `[...]` render verbatim and a long path stays on one
    parseable line, however narrow the console."""
    import yaml

    long_dir = tmp_path / "Application Support" / "haiku.rag" / "collections" / "one"
    config = AppConfig(
        storage=StorageConfig(data_dir=long_dir),
        prompts=PromptsConfig(domain_preamble="[INST] keep this [/INST]"),
    )
    app = HaikuRAGApp(scope=for_path(long_dir / "db", config), config=config)
    app.console = Console(record=True, width=60)

    app.show_settings()

    body = app.console.export_text().split("haiku.rag configuration", 1)[1]
    parsed = yaml.safe_load(body)
    assert parsed["prompts"]["domain_preamble"] == "[INST] keep this [/INST]"
    assert parsed["storage"]["data_dir"] == str(long_dir)


def test_remote_uri_is_the_display_path(tmp_path):
    config = AppConfig(lancedb=LanceDBConfig(databases={"path": "s3://bucket/path"}))
    app = HaikuRAGApp(scope=for_path(None, config), config=config)

    assert app.display_path == "s3://bucket/path"
    assert app._is_local is False


# --- paths that do not open a client -----------------------------------------


@pytest.fixture
def store_stub(monkeypatch):
    """Stub the Store for the paths that use it directly (history, migrate, tags)."""
    store = AsyncMock()

    class StubStore:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return store

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr("haiku.rag.store.engine.Store", StubStore)
    return store


async def test_init_reports_an_existing_database(app):
    await app.init()

    assert "Database already exists" in out(app)


async def test_info_reports_a_missing_path(tmp_path):
    application = HaikuRAGApp(scope=for_path(tmp_path / "gone"), config=AppConfig())
    application.console = Console(record=True, width=200)

    await application.info()

    assert "Database path does not exist" in application.console.export_text(
        clear=False
    )


async def test_history_reports_a_missing_path(tmp_path):
    application = HaikuRAGApp(scope=for_path(tmp_path / "gone"), config=AppConfig())
    application.console = Console(record=True, width=200)

    await application.history()

    assert "Database path does not exist" in application.console.export_text(
        clear=False
    )


async def test_history_rejects_an_unknown_table(app, store_stub):
    await app.history(table="nope")

    assert "Unknown table: nope" in out(app)


async def test_migrate_returns_what_the_store_applied(app, store_stub):
    store_stub.migrate.return_value = ["v0_40_0"]

    assert await app.migrate() == ["v0_40_0"]


async def test_list_tags_reports_none(app, monkeypatch):
    store = AsyncMock()
    store.list_tags.return_value = {}
    monkeypatch.setattr(app, "_tag_read_store", lambda: _as_cm(store))

    await app.list_tags()

    assert "No tags" in out(app)


async def test_list_tags_flags_a_partial_tag(app, monkeypatch):
    from haiku.rag.store.engine import TagInfo

    store = AsyncMock()
    store.list_tags.return_value = {
        "release-1": TagInfo(
            tables={"documents": 3},
            missing_tables=["chunks"],
        )
    }
    monkeypatch.setattr(app, "_tag_read_store", lambda: _as_cm(store))

    await app.list_tags()

    printed = out(app)
    assert "release-1" in printed
    assert "partial" in printed
    assert "chunks" in printed


async def test_create_and_delete_tag_confirm(app, monkeypatch):
    store = AsyncMock()
    monkeypatch.setattr(app, "_tag_write_store", lambda: _as_cm(store))

    await app.create_tag("release-1")
    await app.delete_tag("release-1")

    printed = out(app)
    assert "Created tag 'release-1'" in printed
    assert "Deleted tag 'release-1'" in printed


async def test_restore_tag_reports_the_safety_tag(app, monkeypatch):
    store = AsyncMock()
    store.restore_tag.return_value = "before-restore-20260820T000000Z"
    monkeypatch.setattr(app, "_tag_write_store", lambda: _as_cm(store))

    await app.restore_tag("release-1")

    assert "Restored database to tag 'release-1'" in out(app)


@pytest.mark.parametrize(
    "method, args",
    [
        ("create_tag", ("release-1",)),
        ("list_tags", ()),
        ("delete_tag", ("release-1",)),
        ("restore_tag", ("release-1",)),
    ],
)
async def test_tag_operations_require_the_database(tmp_path, method, args):
    application = HaikuRAGApp(scope=for_path(tmp_path / "gone"), config=AppConfig())

    with pytest.raises(ValueError, match="does not exist"):
        await getattr(application, method)(*args)


async def test_visualize_prints_a_page_per_image(app, client, monkeypatch):
    chunk = Chunk(content="c", order=0)
    chunk.document_uri = "test://doc"
    client.get_chunk_by_id.return_value = chunk
    client.visualize_chunk.return_value = [object(), object()]
    monkeypatch.setattr("textual_image.renderable.Image", lambda img: "<page image>")

    await app.visualize_chunk("c1")

    printed = out(app)
    assert "Visual grounding for chunk c1" in printed
    assert "Page 1/2" in printed
    assert "Page 2/2" in printed


def test_search_result_rendering_includes_provenance(app):
    result = SearchResult(
        content="hit",
        score=0.5,
        chunk_id="c1",
        document_id="doc-1",
        document_uri="test://doc",
        document_title="The Title",
        page_numbers=[2, 3],
        headings=["Chapter", "Section"],
    )

    app._rich_print_search_result(result)

    printed = out(app)
    assert "The Title" in printed
    assert "2, 3" in printed
    assert "Chapter > Section" in printed


async def test_run_mcp_stdio(app, client, monkeypatch):
    server = AsyncMock()
    monkeypatch.setattr("haiku.rag.app._mcp_server_covering", lambda *a, **kw: server)

    await app.run_mcp(transport="stdio")

    server.run_stdio_async.assert_awaited_once()


async def test_run_mcp_http(app, client, monkeypatch):
    server = AsyncMock()
    monkeypatch.setattr("haiku.rag.app._mcp_server_covering", lambda *a, **kw: server)

    await app.run_mcp(host="0.0.0.0", port=9001)

    server.run_http_async.assert_awaited_once_with(
        transport="streamable-http", host="0.0.0.0", port=9001
    )


async def test_run_mcp_survives_interruption(app, client, monkeypatch):
    server = AsyncMock()
    server.run_stdio_async.side_effect = KeyboardInterrupt
    monkeypatch.setattr("haiku.rag.app._mcp_server_covering", lambda *a, **kw: server)

    await app.run_mcp(transport="stdio")


def _as_cm(store):
    class _CM:
        async def __aenter__(self):
            return store

        async def __aexit__(self, *exc):
            return False

    return _CM()


async def test_doctor_renders_the_report(app, monkeypatch):
    from haiku.rag.doctor import CheckResult, Severity

    class Report:
        results = [
            CheckResult(name="tables", severity=Severity.OK, message="tables present"),
            CheckResult(
                name="orphans",
                severity=Severity.WARN,
                message="2 orphaned chunks",
                details=["chunk #1", "chunk #2"],
                remediation="run haiku-rag rebuild",
            ),
            CheckResult(
                name="provider:ollama",
                severity=Severity.FAIL,
                message="ollama unreachable",
            ),
        ]
        failed = True

        def count(self, severity):
            return sum(1 for r in self.results if r.severity is severity)

    async def report(*args, **kwargs):
        kwargs["on_progress"]("checking tables")
        return Report()

    monkeypatch.setattr("haiku.rag.doctor.run_doctor", report)

    failed = await app.doctor()

    assert failed is True
    printed = out(app)
    assert "tables present" in printed
    assert "2 orphaned chunks" in printed
    assert "chunk #1" in printed
    assert "run haiku-rag rebuild" in printed
    # providers are reported under their own rule
    assert "ollama unreachable" in printed
    assert "1 ok" in printed and "1 warning(s)" in printed and "1 failure(s)" in printed


async def test_doctor_reports_the_duplicates_export(app, monkeypatch, tmp_path):
    class Report:
        results = []
        failed = False

        def count(self, severity):
            return 0

    async def report(*args, **kwargs):
        return Report()

    monkeypatch.setattr("haiku.rag.doctor.run_doctor", report)
    target = tmp_path / "dupes.yaml"

    assert await app.doctor(duplicates_out=target) is False
    assert f"written to {target}" in out(app)


async def test_doctor_reports_a_missing_database(tmp_path):
    application = HaikuRAGApp(scope=for_path(tmp_path / "gone"), config=AppConfig())
    application.console = Console(record=True, width=200)

    assert await application.doctor() is True
    assert "does not exist" in application.console.export_text(clear=False)


async def test_download_models_reports_each_stage(app, monkeypatch):
    from haiku.rag.client.downloads import DownloadProgress

    events = [
        DownloadProgress(status="start", model="docling"),
        DownloadProgress(status="done", model="docling"),
        DownloadProgress(status="pulling", model="qwen3-embedding"),
        DownloadProgress(
            status="downloading",
            model="qwen3-embedding",
            digest="sha256:abcdef0123456789",
            total=100,
            completed=50,
        ),
        DownloadProgress(
            status="verifying sha256 digest",
            model="qwen3-embedding",
        ),
        DownloadProgress(status="done", model="qwen3-embedding"),
    ]

    async def stream(config):
        for event in events:
            yield event

    monkeypatch.setattr("haiku.rag.client.downloads.download_models", stream)

    await app.download_models()

    printed = out(app)
    assert "Downloading docling" in printed
    assert "Pulling qwen3-embedding" in printed
    assert "qwen3-embedding" in printed


async def test_document_content_is_truncated_in_lists(app, client):
    long_doc = _doc("l1\nl2\nl3\nl4\nl5")
    client.list_documents.return_value = [long_doc]

    await app.list_documents()

    printed = out(app)
    assert "l1" in printed
    assert "l5" not in printed


def _database_info(vector_index, chunk_rows=300):
    """A complete DatabaseInfo: info() indexes every required table by name."""
    from haiku.rag.store.info import (
        DatabaseInfo,
        EmbeddingsInfo,
        TableInfo,
        VectorIndexInfo,
    )

    rows = {"chunks": chunk_rows}
    return DatabaseInfo(
        path="/tmp/db.lancedb",
        exists=True,
        stored_version="0.75.0",
        embeddings=EmbeddingsInfo(provider="ollama", name="m", vector_dim=3),
        tables=[
            TableInfo(
                name=name,
                exists=True,
                num_rows=rows.get(name, 1),
                num_versions=2 if name == "documents" else 3,
            )
            for name in (
                "documents",
                "document_meta",
                "chunks",
                "document_items",
                "settings",
            )
        ],
        vector_index=VectorIndexInfo(**vector_index),
        # info() indexes these by name when printing the version block.
        packages={
            "haiku_rag": "0.75.0",
            "lancedb": "0.26.0",
            "docling": "2.102.2",
            "pydantic_ai": "2.18.0",
            "docling_document_schema": "1.7.0",
        },
    )


async def test_info_reports_unindexed_chunks_and_document_meta_versions(
    app, monkeypatch
):
    """The index-status branches read from gather_database_info, so drive them
    through a stubbed report rather than building databases."""
    info = _database_info({"exists": True, "indexed_rows": 200, "unindexed_rows": 100})

    async def stub_info(config, db_path):
        return info

    # info() imports it inside the method, so patch where it is looked up.
    monkeypatch.setattr("haiku.rag.store.info.gather_database_info", stub_info)

    await app.info()

    printed = out(app)
    assert "unindexed chunks" in printed
    assert "100" in printed
    assert "versions (document_meta)" in printed


async def test_info_suggests_creating_an_index_when_there_are_enough_chunks(
    app, monkeypatch
):
    info = _database_info({"exists": False})

    async def stub_info(config, db_path):
        return info

    # info() imports it inside the method, so patch where it is looked up.
    monkeypatch.setattr("haiku.rag.store.info.gather_database_info", stub_info)

    await app.info()

    assert "haiku-rag create-index" in out(app)


async def test_doctor_updates_a_terminal_status(app, monkeypatch):
    """On a terminal the checks report progress through console.status."""

    class Report:
        results = []
        failed = False

        def count(self, severity):
            return 0

    labels: list[str] = []

    async def report(*args, **kwargs):
        kwargs["on_progress"]("scanning vectors")
        return Report()

    class Status:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def update(self, label):
            labels.append(label)

    monkeypatch.setattr("haiku.rag.doctor.run_doctor", report)
    monkeypatch.setattr(type(app.console), "is_terminal", property(lambda self: True))
    monkeypatch.setattr(app.console, "status", lambda label: Status())

    await app.doctor()

    assert labels == ["scanning vectors..."]


async def test_history_limits_the_versions_shown(app, monkeypatch):
    store = AsyncMock()
    store.list_tags.return_value = {}
    store.list_table_versions.return_value = [
        {"version": v, "timestamp": None} for v in (1, 2, 3)
    ]

    class StubStore:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return store

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr("haiku.rag.store.engine.Store", StubStore)

    await app.history(table="documents", limit=1)

    printed = out(app)
    assert "v3" in printed
    assert "v1" not in printed


async def test_analyze_prints_citation_renderables(app, client, monkeypatch):
    result = AsyncMock()
    result.answer = "computed"
    result.citations = ["c1"]
    client.analyze.return_value = result

    async def one_citation(citations, client=None, full=False):
        return ["citation renderable"]

    monkeypatch.setattr("haiku.rag.app.format_citations_rich", one_citation)

    await app.analyze("how many?")

    assert "citation renderable" in out(app)
