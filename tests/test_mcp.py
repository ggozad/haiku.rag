import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.mcp import create_mcp_server
from haiku.rag.store.models import Chunk, Document, SearchResult
from haiku.rag.tools.document import DocumentInfo


@pytest.fixture(autouse=True)
def mock_embedder(monkeypatch):
    """Monkeypatch the embedder to return deterministic vectors."""
    import random

    from haiku.rag.embeddings import EmbedderWrapper

    async def fake_embed_query(self, text):
        random.seed(hash(text) % (2**32))
        return [random.random() for _ in range(2560)]

    async def fake_embed_documents(self, texts):
        result = []
        for t in texts:
            random.seed(hash(t) % (2**32))
            result.append([random.random() for _ in range(2560)])
        return result

    monkeypatch.setattr(EmbedderWrapper, "embed_query", fake_embed_query)
    monkeypatch.setattr(EmbedderWrapper, "embed_documents", fake_embed_documents)


@pytest.fixture
async def mcp_db(temp_db_path):
    """Create a test database with sample documents."""
    async with HaikuRAG(temp_db_path, create=True) as rag:
        await rag.create_document(
            "Artificial intelligence is transforming industries worldwide.",
            title="AI Overview",
            uri="test://ai-overview",
        )
        await rag.create_document(
            "Machine learning is a subset of artificial intelligence.",
            title="ML Basics",
            uri="test://ml-basics",
        )
    return temp_db_path


async def _get_tool(mcp, name):
    """Get a tool function from an MCP server by name."""
    tool = await mcp.get_tool(name)
    return tool.fn


class TestMCPReadTools:
    @pytest.mark.asyncio
    async def test_search_documents(self, mcp_db):
        mcp = create_mcp_server(mcp_db, read_only=True)
        search = await _get_tool(mcp, "search_documents")

        results = await search(query="artificial intelligence")
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_documents_with_limit(self, mcp_db):
        mcp = create_mcp_server(mcp_db, read_only=True)
        search = await _get_tool(mcp, "search_documents")

        results = await search(query="artificial intelligence", limit=1)
        assert len(results) == 1

    @pytest.mark.asyncio
    @pytest.mark.filterwarnings("ignore:Found propagated trace context:RuntimeWarning")
    async def test_search_documents_preserves_chunk_meta_through_serialization(
        self, mcp_db
    ):
        """Chunk_meta must survive FastMCP's actual wire serialization.

        Calling the tool function directly bypasses that serialization step entirely."""
        from fastmcp import Client

        async with HaikuRAG(mcp_db, create=True) as rag:
            doc = await rag.get_document_by_uri("test://ai-overview")
            embedding = (await rag.embedder.embed_documents(["x"]))[0]
            await rag.chunk_repository.create(
                Chunk(
                    document_id=doc.id,
                    content="Artificial intelligence is transforming industries worldwide.",
                    metadata={"fake-metadata-for-testing": "42"},
                    embedding=embedding,
                )
            )
            await rag.chunk_repository._ensure_fts_index()

        mcp = create_mcp_server(mcp_db, read_only=True)
        async with Client(mcp) as client:
            result = await client.call_tool(
                "search_documents", {"query": "artificial intelligence"}
            )

        results = result.structured_content["result"]
        assert results
        assert any(
            r["chunk_meta"] == {"fake-metadata-for-testing": "42"} for r in results
        )

    @pytest.mark.asyncio
    async def test_get_document(self, mcp_db):
        mcp = create_mcp_server(mcp_db, read_only=True)
        get_doc = await _get_tool(mcp, "get_document")

        # First get the ID via list
        list_docs = await _get_tool(mcp, "list_documents")
        docs = await list_docs()
        doc_id = docs[0].id

        result = await get_doc(document_id=doc_id)
        assert isinstance(result, Document)
        assert result.content != ""
        assert result.title is not None

    @pytest.mark.asyncio
    async def test_get_document_excludes_docling_fields(self, mcp_db):
        mcp = create_mcp_server(mcp_db, read_only=True)
        get_doc = await _get_tool(mcp, "get_document")

        list_docs = await _get_tool(mcp, "list_documents")
        docs = await list_docs()
        doc_id = docs[0].id

        result = await get_doc(document_id=doc_id)
        serialized = result.model_dump(mode="json")
        assert "docling_document" not in serialized
        assert "docling_version" not in serialized

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, mcp_db):
        mcp = create_mcp_server(mcp_db, read_only=True)
        get_doc = await _get_tool(mcp, "get_document")

        result = await get_doc(document_id="nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_documents(self, mcp_db):
        mcp = create_mcp_server(mcp_db, read_only=True)
        list_docs = await _get_tool(mcp, "list_documents")

        results = await list_docs()
        assert len(results) == 2
        assert all(isinstance(r, DocumentInfo) for r in results)

    @pytest.mark.asyncio
    async def test_list_documents_with_limit(self, mcp_db):
        mcp = create_mcp_server(mcp_db, read_only=True)
        list_docs = await _get_tool(mcp, "list_documents")

        results = await list_docs(limit=1)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_documents_with_filter(self, mcp_db):
        mcp = create_mcp_server(mcp_db, read_only=True)
        list_docs = await _get_tool(mcp, "list_documents")

        results = await list_docs(filter="title = 'AI Overview'")
        assert len(results) == 1
        assert results[0].title == "AI Overview"


class TestMCPWriteTools:
    @pytest.mark.asyncio
    async def test_write_tools_registered_when_not_read_only(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True):
            pass
        mcp = create_mcp_server(temp_db_path, read_only=False)
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "add_document_from_text" in tool_names
        assert "add_document_from_file" in tool_names
        assert "add_document_from_url" in tool_names
        assert "delete_document" in tool_names

    @pytest.mark.asyncio
    async def test_write_tools_not_registered_when_read_only(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True):
            pass
        mcp = create_mcp_server(temp_db_path, read_only=True)
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "add_document_from_text" not in tool_names
        assert "delete_document" not in tool_names

    @pytest.mark.asyncio
    async def test_add_document_from_text(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True):
            pass
        mcp = create_mcp_server(temp_db_path, read_only=False)
        add_text = await _get_tool(mcp, "add_document_from_text")

        doc_id = await add_text(content="Test content for MCP", title="MCP Test Doc")
        assert doc_id is not None

        get_doc = await _get_tool(mcp, "get_document")
        doc = await get_doc(document_id=doc_id)
        assert doc.title == "MCP Test Doc"
        assert doc.content == "Test content for MCP"

    @pytest.mark.asyncio
    async def test_delete_document(self, mcp_db):
        mcp = create_mcp_server(mcp_db, read_only=False)
        list_docs = await _get_tool(mcp, "list_documents")
        delete_doc = await _get_tool(mcp, "delete_document")

        docs = await list_docs()
        assert len(docs) == 2

        result = await delete_doc(document_id=docs[0].id)
        assert result is True

        docs_after = await list_docs()
        assert len(docs_after) == 1

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, mcp_db):
        mcp = create_mcp_server(mcp_db, read_only=False)
        delete_doc = await _get_tool(mcp, "delete_document")

        result = await delete_doc(document_id="nonexistent-id")
        assert result is False


class TestMCPImageQuery:
    """search_documents_by_image is registered only when the embedder is multimodal."""

    @pytest.mark.asyncio
    async def test_image_query_tool_absent_for_text_only_embedder(self, mcp_db):
        """Default text-only embedder must not expose the image-query tool."""
        mcp = create_mcp_server(mcp_db, read_only=True)
        names = {t.name for t in await mcp.list_tools()}
        assert "search_documents_by_image" not in names

    @pytest.mark.asyncio
    async def test_image_query_tool_registered_for_multimodal_embedder(
        self, mcp_db, monkeypatch
    ):
        """When the embedder reports supports_images=True, the tool exists
        and routes a base64 image through ``client.search``."""
        from haiku.rag.embeddings import EmbedderWrapper

        class StubMultimodal(EmbedderWrapper):
            supports_images = True

            def __init__(self):
                super().__init__(embedder=None, vector_dim=2560)

            async def embed_image(self, image):
                # Produce a deterministic-ish vector of the right dim.
                return [0.0] * 2560

        monkeypatch.setattr(
            "haiku.rag.embeddings.get_embedder",
            lambda *a, **kw: StubMultimodal(),
        )

        mcp = create_mcp_server(mcp_db, read_only=True)
        names = {t.name for t in await mcp.list_tools()}
        assert "search_documents_by_image" in names

        search_by_image = await _get_tool(mcp, "search_documents_by_image")
        # Standalone PNG header (won't decode to a real image but our stub doesn't care).
        import base64

        png_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")
        results = await search_by_image(image_base64=png_b64)
        # Empty list is fine (the stub vector won't match the toy fixture).
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_image_query_returns_empty_on_invalid_base64(
        self, mcp_db, monkeypatch
    ):
        """Garbage base64 from the caller is swallowed, returning an empty
        list rather than crashing the MCP server."""
        from haiku.rag.embeddings import EmbedderWrapper

        class StubMultimodal(EmbedderWrapper):
            supports_images = True

            def __init__(self):
                super().__init__(embedder=None, vector_dim=2560)

        monkeypatch.setattr(
            "haiku.rag.embeddings.get_embedder",
            lambda *a, **kw: StubMultimodal(),
        )

        mcp = create_mcp_server(mcp_db, read_only=True)
        search_by_image = await _get_tool(mcp, "search_documents_by_image")

        # Not valid base64 (contains non-base64 chars) — the strict decoder
        # in search_documents_by_image rejects it.
        results = await search_by_image(image_base64="!!! not base64 !!!")
        assert results == []


class TestMCPImageInput:
    @pytest.mark.asyncio
    async def test_ask_question_decodes_images(self, mcp_db, monkeypatch):
        from base64 import b64encode

        captured = {}

        async def fake_ask(self, question, filter=None, images=None):
            captured["images"] = images
            return ("answer", [])

        monkeypatch.setattr(HaikuRAG, "ask", fake_ask)
        mcp = create_mcp_server(mcp_db, read_only=True)
        ask = await _get_tool(mcp, "ask_question")

        png = b"fake image bytes"
        result = await ask(question="q", images_base64=[b64encode(png).decode()])
        assert result == "answer"
        assert captured["images"] == [png]

    @pytest.mark.asyncio
    async def test_analyze_decodes_images(self, mcp_db, monkeypatch):
        from base64 import b64encode
        from types import SimpleNamespace

        captured = {}

        async def fake_analyze(self, question, filter=None, images=None):
            captured["images"] = images
            return SimpleNamespace(answer="answer")

        monkeypatch.setattr(HaikuRAG, "analyze", fake_analyze)
        mcp = create_mcp_server(mcp_db, read_only=True)
        analyze = await _get_tool(mcp, "analyze")

        jpeg = b"fake jpeg bytes"
        result = await analyze(question="q", images_base64=[b64encode(jpeg).decode()])
        assert result == "answer"
        assert captured["images"] == [jpeg]

    @pytest.mark.asyncio
    async def test_ask_question_rejects_invalid_base64(self, mcp_db):
        mcp = create_mcp_server(mcp_db, read_only=True)
        ask = await _get_tool(mcp, "ask_question")

        result = await ask(question="q", images_base64=["!!! not base64 !!!"])
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_ask_question_without_images_passes_none(self, mcp_db, monkeypatch):
        captured = {}

        async def fake_ask(self, question, filter=None, images=None):
            captured["images"] = images
            return ("answer", [])

        monkeypatch.setattr(HaikuRAG, "ask", fake_ask)
        mcp = create_mcp_server(mcp_db, read_only=True)
        ask = await _get_tool(mcp, "ask_question")

        result = await ask(question="q")
        assert result == "answer"
        assert captured["images"] is None


class TestMCPFileAndUrlIngestion:
    @pytest.mark.asyncio
    async def test_add_document_from_file(self, temp_db_path, tmp_path):
        async with HaikuRAG(temp_db_path, create=True):
            pass
        source = tmp_path / "note.txt"
        source.write_text("Ingested from a file path.")

        mcp = create_mcp_server(temp_db_path, read_only=False)
        add_file = await _get_tool(mcp, "add_document_from_file")

        doc_id = await add_file(file_path=str(source), title="File Doc")
        assert doc_id is not None

        get_doc = await _get_tool(mcp, "get_document")
        doc = await get_doc(document_id=doc_id)
        assert doc.title == "File Doc"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_name,kwargs",
        [
            ("add_document_from_file", {"file_path": "/tmp/x.txt"}),
            ("add_document_from_url", {"url": "https://example.com/x.txt"}),
        ],
    )
    @pytest.mark.parametrize(
        "results,expected",
        [
            (
                [Document(id="first", content="a"), Document(id="second", content="b")],
                "first",
            ),
            ([], None),
        ],
        ids=["directory_reports_first_id", "empty_directory_reports_none"],
    )
    async def test_add_tools_handle_multi_document_sources(
        self, mcp_db, monkeypatch, tool_name, kwargs, results, expected
    ):
        """A source resolving to several documents reports the first id."""

        async def fake_from_source(self, source, title=None, metadata=None, **kw):
            return results

        monkeypatch.setattr(HaikuRAG, "create_document_from_source", fake_from_source)
        mcp = create_mcp_server(mcp_db, read_only=False)
        add = await _get_tool(mcp, tool_name)

        assert await add(**kwargs) == expected

    @pytest.mark.asyncio
    async def test_add_document_from_url(self, mcp_db, monkeypatch):
        async def fake_from_source(self, source, title=None, metadata=None, **kwargs):
            assert source == "https://example.com/doc.txt"
            return Document(id="url-doc", content="fetched")

        monkeypatch.setattr(HaikuRAG, "create_document_from_source", fake_from_source)
        mcp = create_mcp_server(mcp_db, read_only=False)
        add_url = await _get_tool(mcp, "add_document_from_url")

        assert await add_url(url="https://example.com/doc.txt") == "url-doc"


class TestMCPToolsDegradeOnError:
    """Every tool swallows client failures and returns its empty value rather
    than propagating an exception to the MCP transport."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "client_method,tool_name,kwargs,expected",
        [
            (
                "create_document_from_source",
                "add_document_from_file",
                {"file_path": "/tmp/x.txt"},
                None,
            ),
            (
                "create_document_from_source",
                "add_document_from_url",
                {"url": "https://example.com/x"},
                None,
            ),
            ("create_document", "add_document_from_text", {"content": "x"}, None),
            ("delete_document", "delete_document", {"document_id": "x"}, False),
            ("search", "search_documents", {"query": "x"}, []),
            ("get_document_by_id", "get_document", {"document_id": "x"}, None),
            ("list_documents", "list_documents", {}, []),
        ],
    )
    async def test_tool_returns_empty_value_when_client_raises(
        self, mcp_db, monkeypatch, client_method, tool_name, kwargs, expected
    ):
        async def boom(self, *args, **kw):
            raise RuntimeError("client exploded")

        monkeypatch.setattr(HaikuRAG, client_method, boom)
        mcp = create_mcp_server(mcp_db, read_only=False)
        tool = await _get_tool(mcp, tool_name)

        assert await tool(**kwargs) == expected

    @pytest.mark.asyncio
    async def test_list_documents_returns_empty_for_invalid_filter(self, mcp_db):
        mcp = create_mcp_server(mcp_db, read_only=True)
        list_docs = await _get_tool(mcp, "list_documents")

        assert await list_docs(filter="no_such_column = 1") == []

    @pytest.mark.asyncio
    async def test_analyze_reports_the_error(self, mcp_db, monkeypatch):
        async def boom(self, question, filter=None, images=None):
            raise RuntimeError("sandbox exploded")

        monkeypatch.setattr(HaikuRAG, "analyze", boom)
        mcp = create_mcp_server(mcp_db, read_only=True)
        analyze = await _get_tool(mcp, "analyze")

        assert "sandbox exploded" in await analyze(question="q")

    @pytest.mark.asyncio
    async def test_ask_question_appends_citations_when_requested(
        self, mcp_db, monkeypatch
    ):
        from haiku.rag.store.models.citation import Citation

        citation = Citation(
            chunk_id="c1",
            document_id="d1",
            content="cited text",
            document_uri="test://ai-overview",
            document_title="AI Overview",
        )

        async def fake_ask(self, question, filter=None, images=None):
            return ("the answer", [citation])

        monkeypatch.setattr(HaikuRAG, "ask", fake_ask)
        mcp = create_mcp_server(mcp_db, read_only=True)
        ask = await _get_tool(mcp, "ask_question")

        with_cite = await ask(question="q", cite=True)
        assert with_cite.startswith("the answer")
        assert "AI Overview" in with_cite

        assert await ask(question="q", cite=False) == "the answer"


class TestMCPClientLifetime:
    @pytest.mark.asyncio
    async def test_tool_calls_share_one_database_open(self, mcp_db, monkeypatch):
        from haiku.rag.store.engine import Store

        opens = 0
        initialize = Store._initialize

        async def counted(self):
            nonlocal opens
            opens += 1
            return await initialize(self)

        monkeypatch.setattr(Store, "_initialize", counted)

        mcp = create_mcp_server(mcp_db, read_only=True)
        search = await _get_tool(mcp, "search_documents")
        list_docs = await _get_tool(mcp, "list_documents")
        await search(query="artificial intelligence")
        await list_docs()
        await search(query="machine learning")

        assert opens == 1

    @pytest.mark.asyncio
    async def test_concurrent_reads_share_one_open(self, mcp_db, monkeypatch):
        import asyncio

        from haiku.rag.store.engine import Store

        opens = 0
        initialize = Store._initialize

        async def counted(self):
            nonlocal opens
            opens += 1
            return await initialize(self)

        monkeypatch.setattr(Store, "_initialize", counted)

        mcp = create_mcp_server(mcp_db, read_only=True)
        list_docs = await _get_tool(mcp, "list_documents")

        results = await asyncio.gather(*(list_docs() for _ in range(5)))

        assert opens == 1
        assert all(len(r) == 2 for r in results)

    @pytest.mark.asyncio
    async def test_a_write_is_visible_to_the_next_read(self, mcp_db):
        """One connection sees its own writes, whatever the consistency interval."""
        mcp = create_mcp_server(mcp_db, read_only=False)
        list_docs = await _get_tool(mcp, "list_documents")
        delete_doc = await _get_tool(mcp, "delete_document")

        docs = await list_docs()
        assert await delete_doc(document_id=docs[0].id) is True

        assert len(await list_docs()) == len(docs) - 1

    @pytest.mark.asyncio
    async def test_lifespan_opens_and_closes_once(self, mcp_db, monkeypatch):
        from haiku.rag.store.engine import Store

        opens = 0
        initialize = Store._initialize

        async def counted(self):
            nonlocal opens
            opens += 1
            return await initialize(self)

        monkeypatch.setattr(Store, "_initialize", counted)

        mcp = create_mcp_server(mcp_db, read_only=True)
        # _lifespan_manager is what every transport enters; the public
        # lifespan() combines provider lifespans only.
        async with mcp._lifespan_manager():
            assert opens == 1, "startup should open the database, not the first call"
            search = await _get_tool(mcp, "search_documents")
            await search(query="artificial intelligence")
            assert opens == 1

        assert opens == 1

    @pytest.mark.asyncio
    async def test_a_uri_backed_database_is_not_replaced_by_a_local_path(
        self, monkeypatch
    ):
        """A path overrides `lancedb.uri`, so the server gets the database's own
        path — None where a URI placed it — not the local stand-in a URI-backed
        ref resolves to for display."""
        from haiku.rag.app import HaikuRAGApp
        from haiku.rag.client.scope import DatabaseScope
        from haiku.rag.config.models import AppConfig, LanceDBConfig

        config = AppConfig(
            lancedb=LanceDBConfig(databases={"prod": "s3://bucket/prod.lancedb"})
        )
        seen: dict = {}

        class _Server:
            async def run_stdio_async(self):
                return None

        def fake_create(db_path=None, config=None, read_only=False):
            seen["db_path"] = db_path
            seen["config"] = config
            return _Server()

        monkeypatch.setattr("haiku.rag.app.create_mcp_server", fake_create)
        app = HaikuRAGApp(
            scope=DatabaseScope.resolve(config, database_name="prod"), config=config
        )

        await app.run_mcp(transport="stdio")

        assert seen["db_path"] is None
        [ref] = DatabaseScope.resolve(
            seen["config"], database_path=seen["db_path"]
        ).databases
        assert ref.uri == "s3://bucket/prod.lancedb"

    @pytest.mark.asyncio
    async def test_startup_fails_when_the_database_cannot_open(self, tmp_path):
        mcp = create_mcp_server(tmp_path / "does-not-exist.lancedb", read_only=True)

        with pytest.raises(FileNotFoundError):
            async with mcp._lifespan_manager():
                pass

    @pytest.mark.asyncio
    async def test_a_second_lifespan_cycle_opens_a_fresh_client(
        self, mcp_db, monkeypatch
    ):
        from haiku.rag.store.engine import Store

        opens = 0
        initialize = Store._initialize

        async def counted(self):
            nonlocal opens
            opens += 1
            return await initialize(self)

        monkeypatch.setattr(Store, "_initialize", counted)

        mcp = create_mcp_server(mcp_db, read_only=True)
        search = await _get_tool(mcp, "search_documents")

        async with mcp._lifespan_manager():
            await search(query="artificial intelligence")
        assert opens == 1

        async with mcp._lifespan_manager():
            results = await search(query="artificial intelligence")
        assert opens == 2
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_same_dim_drift_starts_read_only_but_not_writable(self, mcp_db):
        """Validation is unchanged: same-dimension identity drift warns in
        read-only mode and raises in writable mode. The MCP server no longer
        opts out of it for deletion."""
        from haiku.rag.config import get_config
        from haiku.rag.store.exceptions import ConfigMismatchError

        drifted = get_config().model_copy(deep=True)
        drifted.embeddings.model.name = "a-different-model"

        async with create_mcp_server(
            mcp_db, config=drifted, read_only=True
        )._lifespan_manager():
            pass

        with pytest.raises(ConfigMismatchError):
            async with create_mcp_server(
                mcp_db, config=drifted, read_only=False
            )._lifespan_manager():
                pass
