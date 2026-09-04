from types import SimpleNamespace

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.mcp import _covering as _mcp_covering
from haiku.rag.mcp import create_mcp_server
from haiku.rag.store.models import Chunk, Document, SearchResult
from haiku.rag.tools.document import DocumentInfo
from tests.multi_db.helpers import _config, _seed


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
def multimodal_embedder(monkeypatch):
    """An embedder reporting image support, so the image-query tool registers."""
    from haiku.rag.embeddings import EmbedderWrapper

    class StubMultimodal(EmbedderWrapper):
        supports_images = True

        def __init__(self):
            super().__init__(embedder=None, vector_dim=2560)

    monkeypatch.setattr(
        "haiku.rag.embeddings.get_embedder", lambda *a, **kw: StubMultimodal()
    )


@pytest.fixture
async def mcp_db(temp_db_path):
    """Create a test database with sample documents."""
    async with HaikuRAG(temp_db_path, create=True) as rag:
        await rag.create_document(
            "Artificial intelligence is transforming industries worldwide.",
            title="AI Overview",
            uri="test://ai-overview",
            metadata={"author": "Ada"},
        )
        await rag.create_document(
            "Machine learning is a subset of artificial intelligence.",
            title="ML Basics",
            uri="test://ml-basics",
        )
    return temp_db_path


@pytest.fixture
async def two_dbs(tmp_path):
    """Two configured databases, alpha and beta, one document each."""
    config = _config(tmp_path, ["alpha", "beta"])
    await _seed(config, "alpha", ["alpha document about cats"])
    await _seed(config, "beta", ["beta document about cats"])
    return config


def _covering_all(config):
    from haiku.rag.client.scope import DatabaseScope

    return _mcp_covering(DatabaseScope.resolve(config), config)


async def _get_tool(mcp, name):
    """Get a tool function from an MCP server by name."""
    tool = await mcp.get_tool(name)
    return tool.fn


class TestMCPReadTools:
    @pytest.mark.asyncio
    async def test_search_documents(self, mcp_db):
        mcp = create_mcp_server(mcp_db)
        search = await _get_tool(mcp, "search_documents")

        results = await search(query="artificial intelligence")
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_documents_with_limit(self, mcp_db):
        mcp = create_mcp_server(mcp_db)
        search = await _get_tool(mcp, "search_documents")

        results = await search(query="artificial intelligence", limit=1)
        assert len(results) == 1

    @pytest.mark.asyncio
    @pytest.mark.filterwarnings("ignore:Found propagated trace context:RuntimeWarning")
    async def test_search_documents_with_filter(self, mcp_db):
        from fastmcp import Client

        async with Client(create_mcp_server(mcp_db)) as client:
            result = await client.call_tool(
                "search_documents",
                {"query": "artificial intelligence", "filter": "title = 'ML Basics'"},
            )

        results = result.structured_content["result"]
        assert results
        assert {r["document_title"] for r in results} == {"ML Basics"}

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
            await rag.store.chunks_table.optimize()

        mcp = create_mcp_server(mcp_db)
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
        mcp = create_mcp_server(mcp_db)
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
        mcp = create_mcp_server(mcp_db)
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
        mcp = create_mcp_server(mcp_db)
        get_doc = await _get_tool(mcp, "get_document")

        result = await get_doc(document_id="nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_documents(self, mcp_db):
        mcp = create_mcp_server(mcp_db)
        list_docs = await _get_tool(mcp, "list_documents")

        results = await list_docs()
        assert len(results) == 2
        assert all(isinstance(r, DocumentInfo) for r in results)

    @pytest.mark.asyncio
    async def test_list_documents_with_limit(self, mcp_db):
        mcp = create_mcp_server(mcp_db)
        list_docs = await _get_tool(mcp, "list_documents")

        results = await list_docs(limit=1)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_documents_with_filter(self, mcp_db):
        mcp = create_mcp_server(mcp_db)
        list_docs = await _get_tool(mcp, "list_documents")

        results = await list_docs(filter="title = 'AI Overview'")
        assert len(results) == 1
        assert results[0].title == "AI Overview"

    @pytest.mark.asyncio
    @pytest.mark.filterwarnings("ignore:Found propagated trace context:RuntimeWarning")
    async def test_list_documents_carries_metadata(self, mcp_db):
        from fastmcp import Client

        async with Client(create_mcp_server(mcp_db)) as client:
            result = await client.call_tool("list_documents", {})

        [overview] = [
            d
            for d in result.structured_content["result"]
            if d["title"] == "AI Overview"
        ]
        assert overview["metadata"] == {"author": "Ada"}


@pytest.mark.filterwarnings("ignore:Found propagated trace context:RuntimeWarning")
class TestMCPDescribesItself:
    """What a client learns from initialize and list_tools, over the wire."""

    @pytest.mark.asyncio
    async def test_instructions_and_version_are_set(self, mcp_db):
        from importlib import metadata

        from fastmcp import Client

        async with Client(create_mcp_server(mcp_db)) as client:
            init = client.initialize_result

        assert init.instructions
        assert init.serverInfo.version == metadata.version("haiku.rag-slim")

    @pytest.mark.asyncio
    async def test_instructions_name_the_collections_when_covering_several(
        self, two_dbs
    ):
        from fastmcp import Client

        from haiku.rag.client.scope import DatabaseScope

        async with Client(_covering_all(two_dbs)) as client:
            covering_both = client.initialize_result.instructions
        one = DatabaseScope.resolve(two_dbs, database_name="alpha")
        async with Client(_mcp_covering(one, two_dbs)) as client:
            covering_one = client.initialize_result.instructions

        assert "alpha" in covering_both
        assert "beta" in covering_both
        assert "beta" not in covering_one

    @pytest.mark.asyncio
    async def test_instructions_carry_the_domain_preamble(self, mcp_db):
        from fastmcp import Client

        from haiku.rag.config import get_config

        config = get_config().model_copy(deep=True)
        config.prompts.domain_preamble = "Everything here is about zebras."

        async with Client(create_mcp_server(mcp_db, config=config)) as client:
            with_preamble = client.initialize_result.instructions
        async with Client(create_mcp_server(mcp_db)) as client:
            without = client.initialize_result.instructions

        assert "Everything here is about zebras." in with_preamble
        assert "zebras" not in without

    @pytest.mark.asyncio
    async def test_every_tool_is_annotated_read_only(self, mcp_db, multimodal_embedder):
        from fastmcp import Client

        async with Client(create_mcp_server(mcp_db)) as client:
            tools = await client.list_tools()

        assert len(tools) == 6
        for tool in tools:
            assert tool.annotations is not None, tool.name
            assert tool.annotations.readOnlyHint is True, tool.name
            assert tool.annotations.openWorldHint is False, tool.name
            assert tool.annotations.title, tool.name

    @pytest.mark.asyncio
    async def test_every_parameter_is_described(self, mcp_db, multimodal_embedder):
        from fastmcp import Client

        async with Client(create_mcp_server(mcp_db)) as client:
            tools = await client.list_tools()

        undescribed = [
            f"{tool.name}.{name}"
            for tool in tools
            for name, schema in tool.inputSchema.get("properties", {}).items()
            if not schema.get("description")
        ]
        assert len(tools) == 6
        assert undescribed == []


class TestMCPToolSet:
    @pytest.mark.asyncio
    async def test_the_server_registers_read_tools_only(self, mcp_db):
        mcp = create_mcp_server(mcp_db)

        assert {t.name for t in await mcp.list_tools()} == {
            "search_documents",
            "get_document",
            "list_documents",
            "ask_question",
            "analyze",
        }


class TestMCPCoversTheConfiguredSet:
    @pytest.mark.asyncio
    async def test_results_name_the_database_they_came_from(self, two_dbs):
        mcp = _covering_all(two_dbs)
        search = await _get_tool(mcp, "search_documents")

        results = await search(query="cats")

        assert {r.source for r in results} == {"alpha", "beta"}

    @pytest.mark.asyncio
    async def test_sources_narrows_the_search(self, two_dbs):
        mcp = _covering_all(two_dbs)
        search = await _get_tool(mcp, "search_documents")

        results = await search(query="cats", sources=["beta"])

        assert results
        assert {r.source for r in results} == {"beta"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_name,kwargs",
        [
            ("search_documents", {"query": "cats", "sources": ["nope"]}),
            (
                "search_documents_by_image",
                {"image_base64": "AAAA", "sources": ["nope"]},
            ),
            ("get_document", {"document_id": "x", "source": "nope"}),
            ("ask_question", {"question": "q", "sources": ["nope"]}),
            ("analyze", {"question": "q", "sources": ["nope"]}),
        ],
    )
    async def test_an_unknown_database_is_an_error_not_an_empty_result(
        self, two_dbs, multimodal_embedder, tool_name, kwargs
    ):
        from fastmcp.exceptions import ToolError

        mcp = _covering_all(two_dbs)
        tool = await _get_tool(mcp, tool_name)

        with pytest.raises(ToolError, match="nope"):
            await tool(**kwargs)

    @pytest.mark.asyncio
    async def test_the_listing_covers_every_database(self, two_dbs):
        mcp = _covering_all(two_dbs)
        list_docs = await _get_tool(mcp, "list_documents")

        documents = await list_docs()

        assert {d.source for d in documents} == {"alpha", "beta"}

    @pytest.mark.asyncio
    async def test_get_document_reaches_whichever_database_holds_it(self, two_dbs):
        mcp = _covering_all(two_dbs)
        list_docs = await _get_tool(mcp, "list_documents")
        get_doc = await _get_tool(mcp, "get_document")
        [beta] = [d for d in await list_docs() if d.source == "beta"]

        found = await get_doc(document_id=beta.id)
        named = await get_doc(document_id=beta.id, source="beta")

        assert found.id == named.id == beta.id
        assert found.source == named.source == "beta"

    @pytest.mark.asyncio
    async def test_the_public_factory_covers_a_configured_set(self, two_dbs):
        mcp = create_mcp_server(config=two_dbs)
        search = await _get_tool(mcp, "search_documents")

        results = await search(query="cats")

        assert {r.source for r in results} == {"alpha", "beta"}

    @pytest.mark.asyncio
    async def test_ask_question_names_each_citations_database(
        self, two_dbs, monkeypatch
    ):
        from haiku.rag.store.models.citation import Citation

        def cited(source):
            return Citation(
                chunk_id="c1",
                document_id="d1",
                content="cited text",
                document_uri="test://cats",
                document_title="Cats",
                source=source,
            )

        async def fake_ask(self, question, filter=None, images=None, sources=None):
            return ("the answer", [cited("alpha"), cited("beta")])

        monkeypatch.setattr(HaikuRAG, "ask", fake_ask)
        mcp = _covering_all(two_dbs)
        ask = await _get_tool(mcp, "ask_question")

        answer = await ask(question="q", cite=True)

        assert "alpha" in answer
        assert "beta" in answer

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_name,client_method,returns",
        [
            ("ask_question", "ask", ("answer", [])),
            ("analyze", "analyze", SimpleNamespace(answer="answer")),
        ],
    )
    async def test_agents_search_the_selected_databases(
        self, two_dbs, monkeypatch, tool_name, client_method, returns
    ):
        seen = {}

        async def fake(self, question, filter=None, images=None, sources=None):
            seen["sources"] = sources
            return returns

        monkeypatch.setattr(HaikuRAG, client_method, fake)
        mcp = _covering_all(two_dbs)
        tool = await _get_tool(mcp, tool_name)

        await tool(question="q", sources=["beta"])

        assert seen["sources"] == ["beta"]


class TestMCPImageQuery:
    """search_documents_by_image is registered only when the embedder is multimodal."""

    @pytest.mark.asyncio
    async def test_image_query_tool_absent_for_text_only_embedder(self, mcp_db):
        """Default text-only embedder must not expose the image-query tool."""
        mcp = create_mcp_server(mcp_db)
        names = {t.name for t in await mcp.list_tools()}
        assert "search_documents_by_image" not in names

    @pytest.mark.asyncio
    async def test_image_query_tool_registered_for_multimodal_embedder(
        self, mcp_db, multimodal_embedder, monkeypatch
    ):
        """When the embedder reports supports_images=True, the tool exists
        and routes the decoded image and the selection through ``client.search``."""
        seen = {}

        async def fake_search(self, query, **kwargs):
            seen.update(query=query, **kwargs)
            return []

        monkeypatch.setattr(HaikuRAG, "search", fake_search)

        mcp = create_mcp_server(mcp_db)
        names = {t.name for t in await mcp.list_tools()}
        assert "search_documents_by_image" in names

        search_by_image = await _get_tool(mcp, "search_documents_by_image")
        import base64

        png = b"\x89PNG\r\n\x1a\n"
        results = await search_by_image(
            image_base64=base64.b64encode(png).decode("ascii"),
            filter="uri LIKE 'x%'",
            sources=["alpha"],
        )

        assert results == []
        assert seen["query"] == png
        assert seen["filter"] == "uri LIKE 'x%'"
        assert seen["sources"] == ["alpha"]

    @pytest.mark.asyncio
    async def test_image_query_rejects_characters_outside_the_alphabet(
        self, mcp_db, multimodal_embedder, monkeypatch
    ):
        """A lenient decoder would drop the stray characters and search."""
        searched = False

        async def fake_search(self, query, **kwargs):
            nonlocal searched
            searched = True
            return []

        monkeypatch.setattr(HaikuRAG, "search", fake_search)
        mcp = create_mcp_server(mcp_db)
        search_by_image = await _get_tool(mcp, "search_documents_by_image")

        assert await search_by_image(image_base64="AAAA!!!!") == []
        assert not searched

    @pytest.mark.asyncio
    async def test_image_query_returns_empty_on_invalid_base64(
        self, mcp_db, multimodal_embedder
    ):
        """Garbage base64 from the caller is swallowed, returning an empty
        list rather than crashing the MCP server."""
        mcp = create_mcp_server(mcp_db)
        search_by_image = await _get_tool(mcp, "search_documents_by_image")

        # Not valid base64 (contains non-base64 chars) — the strict decoder
        # in search_documents_by_image rejects it.
        results = await search_by_image(image_base64="!!! not base64 !!!")
        assert results == []

    @pytest.mark.asyncio
    async def test_image_query_returns_empty_when_the_search_raises(
        self, mcp_db, multimodal_embedder, monkeypatch
    ):
        async def boom(self, *args, **kw):
            raise RuntimeError("client exploded")

        monkeypatch.setattr(HaikuRAG, "search", boom)
        mcp = create_mcp_server(mcp_db)
        search_by_image = await _get_tool(mcp, "search_documents_by_image")

        assert await search_by_image(image_base64="AAAA") == []


class TestMCPImageInput:
    @pytest.mark.asyncio
    async def test_ask_question_decodes_images(self, mcp_db, monkeypatch):
        from base64 import b64encode

        captured = {}

        async def fake_ask(self, question, filter=None, images=None, sources=None):
            captured["images"] = images
            return ("answer", [])

        monkeypatch.setattr(HaikuRAG, "ask", fake_ask)
        mcp = create_mcp_server(mcp_db)
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

        async def fake_analyze(self, question, filter=None, images=None, sources=None):
            captured["images"] = images
            return SimpleNamespace(answer="answer")

        monkeypatch.setattr(HaikuRAG, "analyze", fake_analyze)
        mcp = create_mcp_server(mcp_db)
        analyze = await _get_tool(mcp, "analyze")

        jpeg = b"fake jpeg bytes"
        result = await analyze(question="q", images_base64=[b64encode(jpeg).decode()])
        assert result == "answer"
        assert captured["images"] == [jpeg]

    @pytest.mark.asyncio
    async def test_ask_question_rejects_invalid_base64(self, mcp_db):
        mcp = create_mcp_server(mcp_db)
        ask = await _get_tool(mcp, "ask_question")

        result = await ask(question="q", images_base64=["!!! not base64 !!!"])
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_ask_question_without_images_passes_none(self, mcp_db, monkeypatch):
        captured = {}

        async def fake_ask(self, question, filter=None, images=None, sources=None):
            captured["images"] = images
            return ("answer", [])

        monkeypatch.setattr(HaikuRAG, "ask", fake_ask)
        mcp = create_mcp_server(mcp_db)
        ask = await _get_tool(mcp, "ask_question")

        result = await ask(question="q")
        assert result == "answer"
        assert captured["images"] is None


class TestMCPToolsDegradeOnError:
    """Every tool swallows client failures and returns its empty value rather
    than propagating an exception to the MCP transport."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "client_method,tool_name,kwargs,expected",
        [
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
        mcp = create_mcp_server(mcp_db)
        tool = await _get_tool(mcp, tool_name)

        assert await tool(**kwargs) == expected

    @pytest.mark.asyncio
    async def test_list_documents_returns_empty_for_invalid_filter(self, mcp_db):
        mcp = create_mcp_server(mcp_db)
        list_docs = await _get_tool(mcp, "list_documents")

        assert await list_docs(filter="no_such_column = 1") == []

    @pytest.mark.asyncio
    async def test_analyze_reports_the_error(self, mcp_db, monkeypatch):
        async def boom(self, question, filter=None, images=None, sources=None):
            raise RuntimeError("sandbox exploded")

        monkeypatch.setattr(HaikuRAG, "analyze", boom)
        mcp = create_mcp_server(mcp_db)
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
            source="alpha",
        )

        async def fake_ask(self, question, filter=None, images=None, sources=None):
            return ("the answer", [citation])

        monkeypatch.setattr(HaikuRAG, "ask", fake_ask)
        mcp = create_mcp_server(mcp_db)
        ask = await _get_tool(mcp, "ask_question")

        with_cite = await ask(question="q", cite=True)
        assert with_cite.startswith("the answer")
        assert "AI Overview" in with_cite
        # One database: its name adds nothing.
        assert "alpha" not in with_cite

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

        mcp = create_mcp_server(mcp_db)
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

        mcp = create_mcp_server(mcp_db)
        list_docs = await _get_tool(mcp, "list_documents")

        results = await asyncio.gather(*(list_docs() for _ in range(5)))

        assert opens == 1
        assert all(len(r) == 2 for r in results)

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

        mcp = create_mcp_server(mcp_db)
        # _lifespan_manager is what every transport enters; the public
        # lifespan() combines provider lifespans only.
        async with mcp._lifespan_manager():
            assert opens == 1, "startup should open the database, not the first call"
            search = await _get_tool(mcp, "search_documents")
            await search(query="artificial intelligence")
            assert opens == 1

        assert opens == 1

    @pytest.mark.asyncio
    async def test_the_scope_decides_the_database_and_names_its_results(
        self, mcp_db, tmp_path
    ):
        """The scope is the selection: the server reads the one database it
        names, and results carry that name."""
        from haiku.rag.client.scope import DatabaseScope
        from haiku.rag.config.models import AppConfig, LanceDBConfig

        other = tmp_path / "beta.lancedb"
        async with HaikuRAG(other, create=True) as rag:
            await rag.create_document(
                "Zebras graze on the savannah.", title="Zebras", uri="test://zebras"
            )

        config = AppConfig(
            lancedb=LanceDBConfig(databases={"alpha": str(mcp_db), "beta": str(other)})
        )
        scope = DatabaseScope.resolve(config, database_name="alpha")

        mcp = _mcp_covering(scope, config)
        async with mcp._lifespan_manager():
            search = await _get_tool(mcp, "search_documents")
            results = await search(query="artificial intelligence")
            listing = await _get_tool(mcp, "list_documents")
            documents = await listing()

        assert results
        assert {r.source for r in results} == {"alpha"}
        titles = {d.title for d in documents}
        assert "AI Overview" in titles
        assert "Zebras" not in titles

    def test_the_public_factory_refuses_a_path_beside_a_configured_set(self, tmp_path):
        """A path and `lancedb.databases` both place the database."""
        from haiku.rag.config.models import AppConfig, LanceDBConfig
        from haiku.rag.store.exceptions import AmbiguousDatabaseError

        config = AppConfig(
            lancedb=LanceDBConfig(databases={"alpha": str(tmp_path / "a")})
        )

        with pytest.raises(AmbiguousDatabaseError, match="alpha"):
            create_mcp_server(tmp_path / "other.lancedb", config=config)

    @pytest.mark.asyncio
    async def test_the_command_hands_the_server_its_resolved_database(
        self, monkeypatch
    ):
        """`run_mcp` passes the resolved scope, not a path and not a derived
        configuration: the scope keeps both the URI and the name."""
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

        def fake_covering(scope, config):
            seen.update(scope=scope, config=config)
            return _Server()

        monkeypatch.setattr("haiku.rag.app._mcp_server_covering", fake_covering)
        app = HaikuRAGApp(
            scope=DatabaseScope.resolve(config, database_name="prod"), config=config
        )

        await app.run_mcp(transport="stdio")

        [ref] = seen["scope"].databases
        assert ref.name == "prod"
        assert ref.location == "s3://bucket/prod.lancedb"
        # The caller's configuration, not one derived from the ref.
        assert seen["config"].lancedb.databases == {"prod": "s3://bucket/prod.lancedb"}

    @pytest.mark.asyncio
    async def test_startup_fails_when_the_database_cannot_open(self, tmp_path):
        mcp = create_mcp_server(tmp_path / "does-not-exist.lancedb")

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

        mcp = create_mcp_server(mcp_db)
        search = await _get_tool(mcp, "search_documents")

        async with mcp._lifespan_manager():
            await search(query="artificial intelligence")
        assert opens == 1

        async with mcp._lifespan_manager():
            results = await search(query="artificial intelligence")
        assert opens == 2
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_same_dim_drift_starts(self, mcp_db):
        """Same-dimension identity drift warns on a read-only open and raises
        on a writable one; the server starts, so it opened read-only."""
        from haiku.rag.config import get_config

        drifted = get_config().model_copy(deep=True)
        drifted.embeddings.model.name = "a-different-model"

        async with create_mcp_server(mcp_db, config=drifted)._lifespan_manager():
            pass
