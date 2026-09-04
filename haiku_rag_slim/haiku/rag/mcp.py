import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.types import ContentBlock, ImageContent, TextContent, ToolAnnotations
from pydantic import Field

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, get_config
from haiku.rag.context import build_toc
from haiku.rag.store.exceptions import UnknownDatabaseError
from haiku.rag.store.models import Document, SearchResult
from haiku.rag.store.models.document_item import DocumentItem
from haiku.rag.store.schema import DocumentMetaRecord
from haiku.rag.tools.document import DocumentInfo, DocumentSection, OutlineNode
from haiku.rag.tools.search import collect_pictures
from haiku.rag.utils import format_citations

if TYPE_CHECKING:
    from typing import Any

    from haiku.rag.client.scope import DatabaseScope

logger = logging.getLogger(__name__)

_FILTER_COLUMNS = ", ".join(DocumentMetaRecord.model_fields)

Filter = Annotated[
    str | None,
    Field(
        description=(
            f"SQL WHERE clause over the document columns {_FILTER_COLUMNS}, "
            "restricting which documents are used. `metadata` is a JSON string, "
            'so match its keys with LIKE: metadata LIKE \'%"author": "Smith"%\'. '
            "Also uri LIKE '%.pdf', title = 'Q3 report'."
        )
    ),
]
Sources = Annotated[
    list[str] | None,
    Field(description="Collections to use, by name. All of them by default."),
]


def _read_only(title: str) -> ToolAnnotations:
    return ToolAnnotations(title=title, read_only_hint=True, open_world_hint=False)


def _decode_image(image_base64: str) -> bytes:
    import base64

    try:
        return base64.b64decode(image_base64, validate=True)
    except ValueError as e:
        # binascii.Error for characters outside the alphabet or bad padding,
        # ValueError itself for non-ASCII input.
        raise ToolError("Invalid base64 image") from e


def _decode_images(images_base64: list[str] | None) -> list[bytes] | None:
    if not images_base64:
        return None
    return [_decode_image(b64) for b64 in images_base64]


async def _check_filter(
    rag: HaikuRAG, filter: str | None, sources: list[str] | None = None
) -> None:
    """Evaluate a filter on its own before the read that would use it.

    A filtered count on one selected database runs the same predicate on the
    same table and nothing else, so a ValueError here is the query engine
    rejecting the filter; its message names columns and the statement, never
    a location. A ValueError raised later in the read stays masked. Only the
    selection is touched: every database shares the schema, so one suffices.
    """
    if filter is None:
        return
    selected = await rag.clients_covering(sources)
    if not selected:
        return
    try:
        await selected[0].count_documents(filter=filter)
    except ValueError as e:
        raise ToolError(f"Invalid filter {filter!r}: {e}") from e


def _instructions(scope: "DatabaseScope", config: AppConfig, agents: bool) -> str:
    """What the server is for, naming no tools: the client has every tool's
    description from the listing."""
    lines = [
        "haiku-rag is the user's knowledge base: documents they ingested, "
        "searchable by meaning and keyword, readable whole or section by section."
    ]
    if agents:
        lines.append(
            "Questions can be answered from them with citations, or computed "
            "across them with code."
        )
    lines.append(
        "Use it whenever a question could be answered from those documents, "
        "before answering from memory, and say when it had nothing relevant."
    )
    if scope.covers_multiple:
        lines.append(
            f"It holds several collections: {', '.join(scope.names)}. Results "
            "and citations name theirs in `source`; pass `sources` to use a subset."
        )
    if config.prompts.domain_preamble:
        lines.append(config.prompts.domain_preamble)
    return "\n".join(lines)


def _search_result(results: list[SearchResult], covers_multiple: bool) -> ToolResult:
    """Results as the in-process agents read them, plus the matched chunk's
    metadata, then each distinct picture as an image block labelled with its
    result, and the results as structured content without the picture bytes."""
    import base64

    total = len(results)
    text = "\n\n".join(
        result.format_for_agent(
            rank=rank,
            total=total,
            include_collection=covers_multiple,
            include_document_id=True,
            include_chunk_meta=True,
        )
        for rank, result in enumerate(results, 1)
    )
    content: list[ContentBlock] = [
        TextContent(type="text", text=text or "No results found.")
    ]
    pictures, _ = collect_pictures(results)
    for source, chunk_id, self_ref, picture in pictures:
        collection = f" in {source}" if covers_multiple and source else ""
        content.append(
            TextContent(
                type="text",
                text=f"Picture {self_ref} of search result [{chunk_id}]{collection}",
            )
        )
        content.append(
            ImageContent(
                type="image",
                data=base64.b64encode(picture.data).decode("ascii"),
                mime_type="image/png",
            )
        )
    return ToolResult(
        content=content,
        structured_content={
            "result": [
                result.model_dump(mode="json", exclude={"image_data"})
                for result in results
            ]
        },
    )


def _node(toc: "dict[str, Any]") -> OutlineNode:
    return OutlineNode(
        id=toc["self_ref"],
        title=toc["title"],
        level=toc["level"],
        page_numbers=toc["page_numbers"],
        children=[_node(child) for child in toc["children"]],
    )


def _find(toc: list["dict[str, Any]"], section_id: str) -> "dict[str, Any] | None":
    for node in toc:
        if node["self_ref"] == section_id:
            return node
        found = _find(node["children"], section_id)
        if found is not None:
            return found
    return None


def create_mcp_server(
    db_path: Path | None = None,
    config: AppConfig | None = None,
    agents: bool = True,
) -> FastMCP:
    """Create an MCP server over the databases the configuration places.

    Args:
        db_path: Path to the database file, where `config` places none; or
            None to serve the databases the configuration places. Beside
            `lancedb.databases` a path raises `AmbiguousDatabaseError`.
        config: Configuration to use.
        agents: Register `ask_question` and `analyze`, which run a model on
            the server.
    """
    from haiku.rag.client.scope import DatabaseScope

    config = config if config is not None else get_config()
    return _covering(
        DatabaseScope.resolve(config, database_path=db_path), config, agents
    )


def _covering(
    scope: "DatabaseScope", config: AppConfig, agents: bool = True
) -> FastMCP:
    """An MCP server over databases someone already resolved.

    Internal, as ``HaikuRAG._covering`` is: the public factory takes a path and
    resolves it, which is its own job. A caller that resolved already passes the
    scope, so the configured name survives, which results and citations carry as
    ``source``.
    """
    client: HaikuRAG | None = None
    stack = AsyncExitStack()
    client_lock = asyncio.Lock()

    async def _client() -> HaikuRAG:
        """The server's client, opened once.

        Opening cost is per connection, and on object storage the first vector
        query loads the index into the session cache, so a client per tool call
        pays that repeatedly.
        """
        nonlocal client
        async with client_lock:
            if client is None:
                client = await stack.enter_async_context(
                    HaikuRAG._covering(scope, config, read_only=True)
                )
        return client

    @asynccontextmanager
    async def lifespan(_server: FastMCP) -> AsyncIterator[None]:
        # Opened eagerly: an unopenable database fails at startup.
        nonlocal client
        await _client()
        try:
            yield
        finally:
            # The lifespan can be re-entered; without the reset the next cycle
            # hands out the closed client, including when aclose itself fails.
            try:
                await stack.aclose()
            finally:
                client = None

    # Masking keeps paths and provider URLs out of an unexpected error's text;
    # the traceback goes to the server log. A ToolError reaches the client as is.
    mcp = FastMCP(
        "haiku-rag",
        instructions=_instructions(scope, config, agents),
        version=metadata.version("haiku.rag-slim"),
        lifespan=lifespan,
        mask_error_details=True,
    )

    @mcp.tool(annotations=_read_only("Search documents"))
    async def search_documents(
        query: str,
        limit: int | None = None,
        include_images: bool = True,
        filter: Filter = None,
        sources: Sources = None,
    ) -> ToolResult:
        """Search the knowledge base by meaning and keyword.

        Use this first for any question the documents might answer; it needs
        no model and is the cheapest call. Results come best first, each with
        its rank, `Document ID`, `Collection` when the server covers several,
        the document title, section headings, the matched chunk's metadata
        when it has any, and the matching passage expanded to its section;
        pass the id and collection to the document tools. Pictures in the
        results follow as images, each labelled with its result. Ranks, not scores,
        are the signal: scores are not comparable across queries. If nothing
        relevant comes back, rephrase once or narrow with `filter` before
        concluding the material is absent.

        Args:
            query: What to look for, in natural language or keywords.
            limit: How many results to return; the server's configured default
                when omitted.
            include_images: Return the pictures in the results as images.
                False for a smaller response.
        """
        rag = await _client()
        try:
            await _check_filter(rag, filter, sources)
            results = await rag.search(
                query,
                limit=limit,
                filter=filter,
                include_images=include_images,
                sources=sources,
            )
        except UnknownDatabaseError as e:
            raise ToolError(str(e)) from e
        return _search_result(await rag.expand_context(results), rag.covers_multiple)

    # Image-as-query tool, only registered when the configured embedder
    # supports image embeddings. Probed at server-build time when no Store is
    # open, so there is no cached embedder to read; this is the one place
    # outside Store that builds one.
    from haiku.rag.embeddings import get_embedder

    if get_embedder(config).supports_images:

        @mcp.tool(annotations=_read_only("Search documents by image"))
        async def search_documents_by_image(
            image_base64: str,
            limit: int | None = None,
            include_images: bool = True,
            filter: Filter = None,
            sources: Sources = None,
        ) -> ToolResult:
            """Search the knowledge base with an image as the query.

            Use this when the question is about a picture rather than words.
            The image is embedded and matched against document text and
            figures by vector similarity alone. Results have the shape of
            `search_documents` results.

            Args:
                image_base64: The query image, PNG or JPEG bytes as base64.
                limit: How many results to return; the server's configured
                    default when omitted.
                include_images: Return the pictures in the results as images.
                    False for a smaller response.
            """
            raw = _decode_image(image_base64)
            rag = await _client()
            try:
                await _check_filter(rag, filter, sources)
                results = await rag.search(
                    raw,
                    limit=limit,
                    filter=filter,
                    include_images=include_images,
                    sources=sources,
                )
            except UnknownDatabaseError as e:
                raise ToolError(str(e)) from e
            return _search_result(
                await rag.expand_context(results), rag.covers_multiple
            )

    @mcp.tool(annotations=_read_only("Get document"))
    async def get_document(document_id: str, source: str | None = None) -> Document:
        """Read one document whole, in reading order.

        Use this after a search when a passage is not enough. Returns the
        document's content, title, uri and metadata. Ids come from search
        results and `list_documents`.

        Args:
            document_id: The document's id.
            source: The collection holding it. Without one every collection
                is asked.
        """
        rag = await _client()
        try:
            document = await rag.get_document_by_id(document_id, source)
        except UnknownDatabaseError as e:
            raise ToolError(str(e)) from e
        if document is None:
            raise ToolError(f"No document with id {document_id!r}")
        return document

    async def _items_of(document_id: str, source: str | None) -> list[DocumentItem]:
        """A document's items in reading order, from the database holding it."""
        rag = await _client()
        try:
            document = await rag.get_document_by_id(document_id, source)
            if document is None:
                raise ToolError(f"No document with id {document_id!r}")
            owner = await rag.reader_for(source or document.source)
        except UnknownDatabaseError as e:
            raise ToolError(str(e)) from e
        assert owner is not None, "a stored document names its database"
        return await owner.document_item_repository.get_all_items(document_id)

    @mcp.tool(annotations=_read_only("Document outline"))
    async def get_document_outline(
        document_id: str, source: str | None = None
    ) -> list[OutlineNode]:
        """The heading tree of a document, with page numbers.

        Use this on a long document to see its structure before reading, then
        pass a node's `id` to `get_document_section`. Returns the headings
        nested by level; an empty list means the document has no headings,
        so read it with `get_document`.

        Args:
            document_id: The document's id.
            source: The collection holding it. Without one every collection
                is asked.
        """
        return [
            _node(toc) for toc in build_toc(await _items_of(document_id, source), {})
        ]

    @mcp.tool(annotations=_read_only("Document section"))
    async def get_document_section(
        document_id: str, section_id: str, source: str | None = None
    ) -> DocumentSection:
        """The text of one section of a document, subsections included.

        Use this to read a part of a long document instead of the whole.
        `section_id` is a node `id` from `get_document_outline`. Returns the
        section's heading, page numbers and text in reading order, up to the
        next heading of the same or a higher level.

        Args:
            document_id: The document's id.
            section_id: The `id` of a node in the document's outline.
            source: The collection holding it. Without one every collection
                is asked.
        """
        items = await _items_of(document_id, source)
        node = _find(build_toc(items, {}), section_id)
        if node is None:
            raise ToolError(f"No section {section_id!r} in document {document_id!r}")
        start, end = node["item_range"]
        return DocumentSection(
            id=node["self_ref"],
            title=node["title"],
            page_numbers=node["page_numbers"],
            content="\n\n".join(
                item.text
                for item in items
                if start <= item.position < end and item.text
            ),
        )

    @mcp.tool(annotations=_read_only("List documents"))
    async def list_documents(
        limit: int | None = None,
        offset: int | None = None,
        filter: Filter = None,
    ) -> list[DocumentInfo]:
        """List what the knowledge base holds.

        Use this to see which documents exist, their titles, URIs and
        metadata, and so what a `filter` can match. Not a search: it returns
        no passages.

        Args:
            limit: How many documents to return.
            offset: How many documents to skip, for paging.
        """
        rag = await _client()
        await _check_filter(rag, filter)
        documents = await rag.list_documents(limit, offset, filter)
        return [
            DocumentInfo(
                id=doc.id,
                title=doc.title or "Untitled",
                uri=doc.uri or "",
                created=doc.created_at.strftime("%Y-%m-%d"),
                source=doc.source,
                metadata=doc.metadata,
            )
            for doc in documents
        ]

    if agents:

        @mcp.tool(annotations=_read_only("Ask a question"))
        async def ask_question(
            question: str,
            images_base64: list[str] | None = None,
            sources: Sources = None,
        ) -> str:
            """Answer a question from the documents with a retrieval agent.

            Use this when the user wants an answer rather than material to read.
            It runs a model on the server and is slower than a search. Returns
            the answer, followed by citations to the passages it rests on.

            Args:
                question: The question, in natural language.
                images_base64: Images to attach to the question, PNG or JPEG
                    bytes as base64. Needs a vision-capable model on the server.
            """
            images = _decode_images(images_base64)
            rag = await _client()
            try:
                answer, citations = await rag.ask(
                    question, images=images, sources=sources
                )
            except UnknownDatabaseError as e:
                raise ToolError(str(e)) from e
            except Exception as e:
                logger.exception("ask_question failed")
                raise ToolError(f"ask_question failed: {type(e).__name__}") from e
            if citations:
                answer += "\n\n" + format_citations(
                    citations, include_source=rag.covers_multiple
                )
            return answer

        @mcp.tool(annotations=_read_only("Analyze documents"))
        async def analyze(
            question: str,
            filter: Filter = None,
            images_base64: list[str] | None = None,
            sources: Sources = None,
        ) -> str:
            """Compute an answer across documents with code.

            Use this for counting, aggregation, comparison across many documents
            or arithmetic over tables, where reading passages is not enough. A
            model writes and runs Python in a sandbox over the selected documents.
            It is the slowest tool. Returns the answer as text.

            Args:
                question: The question, in natural language.
                images_base64: Images to attach to the question, PNG or JPEG
                    bytes as base64. Needs a vision-capable model on the server.
            """
            images = _decode_images(images_base64)
            rag = await _client()
            try:
                result = await rag.analyze(
                    question, filter=filter, images=images, sources=sources
                )
            except UnknownDatabaseError as e:
                raise ToolError(str(e)) from e
            except Exception as e:
                logger.exception("analyze failed")
                raise ToolError(f"analyze failed: {type(e).__name__}") from e
            return result.answer

    return mcp
