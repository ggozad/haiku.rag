import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, get_config
from haiku.rag.store.exceptions import UnknownDatabaseError
from haiku.rag.store.models import Document, SearchResult
from haiku.rag.store.schema import DocumentMetaRecord
from haiku.rag.tools.document import DocumentInfo
from haiku.rag.utils import format_citations

if TYPE_CHECKING:
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
    return ToolAnnotations(title=title, readOnlyHint=True, openWorldHint=False)


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


def _instructions(scope: "DatabaseScope", config: AppConfig) -> str:
    """What the server is for, naming no tools: the client has every tool's
    description from the listing."""
    lines = [
        "haiku-rag is the user's knowledge base: documents they ingested, "
        "searchable by meaning and keyword, readable whole, answered with "
        "citations, or computed across documents.",
        "Use it whenever a question could be answered from those documents, "
        "before answering from memory, and say when it had nothing relevant.",
    ]
    if scope.covers_multiple:
        lines.append(
            f"It holds several collections: {', '.join(scope.names)}. Results "
            "and citations name theirs in `source`; pass `sources` to use a subset."
        )
    if config.prompts.domain_preamble:
        lines.append(config.prompts.domain_preamble)
    return "\n".join(lines)


def create_mcp_server(
    db_path: Path | None = None,
    config: AppConfig | None = None,
) -> FastMCP:
    """Create an MCP server over the databases the configuration places.

    Args:
        db_path: Path to the database file, where `config` places none; or
            None to serve the databases the configuration places. Beside
            `lancedb.databases` a path raises `AmbiguousDatabaseError`.
        config: Configuration to use.
    """
    from haiku.rag.client.scope import DatabaseScope

    config = config if config is not None else get_config()
    return _covering(DatabaseScope.resolve(config, database_path=db_path), config)


def _covering(scope: "DatabaseScope", config: AppConfig) -> FastMCP:
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
        instructions=_instructions(scope, config),
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
    ) -> list[SearchResult]:
        """Search the knowledge base by meaning and keyword.

        Use this first for any question the documents might answer; it needs
        no model and is the cheapest call. Results come best first, each with
        the document's id, title and collection, the section headings and the
        matching passage. Scores are not comparable across queries, so read
        the order, not the numbers. If nothing relevant comes back, rephrase
        once or narrow with `filter` before concluding the material is absent.

        Args:
            query: What to look for, in natural language or keywords.
            limit: How many results to return; the server's configured default
                when omitted.
            include_images: Attach the bytes of pictures in the results as
                base64 PNG under `image_data`. False for a smaller response.
        """
        rag = await _client()
        try:
            await _check_filter(rag, filter, sources)
            return await rag.search(
                query,
                limit=limit,
                filter=filter,
                include_images=include_images,
                sources=sources,
            )
        except UnknownDatabaseError as e:
            raise ToolError(str(e)) from e

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
        ) -> list[SearchResult]:
            """Search the knowledge base with an image as the query.

            Use this when the question is about a picture rather than words.
            The image is embedded and matched against document text and
            figures by vector similarity alone. Results have the shape of
            `search_documents` results.

            Args:
                image_base64: The query image, PNG or JPEG bytes as base64.
                limit: How many results to return; the server's configured
                    default when omitted.
                include_images: Attach the bytes of pictures in the results as
                    base64 PNG under `image_data`. False for a smaller response.
            """
            raw = _decode_image(image_base64)
            rag = await _client()
            try:
                await _check_filter(rag, filter, sources)
                return await rag.search(
                    raw,
                    limit=limit,
                    filter=filter,
                    include_images=include_images,
                    sources=sources,
                )
            except UnknownDatabaseError as e:
                raise ToolError(str(e)) from e

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

    @mcp.tool(annotations=_read_only("Ask a question"))
    async def ask_question(
        question: str,
        cite: bool = False,
        images_base64: list[str] | None = None,
        sources: Sources = None,
    ) -> str:
        """Answer a question from the documents with a retrieval agent.

        Use this when the user wants an answer rather than material to read.
        It runs a model on the server and is slower than a search. Returns
        the answer, followed by citations to the passages it rests on when
        `cite` is set.

        Args:
            question: The question, in natural language.
            cite: Append citations to the answer.
            images_base64: Images to attach to the question, PNG or JPEG
                bytes as base64. Needs a vision-capable model on the server.
        """
        images = _decode_images(images_base64)
        rag = await _client()
        try:
            answer, citations = await rag.ask(question, images=images, sources=sources)
        except UnknownDatabaseError as e:
            raise ToolError(str(e)) from e
        except Exception as e:
            logger.exception("ask_question failed")
            raise ToolError(f"ask_question failed: {type(e).__name__}") from e
        if cite and citations:
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
