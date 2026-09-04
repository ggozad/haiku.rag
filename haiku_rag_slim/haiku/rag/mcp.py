import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastmcp import FastMCP

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, get_config
from haiku.rag.store.models import Document, SearchResult
from haiku.rag.tools.document import DocumentInfo
from haiku.rag.utils import format_citations

if TYPE_CHECKING:
    from haiku.rag.client.scope import DatabaseScope


def _decode_images(images_base64: list[str] | None) -> list[bytes] | None:
    if not images_base64:
        return None
    import base64

    return [base64.b64decode(b64, validate=True) for b64 in images_base64]


def create_mcp_server(
    db_path: Path | None = None,
    config: AppConfig | None = None,
) -> FastMCP:
    """Create an MCP server over one database.

    Args:
        db_path: Path to the database file, where `config` places none; or
            None to serve the database the configuration places. Beside
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
    from haiku.rag.store.exceptions import AmbiguousDatabaseError

    if scope.covers_multiple:
        raise AmbiguousDatabaseError(
            "an MCP server serves one database, and this scope covers "
            f"{', '.join(scope.names)}; name the one to serve"
        )
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

    mcp = FastMCP("haiku-rag", lifespan=lifespan)

    @mcp.tool()
    async def search_documents(
        query: str, limit: int | None = None, include_images: bool = True
    ) -> list[SearchResult]:
        """Search the RAG system for documents using hybrid search (vector similarity + full-text search).

        When include_images is True (default) and a picture-labeled chunk is
        in the result set, ``SearchResult.image_data`` carries base64-encoded
        PNG bytes keyed by self_ref. Set to False to omit the bytes from the
        response (smaller JSON payload for plain-text consumers).
        """
        try:
            rag = await _client()
            return await rag.search(query, limit=limit, include_images=include_images)
        except Exception:
            return []

    # Image-as-query tool, only registered when the configured embedder
    # supports image embeddings. Probed at server-build time when no Store is
    # open, so there is no cached embedder to read; this is the one place
    # outside Store that builds one.
    from haiku.rag.embeddings import get_embedder

    if get_embedder(config).supports_images:

        @mcp.tool()
        async def search_documents_by_image(
            image_base64: str,
            limit: int | None = None,
            include_images: bool = True,
        ) -> list[SearchResult]:
            """Search the RAG system using an image as the query.

            ``image_base64`` is a base64-encoded image (PNG/JPEG bytes). The
            image is embedded via the configured multimodal embedder and the
            chunks table is searched vector-only. ``include_images`` controls
            whether picture bytes are attached to picture-labeled results.
            """
            import base64

            try:
                raw = base64.b64decode(image_base64)
            except Exception:
                return []
            try:
                rag = await _client()
                return await rag.search(raw, limit=limit, include_images=include_images)
            except Exception:
                return []

    @mcp.tool()
    async def get_document(document_id: str) -> Document | None:
        """Get a document by its ID."""
        try:
            rag = await _client()
            return await rag.get_document_by_id(document_id)
        except Exception:
            return None

    @mcp.tool()
    async def list_documents(
        limit: int | None = None,
        offset: int | None = None,
        filter: str | None = None,
    ) -> list[DocumentInfo]:
        """List all documents with optional pagination and filtering.

        Args:
            limit: Maximum number of documents to return.
            offset: Number of documents to skip.
            filter: Optional SQL WHERE clause to filter documents.
        """
        try:
            rag = await _client()
            documents = await rag.list_documents(limit, offset, filter)

            return [
                DocumentInfo(
                    id=doc.id,
                    title=doc.title or "Untitled",
                    uri=doc.uri or "",
                    created=doc.created_at.strftime("%Y-%m-%d"),
                )
                for doc in documents
            ]
        except Exception:
            return []

    @mcp.tool()
    async def ask_question(
        question: str,
        cite: bool = False,
        images_base64: list[str] | None = None,
    ) -> str:
        """Ask a question using the QA agent.

        Args:
            question: The question to ask.
            cite: Whether to include citations in the response.
            images_base64: Base64-encoded images attached to the question
                (requires a vision-capable QA model).

        Returns:
            The answer as a string.
        """
        try:
            images = _decode_images(images_base64)
            rag = await _client()
            answer, citations = await rag.ask(question, images=images)
            if cite and citations:
                answer += "\n\n" + format_citations(citations)
            return answer
        except Exception as e:
            return f"Error answering question: {e!s}"

    @mcp.tool()
    async def analyze(
        question: str,
        filter: str | None = None,
        images_base64: list[str] | None = None,
    ) -> str:
        """Answer complex questions using the analysis capability.

        Use this for questions requiring computation, aggregation, or
        structural traversal across documents. The capability can write and
        execute Python code in a sandboxed interpreter.

        Args:
            question: The question to answer.
            filter: Optional SQL WHERE clause to filter documents.
            images_base64: Base64-encoded images attached to the question
                (requires a vision-capable analysis model).

        Returns:
            The answer as a string.
        """
        try:
            images = _decode_images(images_base64)
            rag = await _client()
            result = await rag.analyze(question, filter=filter, images=images)
            return result.answer
        except Exception as e:
            return f"Error running analysis capability: {e!s}"

    return mcp
