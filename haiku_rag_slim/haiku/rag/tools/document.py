from pydantic import BaseModel
from pydantic_ai import Agent, FunctionToolset

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.tools.context import ToolContext
from haiku.rag.tools.filters import get_session_filter
from haiku.rag.utils import get_model

DOCUMENT_NAMESPACE = "haiku.rag.document"

DOCUMENT_SUMMARY_PROMPT = """Generate a summary of the document content provided below.

Start with a one-paragraph overview, then list the main topics covered, and highlight any key findings or conclusions.

Guidelines:
- Aim for 1-2 paragraphs for short documents, 3-4 paragraphs for longer ones
- Focus on factual content and key information
- Do not include meta-commentary like "This document discusses..." or "The document covers..."
- Do not speculate beyond what's in the content

Document content:
{content}"""


class DocumentInfo(BaseModel):
    """Document info for list_documents response."""

    title: str
    uri: str
    created: str


class DocumentListResponse(BaseModel):
    """Response from list_documents tool."""

    documents: list[DocumentInfo]
    page: int
    total_pages: int
    total_documents: int


class DocumentState(BaseModel):
    """State for document toolset.

    Tracks documents accessed during tool invocations.
    """

    accessed_documents: list[DocumentInfo] = []


async def find_document(client: HaikuRAG, query: str):
    """Find a document by exact URI, partial URI, or partial title match."""
    # Try exact URI match first
    doc = await client.get_document_by_uri(query)
    if doc is not None:
        return doc

    escaped_query = query.replace("'", "''")
    # Also try without spaces for matching "TB MED 593" to "tbmed593"
    no_spaces = escaped_query.replace(" ", "")

    # Try partial URI match (with and without spaces)
    docs = await client.list_documents(
        limit=1,
        filter=f"LOWER(uri) LIKE LOWER('%{escaped_query}%') OR LOWER(uri) LIKE LOWER('%{no_spaces}%')",
    )
    if docs:
        return docs[0]

    # Try partial title match (with and without spaces)
    docs = await client.list_documents(
        limit=1,
        filter=f"LOWER(title) LIKE LOWER('%{escaped_query}%') OR LOWER(title) LIKE LOWER('%{no_spaces}%')",
    )
    if docs:
        return docs[0]

    return None


def create_document_toolset(
    client: HaikuRAG,
    config: AppConfig,
    context: ToolContext | None = None,
    base_filter: str | None = None,
) -> FunctionToolset:
    """Create a toolset with document management capabilities.

    Args:
        client: HaikuRAG client for document operations.
        config: Application configuration (used for summarization LLM).
        context: Optional ToolContext for state tracking.
            If provided, accessed documents are tracked in DocumentState.
            If SessionState is registered, it will be used for dynamic
            document filtering.
        base_filter: Optional base SQL WHERE clause applied to list operations.

    Returns:
        FunctionToolset with list_documents, get_document, summarize_document tools.
    """
    # Get or create document state if context provided
    state: DocumentState | None = None
    if context is not None:
        state = context.get_or_create(DOCUMENT_NAMESPACE, DocumentState)

    async def list_documents(page: int = 1) -> DocumentListResponse:
        """List available documents in the knowledge base.

        Args:
            page: Page number (default: 1, 50 documents per page)

        Returns:
            Paginated list of documents with metadata.
        """
        page_size = 50
        offset = (page - 1) * page_size

        effective_filter = get_session_filter(context, base_filter)

        docs = await client.list_documents(
            limit=page_size, offset=offset, filter=effective_filter
        )
        total = await client.count_documents(filter=effective_filter)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1

        return DocumentListResponse(
            documents=[
                DocumentInfo(
                    title=doc.title or "Untitled",
                    uri=doc.uri or "",
                    created=doc.created_at.strftime("%Y-%m-%d"),
                )
                for doc in docs
            ],
            page=page,
            total_pages=total_pages,
            total_documents=total,
        )

    async def get_document(query: str) -> str:
        """Retrieve a specific document by title or URI.

        Args:
            query: The document title or URI to look up.

        Returns:
            Document content and metadata, or not found message.
        """
        doc = await find_document(client, query)

        if doc is None:
            return f"Document not found: {query}"

        # Track accessed document in state
        if state is not None:
            state.accessed_documents.append(
                DocumentInfo(
                    title=doc.title or "Untitled",
                    uri=doc.uri or "",
                    created=doc.created_at.strftime("%Y-%m-%d"),
                )
            )

        return (
            f"**{doc.title or 'Untitled'}**\n\n"
            f"- ID: {doc.id}\n"
            f"- URI: {doc.uri}\n"
            f"- Created: {doc.created_at.strftime('%Y-%m-%d %H:%M')}\n\n"
            f"**Content:**\n{doc.content}"
        )

    async def summarize_document(query: str) -> str:
        """Generate a summary of a specific document.

        Args:
            query: The document title or URI to summarize.

        Returns:
            Generated summary or not found message.
        """
        doc = await find_document(client, query)

        if doc is None:
            return f"Document not found: {query}"

        # Track accessed document in state
        if state is not None:
            state.accessed_documents.append(
                DocumentInfo(
                    title=doc.title or "Untitled",
                    uri=doc.uri or "",
                    created=doc.created_at.strftime("%Y-%m-%d"),
                )
            )

        # Use LLM to generate summary
        summary_model = get_model(config.qa.model, config)
        summary_agent: Agent[None, str] = Agent(
            summary_model,
            output_type=str,
        )
        result = await summary_agent.run(
            DOCUMENT_SUMMARY_PROMPT.format(content=doc.content or "")
        )

        return f"**Summary of {doc.title or doc.uri}:**\n\n{result.output}"

    toolset = FunctionToolset()
    toolset.add_function(list_documents)
    toolset.add_function(get_document)
    toolset.add_function(summarize_document)
    return toolset
