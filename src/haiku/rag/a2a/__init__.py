"""A2A (Agent-to-Agent) server integration for haiku.rag."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import logfire
from pydantic_ai import Agent, RunContext

from haiku.rag.config import Config
from haiku.rag.graph.common import get_model

from .context import load_message_history, save_message_history
from .models import AgentDependencies, SearchResult
from .prompts import A2A_SYSTEM_PROMPT
from .skills import (
    extract_question_from_task,
    extract_skill_preference,
    get_agent_skills,
)
from .storage import LRUMemoryStorage
from .worker import ConversationalWorker

try:
    from fasta2a import FastA2A  # type: ignore
    from fasta2a.broker import InMemoryBroker  # type: ignore
    from fasta2a.storage import InMemoryStorage  # type: ignore
except ImportError as e:
    raise ImportError(
        "A2A support requires the 'a2a' extra. "
        "Install with: uv pip install 'haiku.rag[a2a]'"
    ) from e

logfire.configure(send_to_logfire="if-token-present", service_name="a2a")
logfire.instrument_pydantic_ai()

logger = logging.getLogger(__name__)

__all__ = [
    "create_a2a_app",
    "load_message_history",
    "save_message_history",
    "extract_question_from_task",
    "extract_skill_preference",
    "get_agent_skills",
    "LRUMemoryStorage",
]


def create_a2a_app(db_path: Path):
    """Create an A2A app for the conversational QA agent.

    Args:
        db_path: Path to the LanceDB database

    Returns:
        A FastA2A ASGI application
    """
    base_storage = InMemoryStorage()
    storage = LRUMemoryStorage(
        storage=base_storage, max_contexts=Config.A2A_MAX_CONTEXTS
    )
    broker = InMemoryBroker()

    # Create the agent with native search tool
    model = get_model(Config.QA_PROVIDER, Config.QA_MODEL)
    agent = Agent(
        model=model,
        deps_type=AgentDependencies,
        system_prompt=A2A_SYSTEM_PROMPT,
        retries=3,
    )

    @agent.tool
    async def search_documents(
        ctx: RunContext[AgentDependencies],
        query: str,
        limit: int = 3,
    ) -> list[SearchResult]:
        """Search the knowledge base for relevant documents.

        Returns chunks of text with their relevance scores and document URIs.
        Use get_full_document if you need to see the complete document content.
        """
        search_results = await ctx.deps.client.search(query, limit=limit)
        expanded_results = await ctx.deps.client.expand_context(search_results)

        return [
            SearchResult(
                content=chunk.content,
                score=score,
                document_title=chunk.document_title,
                document_uri=(chunk.document_uri or ""),
            )
            for chunk, score in expanded_results
        ]

    @agent.tool
    async def get_full_document(
        ctx: RunContext[AgentDependencies],
        document_uri: str,
    ) -> str:
        """Retrieve the complete content of a document by its URI.

        Use this when you need more context than what's in a search result chunk.
        The document_uri comes from search_documents results.
        """
        document = await ctx.deps.client.get_document_by_uri(document_uri)
        if document is None:
            return f"Document not found: {document_uri}"

        return document.content

    @agent.tool
    async def list_documents(
        ctx: RunContext[AgentDependencies],
        limit: int = 10,
    ) -> list[str]:
        """List documents in the knowledge base.

        Returns document URIs/titles. Use this to help users discover what's available.
        """
        documents = await ctx.deps.client.list_documents(limit=limit)
        return [doc.title or doc.uri or f"Document {doc.id}" for doc in documents]

    worker = ConversationalWorker(
        storage=storage,
        broker=broker,
        db_path=db_path,
        agent=agent,  # type: ignore
    )

    # Create FastA2A app with custom worker lifecycle
    @asynccontextmanager
    async def lifespan(app):
        logger.info(f"Started A2A server (max contexts: {Config.A2A_MAX_CONTEXTS})")
        async with app.task_manager:
            async with worker.run():
                yield

    return FastA2A(
        storage=storage,
        broker=broker,
        name="haiku-rag",
        description="Conversational question answering agent powered by haiku.rag RAG system",
        skills=get_agent_skills(),
        lifespan=lifespan,
    )
