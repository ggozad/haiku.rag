import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import logfire
from pydantic import BaseModel, TypeAdapter
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_core import to_jsonable_python

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.graph.common import get_model
from haiku.rag.qa.agent import SearchResult

logger = logging.getLogger(__name__)

try:
    from fasta2a import FastA2A, Worker  # type: ignore
    from fasta2a.broker import InMemoryBroker  # type: ignore
    from fasta2a.schema import (  # type: ignore
        Artifact,
        DataPart,
        Message,
        TaskIdParams,
        TaskSendParams,
        TextPart,
    )
    from fasta2a.storage import InMemoryStorage  # type: ignore
except ImportError as e:
    raise ImportError(
        "A2A support requires the 'a2a' extra. "
        "Install with: uv pip install 'haiku.rag[a2a]'"
    ) from e

logfire.configure(send_to_logfire="if-token-present", service_name="a2a")
logfire.instrument_pydantic_ai()

ModelMessagesTypeAdapter = TypeAdapter(list[ModelMessage])


class AgentDependencies(BaseModel):
    """Dependencies for the A2A conversational agent."""

    model_config = {"arbitrary_types_allowed": True}
    client: HaikuRAG


A2A_SYSTEM_PROMPT = """You are Haiku.rag, an AI assistant that helps users find information from a document knowledge base.

IMPORTANT: You are NOT any person mentioned in the documents. You retrieve and present information about them.

Tools available:
- search_documents: Query for relevant text chunks
- get_full_document: Get complete document content by document_uri
- list_documents: Show available documents

Your process:
1. Search phase: For straightforward questions use one search, for complex questions search multiple times with different queries
2. Synthesis phase: Combine the search results into a comprehensive answer
3. When user requests full document: use get_full_document with the exact document_uri from Sources

Critical rules:
- ONLY answer based on information found via search_documents
- NEVER fabricate or assume information
- If not found, say: "I cannot find information about this in the knowledge base."
- For follow-ups, understand context (pronouns like "he", "it") but always search for facts
- ALWAYS include citations at the end showing document URIs used
- Be concise and direct

Citation Format:
After your answer, include a "Sources:" section listing document URIs from search results.
Format: "Sources:\n- [document_uri]"

Example:
[Your answer here]

Sources:
- /path/to/document.pdf
- /another/document.md
"""


def load_message_history(context: list[Message]) -> list[ModelMessage]:
    """Load pydantic-ai message history from A2A context.

    The context stores serialized pydantic-ai message history directly,
    which we deserialize and return.

    Args:
        context: A2A context messages

    Returns:
        List of pydantic-ai ModelMessage objects
    """
    if not context:
        return []

    # Context should contain a single "state" message with full history
    for msg in context:
        parts = msg.get("parts", [])
        for part in parts:
            if part.get("kind") == "data":
                metadata = part.get("metadata", {})
                if metadata.get("type") == "conversation_state":
                    stored_history = part.get("data", {}).get("message_history", [])
                    if stored_history:
                        return ModelMessagesTypeAdapter.validate_python(stored_history)

    return []


def save_message_history(message_history: list[ModelMessage]) -> Message:
    """Save pydantic-ai message history to A2A context format.

    Args:
        message_history: Full pydantic-ai message history

    Returns:
        A2A Message containing the serialized state (stored as agent role)
    """
    serialized = to_jsonable_python(message_history)
    return Message(
        role="agent",
        parts=[
            DataPart(
                kind="data",
                data={"message_history": serialized},
                metadata={"type": "conversation_state"},
            )
        ],
        kind="message",
        message_id=str(uuid.uuid4()),
    )


def extract_question_from_task(task_history: list[Message]) -> str | None:
    """Extract the user's question from task history.

    Args:
        task_history: Task history messages

    Returns:
        The question text if found, None otherwise
    """
    for msg in task_history:
        if msg.get("role") == "user":
            for part in msg.get("parts", []):
                if part.get("kind") == "text":
                    text = part.get("text", "").strip()
                    if text:
                        return text
    return None


def create_a2a_app(db_path: Path):
    """Create an A2A app for the conversational QA agent.

    Args:
        db_path: Path to the LanceDB database

    Returns:
        A FastA2A ASGI application
    """
    storage = InMemoryStorage()
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
        # Remove quotes from queries as this requires positional indexing in lancedb
        query = query.replace('"', "")
        search_results = await ctx.deps.client.search(query, limit=limit)
        expanded_results = await ctx.deps.client.expand_context(search_results)

        return [
            SearchResult(
                content=chunk.content,
                score=score,
                document_uri=(chunk.document_title or chunk.document_uri or ""),
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

    class ConversationalWorker(Worker[list[Message]]):
        async def run_task(self, params: TaskSendParams) -> None:
            task = await self.storage.load_task(params["id"])
            if task is None:
                raise ValueError(f"Task {params['id']} not found")

            if task["status"]["state"] != "submitted":
                raise ValueError(
                    f"Task {params['id']} already processed: {task['status']['state']}"
                )

            await self.storage.update_task(task["id"], state="working")

            # Extract the user's question
            question = extract_question_from_task(task.get("history", []))
            if not question:
                await self.storage.update_task(task["id"], state="failed")
                return

            try:
                # Load conversation context
                context = await self.storage.load_context(task["context_id"]) or []
                # Load conversation history
                message_history = load_message_history(context)

                # Create fresh client for this task and run agent
                async with HaikuRAG(db_path) as client:
                    deps = AgentDependencies(client=client)

                    # Run agent with full conversation history including tool calls
                    result = await agent.run(
                        question, deps=deps, message_history=message_history
                    )

                    # Build response message for A2A protocol
                    response_message = Message(
                        role="agent",
                        parts=[TextPart(kind="text", text=str(result.output))],
                        kind="message",
                        message_id=str(uuid.uuid4()),
                    )

                    # Update context with complete conversation state
                    # Store all messages from this run (includes tool calls & results)
                    updated_history = message_history + result.new_messages()
                    state_message = save_message_history(updated_history)

                    # Replace old state with new complete state
                    await self.storage.update_context(
                        task["context_id"], [state_message]
                    )

                    # Build rich artifacts with search results and answer
                    artifacts = self.build_artifacts(result)

                    await self.storage.update_task(
                        task["id"],
                        state="completed",
                        new_messages=[response_message],
                        new_artifacts=artifacts,
                    )
            except Exception as e:
                logger.error(
                    "Task execution failed: task_id=%s, question=%s, error=%s",
                    task["id"],
                    question,
                    str(e),
                    exc_info=True,
                )
                await self.storage.update_task(task["id"], state="failed")
                raise

        async def cancel_task(self, params: TaskIdParams) -> None:
            """Cancel a task - not implemented for this worker."""
            pass

        def build_message_history(self, history: list[Message]) -> list[Message]:
            """Required by Worker interface but unused - history stored in context."""
            return history

        def build_artifacts(self, result) -> list[Artifact]:
            """Build artifacts from agent result.

            Note: Full conversation history (including tool calls) is stored in
            context, so we only create a simple answer artifact here.
            """
            return [
                Artifact(
                    artifact_id=str(uuid.uuid4()),
                    name="answer",
                    parts=[TextPart(kind="text", text=str(result.output))],
                )
            ]

    worker = ConversationalWorker(storage=storage, broker=broker)

    # Create FastA2A app with custom worker lifecycle
    @asynccontextmanager
    async def lifespan(app):
        async with app.task_manager:
            async with worker.run():
                yield

    return FastA2A(
        storage=storage,
        broker=broker,
        name="haiku-rag",
        description="Conversational question answering agent powered by haiku.rag RAG system",
        lifespan=lifespan,
    )
