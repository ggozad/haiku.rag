import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import logfire
from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config

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


def a2a_to_pydantic_messages(a2a_messages: list[Message]) -> list[ModelMessage]:
    """Convert A2A messages to pydantic-ai ModelMessage format.

    Args:
        a2a_messages: List of A2A Message objects

    Returns:
        List of pydantic-ai ModelMessage objects suitable for agent.run()
    """
    pydantic_messages = []

    for msg in a2a_messages:
        role = msg.get("role", "user")
        parts = msg.get("parts", [])

        # Extract text content from all text parts
        text_content = " ".join(
            part.get("text", "") for part in parts if part.get("kind") == "text"
        )

        if not text_content:
            continue

        # Build message dict with proper part_kind discriminators
        if role == "user":
            pydantic_messages.append(
                {
                    "parts": [{"content": text_content, "part_kind": "user-prompt"}],
                    "kind": "request",
                }
            )
        elif role == "agent":
            # Agent responses become ModelResponse with TextPart
            pydantic_messages.append(
                {
                    "parts": [{"content": text_content, "part_kind": "text"}],
                    "kind": "response",
                    "model_name": "unknown",
                }
            )

    # Validate and convert to proper ModelMessage objects
    if pydantic_messages:
        return ModelMessagesTypeAdapter.validate_python(pydantic_messages)

    return []


def create_qa_a2a_app(
    db_path: Path,
    deep: bool = False,
):
    """Create an A2A app for the QA agent.

    Args:
        db_path: Path to the LanceDB database
        deep: Use deep multi-agent QA for complex questions

    Returns:
        A FastA2A ASGI application
    """
    if deep:
        raise NotImplementedError("Deep QA agent not yet implemented for A2A")

    from haiku.rag.qa.agent import Dependencies, QuestionAnswerAgent

    # Create the agent (client will be provided per-task in custom worker)
    temp_client = HaikuRAG(db_path)
    qa_agent = QuestionAnswerAgent(
        client=temp_client,
        provider=Config.QA_PROVIDER,
        model=Config.QA_MODEL,
    )

    # Create custom worker using base Worker class
    storage = InMemoryStorage()
    broker = InMemoryBroker()

    class QAWorker(Worker[list[Message]]):
        async def run_task(self, params: TaskSendParams) -> None:
            task = await self.storage.load_task(params["id"])
            if task is None:
                raise ValueError(f"Task {params['id']} not found")

            if task["status"]["state"] != "submitted":
                raise ValueError(
                    f"Task {params['id']} already processed: {task['status']['state']}"
                )

            await self.storage.update_task(task["id"], state="working")

            # Load full conversation context from previous tasks
            context = await self.storage.load_context(task["context_id"]) or []
            current_task_history = task.get("history", [])

            # Extract the user's question from the latest message
            user_messages = [
                msg for msg in current_task_history if msg["role"] == "user"
            ]
            if not user_messages:
                await self.storage.update_task(task["id"], state="failed")
                return

            last_user_msg = user_messages[-1]
            question = ""
            for part in last_user_msg.get("parts", []):
                if part.get("kind") == "text":
                    question = part.get("text", "")
                    break

            try:
                # Create fresh client for this task and run QA agent
                async with HaikuRAG(db_path) as client:
                    deps = Dependencies(client=client)

                    # Convert conversation history to pydantic-ai format
                    message_history = a2a_to_pydantic_messages(context)

                    # Run agent with full conversation history
                    result = await qa_agent._agent.run(
                        question, deps=deps, message_history=message_history
                    )

                    # Build response message
                    response_message = Message(
                        role="agent",
                        parts=[TextPart(kind="text", text=str(result.output))],
                        kind="message",
                        message_id=str(uuid.uuid4()),
                    )

                    # Store complete agent state (all messages including tool calls)
                    # Add both the user question and agent response to context
                    context.extend(current_task_history)
                    context.append(response_message)
                    await self.storage.update_context(task["context_id"], context)

                    # Build rich artifacts with search results and answer
                    artifacts = self.build_artifacts(result)

                    await self.storage.update_task(
                        task["id"],
                        state="completed",
                        new_messages=[response_message],
                        new_artifacts=artifacts,
                    )
            except Exception:
                await self.storage.update_task(task["id"], state="failed")
                raise

        async def cancel_task(self, params: TaskIdParams) -> None:
            pass

        def build_message_history(self, history: list[Message]) -> list[Message]:
            return history

        def build_artifacts(self, result) -> list[Artifact]:
            """Build rich artifacts from agent result including search details."""
            artifacts: list[Artifact] = []

            # Main answer artifact
            artifacts.append(
                Artifact(
                    artifact_id=str(uuid.uuid4()),
                    name="answer",
                    parts=[TextPart(kind="text", text=str(result.output))],
                )
            )

            # Extract search tool calls and results from message history
            search_results = []
            for msg in result.all_messages():
                if isinstance(msg, ModelResponse):
                    for part in msg.parts:
                        if isinstance(part, ToolCallPart):
                            if part.tool_name == "search_documents":
                                search_results.append(
                                    {
                                        "tool_call": part.tool_name,
                                        "args": part.args,
                                    }
                                )

            # Create search results artifact if we found any searches
            if search_results:
                artifacts.append(
                    Artifact(
                        artifact_id=str(uuid.uuid4()),
                        name="search_activity",
                        parts=[
                            DataPart(
                                kind="data",
                                data={
                                    "searches": search_results,
                                    "count": len(search_results),
                                },
                                metadata={"type": "search_history"},
                            )
                        ],
                    )
                )

            return artifacts

    worker = QAWorker(storage=storage, broker=broker)

    # Create FastA2A app with custom worker lifecycle
    @asynccontextmanager
    async def lifespan(app):
        async with app.task_manager:
            async with worker.run():
                yield

    return FastA2A(
        storage=storage,
        broker=broker,
        name="haiku-rag-qa",
        description="Question answering agent powered by haiku.rag RAG system",
        lifespan=lifespan,
    )


def create_research_a2a_app(db_path: Path):
    """Create an A2A app for the research agent.

    Args:
        db_path: Path to the LanceDB database

    Returns:
        A FastA2A ASGI application
    """
    raise NotImplementedError("Research agent not yet implemented for A2A")
