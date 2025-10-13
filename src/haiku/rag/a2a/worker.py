"""A2A worker implementation for conversational QA."""

import logging
import uuid
from pathlib import Path

from pydantic_ai import Agent

from haiku.rag.a2a.context import load_message_history, save_message_history
from haiku.rag.a2a.models import AgentDependencies
from haiku.rag.a2a.skills import extract_question_from_task
from haiku.rag.client import HaikuRAG

try:
    from fasta2a import Worker  # type: ignore
    from fasta2a.schema import (  # type: ignore
        Artifact,
        Message,
        TaskIdParams,
        TaskSendParams,
        TextPart,
    )
except ImportError as e:
    raise ImportError(
        "A2A support requires the 'a2a' extra. "
        "Install with: uv pip install 'haiku.rag[a2a]'"
    ) from e

logger = logging.getLogger(__name__)


class ConversationalWorker(Worker[list[Message]]):
    """Worker that handles conversational QA tasks."""

    def __init__(
        self,
        storage,
        broker,
        db_path: Path,
        agent: "Agent[AgentDependencies, str]",
    ):
        super().__init__(storage=storage, broker=broker)
        self.db_path = db_path
        self.agent = agent

    async def run_task(self, params: TaskSendParams) -> None:
        task = await self.storage.load_task(params["id"])
        if task is None:
            raise ValueError(f"Task {params['id']} not found")

        if task["status"]["state"] != "submitted":
            raise ValueError(
                f"Task {params['id']} already processed: {task['status']['state']}"
            )

        await self.storage.update_task(task["id"], state="working")

        task_history = task.get("history", [])
        question = extract_question_from_task(task_history)

        if not question:
            await self.storage.update_task(task["id"], state="failed")
            return

        try:
            async with HaikuRAG(self.db_path) as client:
                context = await self.storage.load_context(task["context_id"]) or []
                message_history = load_message_history(context)

                from haiku.rag.a2a.models import AgentDependencies

                deps = AgentDependencies(client=client)

                result = await self.agent.run(
                    question, deps=deps, message_history=message_history
                )

                answer = str(result.output)

                response_message = Message(
                    role="agent",
                    parts=[TextPart(kind="text", text=answer)],
                    kind="message",
                    message_id=str(uuid.uuid4()),
                )

                # Update context with complete conversation state
                updated_history = message_history + result.new_messages()
                state_message = save_message_history(updated_history)

                await self.storage.update_context(task["context_id"], [state_message])

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
