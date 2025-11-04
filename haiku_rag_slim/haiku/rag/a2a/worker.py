import json
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

                deps = AgentDependencies(client=client)

                result = await self.agent.run(
                    question, deps=deps, message_history=message_history
                )

                # Detect which skill was used
                skill_type = self._detect_skill(result)

                # Build messages based on skill type
                response_messages = self._build_response_messages(result, skill_type)

                # Update context with complete conversation state
                updated_history = message_history + result.new_messages()
                state_message = save_message_history(updated_history)

                await self.storage.update_context(task["context_id"], [state_message])

                artifacts = self.build_artifacts(result, skill_type, question)

                await self.storage.update_task(
                    task["id"],
                    state="completed",
                    new_messages=response_messages,
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

    def _detect_skill(self, result) -> str:
        """Detect which skill was used based on tool calls and response pattern.

        Returns:
            "search", "retrieve", or "qa"
        """
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        tool_calls = []
        for msg in result.new_messages():
            if isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, ToolCallPart):
                        tool_calls.append(part.tool_name)

        # Check if output looks like formatted search results
        output_str = str(result.output).strip()
        # Check for either format: "Found N relevant results" or "**Search results for"
        is_search_format = (
            output_str.startswith("Found ") and "relevant results" in output_str[:100]
        ) or output_str.startswith("**Search results for")

        skill_type = "qa"
        # If output is in search format and only search tools were used, it's a search
        if is_search_format and all(tc == "search_documents" for tc in tool_calls):
            skill_type = "search"
        elif "get_full_document" in tool_calls and len(tool_calls) == 1:
            skill_type = "retrieve"

        return skill_type

    def _build_response_messages(self, result, skill_type: str) -> list[Message]:
        """Build response messages based on skill type.

        All skills return a single text message with LLM's response.
        Structured data is provided via artifacts for search/retrieve.
        """
        if skill_type == "search":
            # Return LLM's formatted response
            return [
                Message(
                    role="agent",
                    parts=[TextPart(kind="text", text=str(result.output))],
                    kind="message",
                    message_id=str(uuid.uuid4()),
                )
            ]
        elif skill_type == "retrieve":
            # Extract document content
            from pydantic_ai.messages import ModelRequest, ToolReturnPart

            document_content = ""
            for msg in result.new_messages():
                if isinstance(msg, ModelRequest):
                    for part in msg.parts:
                        if (
                            isinstance(part, ToolReturnPart)
                            and part.tool_name == "get_full_document"
                        ):
                            document_content = part.content
                            break

            return [
                Message(
                    role="agent",
                    parts=[TextPart(kind="text", text=document_content)],
                    kind="message",
                    message_id=str(uuid.uuid4()),
                )
            ]
        else:
            # Conversational Q&A - use agent's answer
            return [
                Message(
                    role="agent",
                    parts=[TextPart(kind="text", text=str(result.output))],
                    kind="message",
                    message_id=str(uuid.uuid4()),
                )
            ]

    def build_artifacts(
        self, result, skill_type: str | None = None, question: str | None = None
    ) -> list[Artifact]:
        """Build artifacts from agent result based on tool calls.

        Creates artifacts for:
        - Each tool call (search_documents, get_full_document)
        - Q&A operations: additional artifact with question and answer (only if tools were used)
        """
        if skill_type is None:
            skill_type = self._detect_skill(result)

        artifacts = []

        # Always create artifacts for all tool calls
        tool_artifacts = self._build_all_tool_artifacts(result)
        artifacts.extend(tool_artifacts)

        # For Q&A, always add a Q&A artifact with question and answer
        # This includes follow-up questions, clarifications, and conversational responses
        if skill_type == "qa" and question:
            from fasta2a.schema import DataPart

            artifacts.append(
                Artifact(
                    artifact_id=str(uuid.uuid4()),
                    name="qa_result",
                    parts=[
                        DataPart(
                            kind="data",
                            data={
                                "question": question,
                                "answer": str(result.output),
                                "skill": "document-qa",
                            },
                            metadata={"skill": "document-qa"},
                        )
                    ],
                )
            )

        return artifacts

    def _build_all_tool_artifacts(self, result) -> list[Artifact]:
        """Build artifacts for all tool calls."""
        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
            ToolCallPart,
            ToolReturnPart,
        )

        artifacts = []

        # Track tool calls and their returns by call_id
        tool_returns = {}
        for msg in result.new_messages():
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        result_count = (
                            len(part.content) if isinstance(part.content, list) else 1
                        )
                        logger.info(
                            "Tool return: tool_call_id=%s, tool_name=%s, result_count=%s",
                            part.tool_call_id,
                            part.tool_name,
                            result_count,
                        )
                        tool_returns[part.tool_call_id] = (part.tool_name, part.content)

        # Create artifacts for each tool call
        for msg in result.new_messages():
            if isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, ToolCallPart):
                        tool_name, content = tool_returns.get(
                            part.tool_call_id, (None, None)
                        )

                        if tool_name == "search_documents" and content:
                            from fasta2a.schema import DataPart

                            # Extract query from tool call arguments
                            query = ""
                            if isinstance(part.args, dict):
                                query = part.args.get("query", "")
                            elif isinstance(part.args, str):
                                # Args is a JSON string - parse it
                                try:
                                    args_dict = json.loads(part.args)
                                    query = args_dict.get("query", "")
                                except (json.JSONDecodeError, AttributeError):
                                    query = ""
                            elif hasattr(part.args, "get") and callable(
                                getattr(part.args, "get", None)
                            ):
                                # ArgsDict or dict-like object
                                query = part.args.get("query", "")  # type: ignore
                            elif hasattr(part.args, "query"):
                                # Object with query attribute
                                query = str(part.args.query)  # type: ignore

                            artifacts.append(
                                Artifact(
                                    artifact_id=str(uuid.uuid4()),
                                    name="search_results",
                                    parts=[
                                        DataPart(
                                            kind="data",
                                            data={"results": content, "query": query},
                                            metadata={"query": query},
                                        )
                                    ],
                                )
                            )
                        elif tool_name == "get_full_document" and content:
                            artifacts.append(
                                Artifact(
                                    artifact_id=str(uuid.uuid4()),
                                    name="document",
                                    parts=[TextPart(kind="text", text=content)],
                                )
                            )

        return artifacts
