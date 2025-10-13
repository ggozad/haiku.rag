"""A2A worker implementation for conversational QA."""

import logging
import uuid
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from haiku.rag.a2a.context import load_message_history, save_message_history
from haiku.rag.a2a.models import AgentDependencies
from haiku.rag.a2a.skills import extract_question_from_task, extract_skill_preference
from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.graph.common import get_model
from haiku.rag.qa.deep.dependencies import DeepQAContext
from haiku.rag.qa.deep.graph import build_deep_qa_graph
from haiku.rag.qa.deep.nodes import DeepQAPlanNode
from haiku.rag.qa.deep.state import DeepQADeps, DeepQAState

try:
    from fasta2a import Worker  # type: ignore
    from fasta2a.schema import (  # type: ignore
        Artifact,
        DataPart,
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

    async def evaluate_answer_adequacy(self, question: str, answer: str) -> bool:
        """Use LLM to evaluate if answer adequately addresses the question.

        Args:
            question: The original question
            answer: The answer to evaluate

        Returns:
            True if answer is adequate, False if more research needed
        """

        class AnswerEvaluation(BaseModel):
            is_adequate: bool = Field(
                description="True if the answer adequately addresses the question, False if more research is needed"
            )
            reasoning: str = Field(description="Brief explanation of the evaluation")

        from .prompts import ANSWER_EVALUATION_PROMPT

        evaluation_agent = Agent(
            model=get_model(Config.QA_PROVIDER, Config.QA_MODEL),
            output_type=AnswerEvaluation,
            system_prompt=ANSWER_EVALUATION_PROMPT,
            retries=1,
        )

        prompt = f"""Question: {question}

Answer: {answer}

Does this answer adequately address the question?"""

        result = await evaluation_agent.run(prompt)
        logger.info(
            f"Answer evaluation: is_adequate={result.output.is_adequate}, reasoning={result.output.reasoning}"
        )
        return result.output.is_adequate

    async def run_task(self, params: TaskSendParams) -> None:
        task = await self.storage.load_task(params["id"])
        if task is None:
            raise ValueError(f"Task {params['id']} not found")

        if task["status"]["state"] != "submitted":
            raise ValueError(
                f"Task {params['id']} already processed: {task['status']['state']}"
            )

        await self.storage.update_task(task["id"], state="working")

        # Extract skill preference and question
        task_history = task.get("history", [])
        skill = extract_skill_preference(task_history)
        question = extract_question_from_task(task_history)

        if not question:
            await self.storage.update_task(task["id"], state="failed")
            return

        logger.info(f"Task {task['id']} requested skill: {skill}")

        try:
            async with HaikuRAG(self.db_path) as client:
                if skill == "deep-qa":
                    # Explicitly requested deep QA
                    logger.info(f"Task {task['id']}: Running deep QA (explicit)")
                    deep_result, deep_state = await self.run_deep_qa(client, question)

                    response_message = Message(
                        role="agent",
                        parts=[TextPart(kind="text", text=deep_result.answer)],
                        kind="message",
                        message_id=str(uuid.uuid4()),
                    )

                    artifacts = self.build_deep_qa_artifacts(deep_result, deep_state)

                    await self.storage.update_task(
                        task["id"],
                        state="completed",
                        new_messages=[response_message],
                        new_artifacts=artifacts,
                    )
                else:
                    # Try simple QA first (default behavior or explicit document-qa)
                    logger.info(f"Task {task['id']}: Trying simple QA first")

                    context = await self.storage.load_context(task["context_id"]) or []
                    message_history = load_message_history(context)

                    from .models import AgentDependencies

                    deps = AgentDependencies(client=client)

                    result = await self.agent.run(
                        question, deps=deps, message_history=message_history
                    )

                    answer = str(result.output)

                    # Evaluate answer adequacy
                    is_adequate = await self.evaluate_answer_adequacy(question, answer)

                    if not is_adequate:
                        # Escalate to deep QA
                        logger.info(
                            f"Task {task['id']}: Answer inadequate, escalating to deep QA"
                        )
                        deep_result, deep_state = await self.run_deep_qa(
                            client, question
                        )

                        response_message = Message(
                            role="agent",
                            parts=[TextPart(kind="text", text=deep_result.answer)],
                            kind="message",
                            message_id=str(uuid.uuid4()),
                        )

                        artifacts = self.build_deep_qa_artifacts(
                            deep_result, deep_state
                        )

                        await self.storage.update_task(
                            task["id"],
                            state="completed",
                            new_messages=[response_message],
                            new_artifacts=artifacts,
                        )
                    else:
                        # Simple QA answer is adequate
                        logger.info(f"Task {task['id']}: Simple QA answer is adequate")

                        response_message = Message(
                            role="agent",
                            parts=[TextPart(kind="text", text=answer)],
                            kind="message",
                            message_id=str(uuid.uuid4()),
                        )

                        # Update context with complete conversation state
                        updated_history = message_history + result.new_messages()
                        state_message = save_message_history(updated_history)

                        await self.storage.update_context(
                            task["context_id"], [state_message]
                        )

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

    async def run_deep_qa(self, client: HaikuRAG, question: str):
        """Run deep QA graph for complex questions.

        Args:
            client: HaikuRAG client
            question: User's question

        Returns:
            Tuple of (DeepQAAnswer, DeepQAState) with answer and state
        """
        graph = build_deep_qa_graph()
        context = DeepQAContext(original_question=question, use_citations=False)
        state = DeepQAState(context=context)
        deps = DeepQADeps(client=client, console=None)
        start_node = DeepQAPlanNode(provider=Config.QA_PROVIDER, model=Config.QA_MODEL)

        result = await graph.run(start_node=start_node, state=state, deps=deps)
        return result.output, state

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

    def build_deep_qa_artifacts(self, result, state: DeepQAState) -> list[Artifact]:
        """Build rich artifacts from deep QA result.

        Args:
            result: DeepQAAnswer with final answer
            state: DeepQAState with research process details

        Returns:
            List of artifacts including answer and research breakdown
        """
        artifacts = [
            # Final answer artifact
            Artifact(
                artifact_id=str(uuid.uuid4()),
                name="answer",
                parts=[TextPart(kind="text", text=result.answer)],
            )
        ]

        # Add research process artifact with sub-questions and answers
        if state.context.qa_responses:
            research_data = {
                "original_question": state.context.original_question,
                "iterations": state.iterations,
                "sub_questions_answered": [
                    {
                        "question": qa.query,
                        "answer": qa.answer,
                        "sources": qa.sources,
                    }
                    for qa in state.context.qa_responses
                ],
            }

            artifacts.append(
                Artifact(
                    artifact_id=str(uuid.uuid4()),
                    name="research_process",
                    parts=[
                        DataPart(
                            kind="data",
                            data=research_data,
                            metadata={"type": "deep_qa_research"},
                        )
                    ],
                )
            )

        return artifacts
