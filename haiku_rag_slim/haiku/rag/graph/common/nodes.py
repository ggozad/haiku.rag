"""Common node implementations for graph workflows."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from pydantic_ai import Agent, RunContext
from pydantic_ai.output import ToolOutput
from pydantic_graph.beta import StepContext

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.config.models import AppConfig, ModelConfig
from haiku.rag.graph.agui.emitter import AGUIEmitter
from haiku.rag.graph.common import get_model
from haiku.rag.graph.common.models import RawSearchAnswer, ResearchPlan, SearchAnswer
from haiku.rag.graph.common.prompts import PLAN_PROMPT, SEARCH_AGENT_PROMPT
from haiku.rag.store.models import SearchResult


class GraphContext(Protocol):
    """Protocol for graph context objects."""

    original_question: str
    sub_questions: list[str]

    def add_qa_response(self, qa: SearchAnswer) -> None:
        """Add a QA response to context."""
        ...


class GraphState(Protocol):
    """Protocol for graph state objects."""

    context: GraphContext
    max_concurrency: int
    search_filter: str | None


class GraphDeps(Protocol):
    """Protocol for graph dependencies."""

    client: HaikuRAG
    agui_emitter: AGUIEmitter[Any, Any] | None
    semaphore: asyncio.Semaphore | None


class GraphAgentDeps(Protocol):
    """Protocol for agent dependencies."""

    client: HaikuRAG
    context: GraphContext
    search_results: list[SearchResult]


def create_plan_node[AgentDepsT: GraphAgentDeps](
    model_config: ModelConfig,
    deps_type: type[AgentDepsT],
    activity_message: str = "Creating plan",
    output_retries: int | None = None,
    config: AppConfig = Config,
) -> Callable[[StepContext[Any, Any, None]], Awaitable[None]]:
    """Create a plan node for any graph.

    Args:
        model_config: ModelConfig with provider, model, and settings
        deps_type: Type of dependencies for the agent (e.g., ResearchDependencies, DeepQADependencies)
        activity_message: Message to show during planning activity
        output_retries: Number of output retries for the agent (optional)
        config: AppConfig object (defaults to global Config)

    Returns:
        Async function that can be used as a graph step
    """

    async def plan(ctx: StepContext[Any, Any, None], /) -> None:
        state: GraphState = ctx.state  # type: ignore[assignment]
        deps: GraphDeps = ctx.deps  # type: ignore[assignment]

        if deps.agui_emitter:
            deps.agui_emitter.start_step("plan")
            deps.agui_emitter.update_activity(
                "planning", {"stepName": "plan", "message": activity_message}
            )

        try:
            # Build agent configuration
            agent_config = {
                "model": get_model(model_config, config),
                "output_type": ResearchPlan,
                "instructions": (
                    PLAN_PROMPT
                    + "\n\nUse the gather_context tool once on the main question before planning."
                ),
                "retries": 3,
                "deps_type": deps_type,
            }
            if output_retries is not None:
                agent_config["output_retries"] = output_retries

            plan_agent = Agent(**agent_config)

            # Capture search filter for use in tool
            search_filter = state.search_filter

            @plan_agent.tool
            async def gather_context(
                ctx2: RunContext[AgentDepsT], query: str, limit: int | None = None
            ) -> str:
                results = await ctx2.deps.client.search(
                    query, limit=limit, filter=search_filter
                )
                results = await ctx2.deps.client.expand_context(results)
                return "\n\n".join(r.content for r in results)

            # Tool is registered via decorator above
            _ = gather_context

            prompt = (
                "Plan a focused approach for the main question.\n\n"
                f"Main question: {state.context.original_question}"
            )

            # Create agent dependencies
            agent_deps = deps_type(client=deps.client, context=state.context)  # type: ignore[call-arg]
            plan_result = await plan_agent.run(prompt, deps=agent_deps)
            state.context.sub_questions = list(plan_result.output.sub_questions)

            # State now contains the plan - emit state update and narrate
            if deps.agui_emitter:
                deps.agui_emitter.update_state(state)
                count = len(state.context.sub_questions)
                deps.agui_emitter.update_activity(
                    "planning",
                    {
                        "stepName": "plan",
                        "message": f"Created plan with {count} sub-questions",
                        "sub_questions": list(state.context.sub_questions),
                    },
                )
        finally:
            if deps.agui_emitter:
                deps.agui_emitter.finish_step()

    return plan


def create_search_node[AgentDepsT: GraphAgentDeps](
    model_config: ModelConfig,
    deps_type: type[AgentDepsT],
    with_step_wrapper: bool = True,
    success_message_format: str = "Answered: {sub_q}",
    handle_exceptions: bool = False,
    config: AppConfig = Config,
) -> Callable[[StepContext[Any, Any, str]], Awaitable[SearchAnswer]]:
    """Create a search_one node for any graph.

    Args:
        model_config: ModelConfig with provider, model, and settings
        deps_type: Type of dependencies for the agent
        with_step_wrapper: Whether to wrap with agui_emitter start/finish step
        success_message_format: Format string for success activity message
        handle_exceptions: Whether to handle exceptions with fallback answer
        config: AppConfig object (defaults to global Config)

    Returns:
        Async function that can be used as a graph step
    """

    async def search_one(ctx: StepContext[Any, Any, str], /) -> SearchAnswer:
        state: GraphState = ctx.state  # type: ignore[assignment]
        deps: GraphDeps = ctx.deps  # type: ignore[assignment]
        sub_q = ctx.inputs

        # Create unique step name from question text
        step_name = f"search: {sub_q}"

        if deps.agui_emitter and with_step_wrapper:
            deps.agui_emitter.start_step(step_name)

        try:
            # Create semaphore if not already provided
            if deps.semaphore is None:
                deps.semaphore = asyncio.Semaphore(state.max_concurrency)

            # Use semaphore to control concurrency
            async with deps.semaphore:
                return await _do_search(
                    state,
                    deps,
                    sub_q,
                    model_config,
                    deps_type,
                    success_message_format,
                    handle_exceptions,
                    config,
                )
        finally:
            if deps.agui_emitter and with_step_wrapper:
                deps.agui_emitter.finish_step()

    return search_one


async def _do_search[AgentDepsT: GraphAgentDeps](
    state: GraphState,
    deps: GraphDeps,
    sub_q: str,
    model_config: ModelConfig,
    deps_type: type[AgentDepsT],
    success_message_format: str,
    handle_exceptions: bool,
    config: AppConfig,
) -> SearchAnswer:
    """Internal search implementation."""
    if deps.agui_emitter:
        deps.agui_emitter.update_activity(
            "searching",
            {
                "stepName": "search_one",
                "message": f"Searching: {sub_q}",
                "query": sub_q,
            },
        )

    agent = Agent(
        model=get_model(model_config, config),
        output_type=ToolOutput(RawSearchAnswer, max_retries=3),
        instructions=SEARCH_AGENT_PROMPT,
        retries=3,
        deps_type=deps_type,
    )

    # Capture search filter for use in tool
    search_filter = state.search_filter

    @agent.tool
    async def search_and_answer(
        ctx2: RunContext[AgentDepsT], query: str, limit: int | None = None
    ) -> str:
        """Search the knowledge base for relevant documents.

        Returns results with chunk IDs and relevance scores.
        Reference results by their chunk_id in cited_chunks.
        """
        results = await ctx2.deps.client.search(
            query, limit=limit, filter=search_filter
        )
        results = await ctx2.deps.client.expand_context(results)
        # Store results for citation resolution
        ctx2.deps.search_results = results

        # Format with metadata for agent context
        parts = [r.format_for_agent() for r in results]

        if not parts:
            return f"No relevant information found in the knowledge base for: {query}"

        return "\n\n".join(parts)

    # Tool is registered via decorator above
    _ = search_and_answer

    agent_deps = deps_type(client=deps.client, context=state.context)  # type: ignore[call-arg]

    try:
        result = await agent.run(sub_q, deps=agent_deps)
        raw_answer = result.output
        if raw_answer:
            # Convert RawSearchAnswer to SearchAnswer with resolved citations
            answer = SearchAnswer.from_raw(raw_answer, agent_deps.search_results)
            state.context.add_qa_response(answer)
            # State updated with new answer - emit state update and narrate
            if deps.agui_emitter:
                deps.agui_emitter.update_state(state)
                # Format the success message
                if "{confidence" in success_message_format:
                    message = success_message_format.format(
                        sub_q=sub_q, confidence=answer.confidence
                    )
                else:
                    message = success_message_format.format(sub_q=sub_q)
                deps.agui_emitter.update_activity(
                    "searching",
                    {
                        "stepName": "search_one",
                        "message": message,
                        "query": sub_q,
                        "confidence": answer.confidence,
                    },
                )
            return answer
        # Return empty SearchAnswer if no result
        return SearchAnswer(query=sub_q, answer="", confidence=0.0)
    except Exception as e:
        if handle_exceptions:
            # Narrate the error
            if deps.agui_emitter:
                deps.agui_emitter.update_activity(
                    "searching",
                    {
                        "stepName": "search_one",
                        "message": f"Search failed: {e}",
                        "query": sub_q,
                        "error": str(e),
                    },
                )
            failure_answer = SearchAnswer(
                query=sub_q,
                answer=f"Search failed after retries: {str(e)}",
                confidence=0.0,
            )
            return failure_answer
        else:
            raise
