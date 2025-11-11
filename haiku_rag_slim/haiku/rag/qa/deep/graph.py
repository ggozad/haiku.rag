from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.format_prompt import format_as_xml
from pydantic_ai.output import ToolOutput
from pydantic_graph.beta import Graph, GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_append

from haiku.rag.config import Config
from haiku.rag.config.models import AppConfig
from haiku.rag.graph_common import get_model
from haiku.rag.graph_common.models import ResearchPlan, SearchAnswer
from haiku.rag.graph_common.prompts import PLAN_PROMPT, SEARCH_AGENT_PROMPT
from haiku.rag.qa.deep.dependencies import DeepQADependencies
from haiku.rag.qa.deep.models import DeepQAAnswer, DeepQAEvaluation
from haiku.rag.qa.deep.prompts import (
    DECISION_PROMPT,
    SYNTHESIS_PROMPT,
    SYNTHESIS_PROMPT_WITH_CITATIONS,
)
from haiku.rag.qa.deep.state import DeepQADeps, DeepQAState


def build_deep_qa_graph(
    config: AppConfig = Config,
) -> Graph[DeepQAState, DeepQADeps, None, DeepQAAnswer]:
    """Build the Deep QA graph.

    Args:
        config: AppConfig object (uses config.qa for provider, model, and graph parameters)

    Returns:
        Configured Deep QA graph
    """
    provider = config.qa.provider
    model = config.qa.model
    g = GraphBuilder(
        state_type=DeepQAState,
        deps_type=DeepQADeps,
        output_type=DeepQAAnswer,
    )

    @g.step
    async def plan(ctx: StepContext[DeepQAState, DeepQADeps, None]) -> None:
        state = ctx.state
        deps = ctx.deps

        if deps.agui_emitter:
            deps.agui_emitter.start_step("plan")
            deps.agui_emitter.update_activity("planning", "Planning approach")

        try:
            plan_agent = Agent(
                model=get_model(provider, model),
                output_type=ResearchPlan,
                instructions=(
                    PLAN_PROMPT
                    + "\n\nUse the gather_context tool once on the main question before planning."
                ),
                retries=3,
                deps_type=DeepQADependencies,
            )

            @plan_agent.tool
            async def gather_context(
                ctx2: RunContext[DeepQADependencies], query: str, limit: int = 6
            ) -> str:
                results = await ctx2.deps.client.search(query, limit=limit)
                expanded = await ctx2.deps.client.expand_context(results)
                return "\n\n".join(chunk.content for chunk, _ in expanded)

            prompt = (
                "Plan a focused approach for the main question.\n\n"
                f"Main question: {state.context.original_question}"
            )

            agent_deps = DeepQADependencies(
                client=deps.client,
                context=state.context,
                console=None,
            )
            plan_result = await plan_agent.run(prompt, deps=agent_deps)
            state.context.sub_questions = list(plan_result.output.sub_questions)

            if deps.agui_emitter:
                deps.agui_emitter.update_state(state)
                count = len(state.context.sub_questions)
                deps.agui_emitter.update_activity(
                    "planning", f"Created plan with {count} sub-questions"
                )
        finally:
            if deps.agui_emitter:
                deps.agui_emitter.finish_step()

    @g.step
    async def search_one(
        ctx: StepContext[DeepQAState, DeepQADeps, str],
    ) -> SearchAnswer:
        state = ctx.state
        deps = ctx.deps
        sub_q = ctx.inputs

        # Create semaphore if not already provided
        if deps.semaphore is None:
            import asyncio

            deps.semaphore = asyncio.Semaphore(state.max_concurrency)

        # Use semaphore to control concurrency
        async with deps.semaphore:
            return await _do_search(state, deps, sub_q)

    async def _do_search(
        state: DeepQAState,
        deps: DeepQADeps,
        sub_q: str,
    ) -> SearchAnswer:
        if deps.agui_emitter:
            deps.agui_emitter.update_activity("searching", f"Searching: {sub_q}")

        agent = Agent(
            model=get_model(provider, model),
            output_type=ToolOutput(SearchAnswer, max_retries=3),
            instructions=SEARCH_AGENT_PROMPT,
            retries=3,
            deps_type=DeepQADependencies,
        )

        @agent.tool
        async def search_and_answer(
            ctx2: RunContext[DeepQADependencies], query: str, limit: int = 5
        ) -> str:
            search_results = await ctx2.deps.client.search(query, limit=limit)
            expanded = await ctx2.deps.client.expand_context(search_results)

            entries: list[dict[str, Any]] = [
                {
                    "text": chunk.content,
                    "score": score,
                    "document_uri": (chunk.document_title or chunk.document_uri or ""),
                }
                for chunk, score in expanded
            ]
            if not entries:
                return (
                    f"No relevant information found in the knowledge base for: {query}"
                )

            return format_as_xml(entries, root_tag="snippets")

        agent_deps = DeepQADependencies(
            client=deps.client,
            context=state.context,
            console=None,
        )
        try:
            result = await agent.run(sub_q, deps=agent_deps)
            answer = result.output
            if answer:
                state.context.add_qa_response(answer)
                if deps.agui_emitter:
                    deps.agui_emitter.update_state(state)
                    deps.agui_emitter.update_activity("searching", f"Answered: {sub_q}")
            return answer
        except Exception as e:
            failure_answer = SearchAnswer(
                query=sub_q,
                answer=f"Search failed after retries: {str(e)}",
                confidence=0.0,
            )
            return failure_answer

    @g.step
    async def get_batch(
        ctx: StepContext[DeepQAState, DeepQADeps, None | bool],
    ) -> list[str] | None:
        """Get all remaining questions for this iteration."""
        state = ctx.state

        if not state.context.sub_questions:
            return None

        # Take ALL remaining questions - max_concurrency controls parallel execution within .map()
        batch = list(state.context.sub_questions)
        state.context.sub_questions.clear()
        return batch

    @g.step
    async def decide(
        ctx: StepContext[DeepQAState, DeepQADeps, list[SearchAnswer]],
    ) -> bool:
        state = ctx.state
        deps = ctx.deps

        if deps.agui_emitter:
            deps.agui_emitter.start_step("decide")
            deps.agui_emitter.update_activity(
                "evaluating", "Evaluating information sufficiency"
            )

        try:
            agent = Agent(
                model=get_model(provider, model),
                output_type=DeepQAEvaluation,
                instructions=DECISION_PROMPT,
                retries=3,
                deps_type=DeepQADependencies,
            )

            context_data = {
                "original_question": state.context.original_question,
                "gathered_answers": [
                    {
                        "question": qa.query,
                        "answer": qa.answer,
                        "sources": qa.sources,
                    }
                    for qa in state.context.qa_responses
                ],
            }
            context_xml = format_as_xml(context_data, root_tag="gathered_information")

            prompt = (
                "Evaluate whether we have sufficient information to answer the question.\n\n"
                f"{context_xml}"
            )

            agent_deps = DeepQADependencies(
                client=deps.client,
                context=state.context,
                console=None,
            )
            result = await agent.run(prompt, deps=agent_deps)
            evaluation = result.output

            state.iterations += 1

            for new_q in evaluation.new_questions:
                if new_q not in state.context.sub_questions:
                    state.context.sub_questions.append(new_q)

            if deps.agui_emitter:
                deps.agui_emitter.update_state(state)
                status = "sufficient" if evaluation.is_sufficient else "insufficient"
                deps.agui_emitter.update_activity(
                    "evaluating",
                    f"Information {status} after {state.iterations} iteration(s)",
                )

            should_continue = (
                not evaluation.is_sufficient and state.iterations < state.max_iterations
            )

            return should_continue
        finally:
            if deps.agui_emitter:
                deps.agui_emitter.finish_step()

    @g.step
    async def synthesize(
        ctx: StepContext[DeepQAState, DeepQADeps, None | bool],
    ) -> DeepQAAnswer:
        state = ctx.state
        deps = ctx.deps

        if deps.agui_emitter:
            deps.agui_emitter.start_step("synthesize")
            deps.agui_emitter.update_activity(
                "synthesizing", "Synthesizing final answer"
            )

        try:
            prompt_template = (
                SYNTHESIS_PROMPT_WITH_CITATIONS
                if state.context.use_citations
                else SYNTHESIS_PROMPT
            )

            agent = Agent(
                model=get_model(provider, model),
                output_type=DeepQAAnswer,
                instructions=prompt_template,
                retries=3,
                deps_type=DeepQADependencies,
            )

            context_data = {
                "original_question": state.context.original_question,
                "sub_answers": [
                    {
                        "question": qa.query,
                        "answer": qa.answer,
                        "sources": qa.sources,
                    }
                    for qa in state.context.qa_responses
                ],
            }
            context_xml = format_as_xml(context_data, root_tag="gathered_information")

            prompt = f"Synthesize a comprehensive answer to the original question.\n\n{context_xml}"

            agent_deps = DeepQADependencies(
                client=deps.client,
                context=state.context,
                console=None,
            )
            result = await agent.run(prompt, deps=agent_deps)

            if deps.agui_emitter:
                deps.agui_emitter.update_activity("synthesizing", "Answer complete")

            return result.output
        finally:
            if deps.agui_emitter:
                deps.agui_emitter.finish_step()

    # Build the graph structure
    collect_answers = g.join(
        reduce_list_append,
        initial_factory=list[SearchAnswer],
    )

    g.add(
        g.edge_from(g.start_node).to(plan),
        g.edge_from(plan).to(get_batch),
    )

    # Branch based on whether we have questions
    g.add(
        g.edge_from(get_batch).to(
            g.decision()
            .branch(g.match(list).label("Has questions").map().to(search_one))
            .branch(g.match(type(None)).label("No questions").to(synthesize))
        ),
        g.edge_from(search_one).to(collect_answers),
        g.edge_from(collect_answers).to(decide),
    )

    # Branch based on decision
    g.add(
        g.edge_from(decide).to(
            g.decision()
            .branch(
                g.match(bool, matches=lambda x: x).label("Continue QA").to(get_batch)
            )
            .branch(
                g.match(bool, matches=lambda x: not x)
                .label("Done with QA")
                .to(synthesize)
            )
        ),
        g.edge_from(synthesize).to(g.end_node),
    )

    return g.build()
