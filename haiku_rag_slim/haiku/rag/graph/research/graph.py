from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.format_prompt import format_as_xml
from pydantic_ai.output import ToolOutput
from pydantic_graph.beta import Graph, GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_append

from haiku.rag.config import Config
from haiku.rag.config.models import AppConfig
from haiku.rag.graph.common import get_model
from haiku.rag.graph.common.models import ResearchPlan, SearchAnswer
from haiku.rag.graph.common.prompts import PLAN_PROMPT, SEARCH_AGENT_PROMPT
from haiku.rag.graph.research.common import (
    format_analysis_for_prompt,
    format_context_for_prompt,
)
from haiku.rag.graph.research.dependencies import ResearchDependencies
from haiku.rag.graph.research.models import (
    EvaluationResult,
    InsightAnalysis,
    ResearchReport,
)
from haiku.rag.graph.research.prompts import (
    DECISION_AGENT_PROMPT,
    INSIGHT_AGENT_PROMPT,
    SYNTHESIS_AGENT_PROMPT,
)
from haiku.rag.graph.research.state import ResearchDeps, ResearchState


def build_research_graph(
    config: AppConfig = Config,
) -> Graph[ResearchState, ResearchDeps, None, ResearchReport]:
    """Build the Research graph.

    Args:
        config: AppConfig object (uses config.research for provider, model, and graph parameters)

    Returns:
        Configured Research graph
    """
    provider = config.research.provider
    model = config.research.model
    g = GraphBuilder(
        state_type=ResearchState,
        deps_type=ResearchDeps,
        output_type=ResearchReport,
    )

    @g.step
    async def plan(ctx: StepContext[ResearchState, ResearchDeps, None]) -> None:
        state = ctx.state
        deps = ctx.deps

        if deps.agui_emitter:
            deps.agui_emitter.start_step("plan")
            deps.agui_emitter.update_activity("planning", "Creating research plan")

        try:
            plan_agent = Agent(
                model=get_model(provider, model),
                output_type=ResearchPlan,
                instructions=(
                    PLAN_PROMPT
                    + "\n\nUse the gather_context tool once on the main question before planning."
                ),
                retries=3,
                output_retries=3,
                deps_type=ResearchDependencies,
            )

            @plan_agent.tool
            async def gather_context(
                ctx2: RunContext[ResearchDependencies], query: str, limit: int = 6
            ) -> str:
                results = await ctx2.deps.client.search(query, limit=limit)
                expanded = await ctx2.deps.client.expand_context(results)
                return "\n\n".join(chunk.content for chunk, _ in expanded)

            prompt = (
                "Plan a focused approach for the main question.\n\n"
                f"Main question: {state.context.original_question}"
            )

            agent_deps = ResearchDependencies(
                client=deps.client,
                context=state.context,
            )
            plan_result = await plan_agent.run(prompt, deps=agent_deps)
            state.context.sub_questions = list(plan_result.output.sub_questions)

            # State now contains the plan - emit state update and narrate
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
        ctx: StepContext[ResearchState, ResearchDeps, str],
    ) -> SearchAnswer:
        state = ctx.state
        deps = ctx.deps
        sub_q = ctx.inputs

        if deps.agui_emitter:
            deps.agui_emitter.start_step("search_one")

        try:
            # Create semaphore if not already provided
            if deps.semaphore is None:
                import asyncio

                deps.semaphore = asyncio.Semaphore(state.max_concurrency)

            # Use semaphore to control concurrency
            async with deps.semaphore:
                return await _do_search(state, deps, sub_q)
        finally:
            if deps.agui_emitter:
                deps.agui_emitter.finish_step()

    async def _do_search(
        state: ResearchState,
        deps: ResearchDeps,
        sub_q: str,
    ) -> SearchAnswer:
        if deps.agui_emitter:
            deps.agui_emitter.update_activity("searching", f"Searching: {sub_q}")

        agent = Agent(
            model=get_model(provider, model),
            output_type=ToolOutput(SearchAnswer, max_retries=3),
            instructions=SEARCH_AGENT_PROMPT,
            retries=3,
            deps_type=ResearchDependencies,
        )

        @agent.tool
        async def search_and_answer(
            ctx2: RunContext[ResearchDependencies], query: str, limit: int = 5
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

        agent_deps = ResearchDependencies(
            client=deps.client,
            context=state.context,
        )
        try:
            result = await agent.run(sub_q, deps=agent_deps)
            answer = result.output
            if answer:
                state.context.add_qa_response(answer)
                # State updated with new answer - emit state update and narrate
                if deps.agui_emitter:
                    deps.agui_emitter.update_state(state)
                    deps.agui_emitter.update_activity(
                        "searching",
                        f"Found answer with {answer.confidence:.0%} confidence",
                    )
            return answer
        except Exception as e:
            # Narrate the error
            if deps.agui_emitter:
                deps.agui_emitter.update_activity("searching", f"Search failed: {e}")
            failure_answer = SearchAnswer(
                query=sub_q,
                answer=f"Search failed after retries: {str(e)}",
                confidence=0.0,
            )
            return failure_answer

    @g.step
    async def get_batch(
        ctx: StepContext[ResearchState, ResearchDeps, None | bool],
    ) -> list[str] | None:
        """Get all remaining questions for this iteration."""
        state = ctx.state

        if not state.context.sub_questions:
            return None

        # Take ALL remaining questions and process them in parallel
        batch = list(state.context.sub_questions)
        state.context.sub_questions.clear()
        return batch

    @g.step
    async def analyze_insights(
        ctx: StepContext[ResearchState, ResearchDeps, list[SearchAnswer]],
    ) -> None:
        state = ctx.state
        deps = ctx.deps

        if deps.agui_emitter:
            deps.agui_emitter.start_step("analyze_insights")
            deps.agui_emitter.update_activity(
                "analyzing", "Synthesizing insights and gaps"
            )

        try:
            agent = Agent(
                model=get_model(provider, model),
                output_type=InsightAnalysis,
                instructions=INSIGHT_AGENT_PROMPT,
                retries=3,
                output_retries=3,
                deps_type=ResearchDependencies,
            )

            context_xml = format_context_for_prompt(state.context)
            prompt = (
                "Review the latest research context and update the shared ledger of insights, gaps,"
                " and follow-up questions.\n\n"
                f"{context_xml}"
            )
            agent_deps = ResearchDependencies(
                client=deps.client,
                context=state.context,
            )
            result = await agent.run(prompt, deps=agent_deps)
            analysis: InsightAnalysis = result.output

            state.context.integrate_analysis(analysis)
            state.last_analysis = analysis

            # State updated with insights/gaps - emit state update and narrate
            if deps.agui_emitter:
                deps.agui_emitter.update_state(state)
                highlights = len(analysis.highlights) if analysis.highlights else 0
                gaps = len(analysis.gap_assessments) if analysis.gap_assessments else 0
                resolved = len(analysis.resolved_gaps) if analysis.resolved_gaps else 0
                parts = []
                if highlights:
                    parts.append(f"{highlights} insights")
                if gaps:
                    parts.append(f"{gaps} gaps")
                if resolved:
                    parts.append(f"{resolved} resolved")
                summary = ", ".join(parts) if parts else "No updates"
                deps.agui_emitter.update_activity("analyzing", f"Analysis: {summary}")
        finally:
            if deps.agui_emitter:
                deps.agui_emitter.finish_step()

    @g.step
    async def decide(ctx: StepContext[ResearchState, ResearchDeps, None]) -> bool:
        state = ctx.state
        deps = ctx.deps

        if deps.agui_emitter:
            deps.agui_emitter.start_step("decide")
            deps.agui_emitter.update_activity(
                "evaluating", "Evaluating research sufficiency"
            )

        try:
            agent = Agent(
                model=get_model(provider, model),
                output_type=EvaluationResult,
                instructions=DECISION_AGENT_PROMPT,
                retries=3,
                output_retries=3,
                deps_type=ResearchDependencies,
            )

            context_xml = format_context_for_prompt(state.context)
            analysis_xml = format_analysis_for_prompt(state.last_analysis)
            prompt_parts = [
                "Assess whether the research now answers the original question with adequate confidence.",
                context_xml,
                analysis_xml,
            ]
            if state.last_eval is not None:
                prev = state.last_eval
                prompt_parts.append(
                    "<previous_evaluation>"
                    f"<confidence>{prev.confidence_score:.2f}</confidence>"
                    f"<is_sufficient>{str(prev.is_sufficient).lower()}</is_sufficient>"
                    f"<reasoning>{prev.reasoning}</reasoning>"
                    "</previous_evaluation>"
                )
            prompt = "\n\n".join(part for part in prompt_parts if part)

            agent_deps = ResearchDependencies(
                client=deps.client,
                context=state.context,
            )
            decision_result = await agent.run(prompt, deps=agent_deps)
            output = decision_result.output

            state.last_eval = output
            state.iterations += 1

            for new_q in output.new_questions:
                if new_q not in state.context.sub_questions:
                    state.context.sub_questions.append(new_q)

            # State updated with evaluation - emit state update and narrate
            if deps.agui_emitter:
                deps.agui_emitter.update_state(state)
                sufficient = "Yes" if output.is_sufficient else "No"
                deps.agui_emitter.update_activity(
                    "evaluating",
                    f"Confidence: {output.confidence_score:.0%}, Sufficient: {sufficient}",
                )

            should_continue = (
                not output.is_sufficient
                or output.confidence_score < state.confidence_threshold
            ) and state.iterations < state.max_iterations

            return should_continue
        finally:
            if deps.agui_emitter:
                deps.agui_emitter.finish_step()

    @g.step
    async def synthesize(
        ctx: StepContext[ResearchState, ResearchDeps, None | bool],
    ) -> ResearchReport:
        state = ctx.state
        deps = ctx.deps

        if deps.agui_emitter:
            deps.agui_emitter.start_step("synthesize")
            deps.agui_emitter.update_activity(
                "synthesizing", "Generating final research report"
            )

        try:
            agent = Agent(
                model=get_model(provider, model),
                output_type=ResearchReport,
                instructions=SYNTHESIS_AGENT_PROMPT,
                retries=3,
                output_retries=3,
                deps_type=ResearchDependencies,
            )

            context_xml = format_context_for_prompt(state.context)
            prompt = (
                "Generate a comprehensive research report based on all gathered information.\n\n"
                f"{context_xml}\n\n"
                "Create a detailed report that synthesizes all findings into a coherent response."
            )
            agent_deps = ResearchDependencies(
                client=deps.client,
                context=state.context,
            )
            result = await agent.run(prompt, deps=agent_deps)
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
        g.edge_from(collect_answers).to(analyze_insights),
        g.edge_from(analyze_insights).to(decide),
    )

    # Branch based on decision
    g.add(
        g.edge_from(decide).to(
            g.decision()
            .branch(
                g.match(bool, matches=lambda x: x)
                .label("Continue research")
                .to(get_batch)
            )
            .branch(
                g.match(bool, matches=lambda x: not x)
                .label("Done researching")
                .to(synthesize)
            )
        ),
        g.edge_from(synthesize).to(g.end_node),
    )

    return g.build()
