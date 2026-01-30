import asyncio
from typing import Literal, overload

from pydantic_ai import Agent, RunContext, format_as_xml
from pydantic_ai.output import ToolOutput
from pydantic_graph.beta import Graph, GraphBuilder, StepContext

from haiku.rag.agents.research.dependencies import ResearchContext, ResearchDependencies
from haiku.rag.agents.research.models import (
    Citation,
    ConversationalAnswer,
    IterativePlanResult,
    RawSearchAnswer,
    ResearchReport,
    SearchAnswer,
)
from haiku.rag.agents.research.prompts import (
    CONVERSATIONAL_SYNTHESIS_PROMPT,
    ITERATIVE_PLAN_PROMPT,
    ITERATIVE_PLAN_PROMPT_WITH_CONTEXT,
    SEARCH_PROMPT,
    SYNTHESIS_PROMPT,
)
from haiku.rag.agents.research.state import ResearchDeps, ResearchState
from haiku.rag.config import Config
from haiku.rag.config.models import AppConfig
from haiku.rag.utils import build_prompt, get_model


def format_context_for_prompt(
    context: ResearchContext,
    include_pending_questions: bool = True,
) -> str:
    """Format the research context as XML for prompts.

    Args:
        context: The research context to format.
        include_pending_questions: Whether to include pending sub-questions.
            Set to False for synthesis prompts where pending questions aren't relevant.
    """
    context_data: dict[str, object] = {}

    if context.session_context:
        context_data["background"] = context.session_context

    context_data["question"] = context.original_question

    if include_pending_questions and context.sub_questions:
        context_data["pending_questions"] = context.sub_questions

    if context.qa_responses:
        context_data["prior_answers"] = [
            {
                "question": qa.query,
                "answer": qa.answer,
                "confidence": qa.confidence,
                "source": qa.primary_source,
            }
            for qa in context.qa_responses
        ]

    return format_as_xml(context_data, root_tag="context")


async def _iterative_plan_logic(
    state: ResearchState,
    deps: ResearchDeps,
    config: AppConfig,
) -> IterativePlanResult:
    """Evaluate context and decide next question or mark complete."""
    has_prior_answers = bool(state.context.qa_responses)

    # If max iterations reached, skip LLM and mark complete
    if state.iterations >= state.max_iterations:
        return IterativePlanResult(
            is_complete=True,
            next_question=None,
            reasoning=f"Max iterations ({state.max_iterations}) reached.",
        )

    model_config = config.research.model

    if has_prior_answers:
        effective_prompt = build_prompt(ITERATIVE_PLAN_PROMPT_WITH_CONTEXT, config)
    else:
        effective_prompt = build_prompt(ITERATIVE_PLAN_PROMPT, config)

    plan_agent: Agent[ResearchDependencies, IterativePlanResult] = Agent(  # type: ignore[assignment]
        model=get_model(model_config, config),
        output_type=IterativePlanResult,
        instructions=effective_prompt,
        retries=3,
        output_retries=3,
        deps_type=ResearchDependencies,
    )

    # Build prompt based on current state
    if has_prior_answers:
        context_xml = format_context_for_prompt(state.context)
        prompt = (
            f"Review the gathered evidence and decide whether to continue or synthesize.\n\n"
            f"{context_xml}"
        )
    else:
        context_xml = format_context_for_prompt(state.context)
        prompt = f"Plan the research investigation.\n\n{context_xml}"

    agent_deps = ResearchDependencies(client=deps.client, context=state.context)
    result = await plan_agent.run(prompt, deps=agent_deps)

    # Enforce: if no prior answers, must have a next_question to investigate
    if not has_prior_answers:
        if result.output.is_complete or not result.output.next_question:
            return IterativePlanResult(
                is_complete=False,
                next_question=result.output.next_question
                or state.context.original_question,
                reasoning=result.output.reasoning,
            )

    return result.output


async def _search_one_step_logic(
    state: ResearchState,
    deps: ResearchDeps,
    config: AppConfig,
    search_prompt: str,
    sub_q: str,
) -> SearchAnswer:
    """Answer a single question using the knowledge base."""
    model_config = config.research.model

    if deps.semaphore is None:
        deps.semaphore = asyncio.Semaphore(state.max_concurrency)

    async with deps.semaphore:
        agent: Agent[ResearchDependencies, RawSearchAnswer] = Agent(  # type: ignore[assignment]
            model=get_model(model_config, config),
            output_type=ToolOutput(RawSearchAnswer, max_retries=3),
            instructions=search_prompt,
            retries=3,
            deps_type=ResearchDependencies,
        )

        search_filter = state.search_filter

        @agent.tool
        async def search_and_answer(
            ctx2: RunContext[ResearchDependencies],
            query: str,
            limit: int | None = None,
        ) -> str:
            """Search the knowledge base for relevant documents."""
            results = await ctx2.deps.client.search(
                query, limit=limit, filter=search_filter
            )
            results = await ctx2.deps.client.expand_context(results)
            ctx2.deps.search_results = results
            total = len(results)
            parts = [
                r.format_for_agent(rank=i + 1, total=total)
                for i, r in enumerate(results)
            ]
            if not parts:
                return f"No relevant information found for: {query}"
            return "\n\n".join(parts)

        agent_deps = ResearchDependencies(client=deps.client, context=state.context)

        result = await agent.run(sub_q, deps=agent_deps)
        raw_answer = result.output

        # Increment iterations after each search completes
        state.iterations += 1

        if raw_answer:
            answer = SearchAnswer.from_raw(raw_answer, agent_deps.search_results)
            state.context.add_qa_response(answer)
            return answer
        return SearchAnswer(query=sub_q, answer="", confidence=0.0)


@overload
def build_research_graph(
    config: AppConfig = ...,
    output_mode: Literal["report"] = ...,
) -> Graph[ResearchState, ResearchDeps, None, ResearchReport]: ...


@overload
def build_research_graph(
    config: AppConfig = ...,
    output_mode: Literal["conversational"] = ...,
) -> Graph[ResearchState, ResearchDeps, None, ConversationalAnswer]: ...


def build_research_graph(
    config: AppConfig = Config,
    output_mode: Literal["report", "conversational"] = "report",
) -> Graph[ResearchState, ResearchDeps, None, ResearchReport | ConversationalAnswer]:
    """Build the iterative research graph.

    Args:
        config: AppConfig object (uses config.research for provider, model, and graph parameters)
        output_mode: Output format - "report" for ResearchReport, "conversational" for ConversationalAnswer

    Returns:
        Configured research graph with iterative planning
    """
    model_config = config.research.model

    search_prompt = build_prompt(SEARCH_PROMPT, config)

    if output_mode == "report":
        synthesis_prompt = build_prompt(
            config.prompts.synthesis or SYNTHESIS_PROMPT, config
        )
    else:
        synthesis_prompt = build_prompt(CONVERSATIONAL_SYNTHESIS_PROMPT, config)

    g = GraphBuilder(
        state_type=ResearchState,
        deps_type=ResearchDeps,
        output_type=ResearchReport if output_mode == "report" else ConversationalAnswer,
    )

    @g.step
    async def plan_next(
        ctx: StepContext[ResearchState, ResearchDeps, None | SearchAnswer],
    ) -> IterativePlanResult:
        """Evaluate context and decide next question or complete."""
        return await _iterative_plan_logic(ctx.state, ctx.deps, config)

    @g.step
    async def search_one(
        ctx: StepContext[ResearchState, ResearchDeps, str],
    ) -> SearchAnswer:
        """Answer a single question using the knowledge base."""
        try:
            return await _search_one_step_logic(
                ctx.state, ctx.deps, config, search_prompt, ctx.inputs
            )
        except Exception as e:
            return SearchAnswer(
                query=ctx.inputs,
                answer=f"Search failed: {str(e)}",
                confidence=0.0,
            )

    if output_mode == "report":

        @g.step
        async def synthesize(
            ctx: StepContext[ResearchState, ResearchDeps, IterativePlanResult],
        ) -> ResearchReport:
            """Generate final research report."""
            state = ctx.state
            deps = ctx.deps

            agent: Agent[ResearchDependencies, ResearchReport] = Agent(  # type: ignore[assignment]
                model=get_model(model_config, config),
                output_type=ResearchReport,
                instructions=synthesis_prompt,
                retries=3,
                output_retries=3,
                deps_type=ResearchDependencies,
            )

            context_xml = format_context_for_prompt(
                state.context, include_pending_questions=False
            )
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

    else:

        @g.step
        async def synthesize(
            ctx: StepContext[ResearchState, ResearchDeps, IterativePlanResult],
        ) -> ConversationalAnswer:
            """Generate conversational answer from gathered evidence."""
            state = ctx.state
            deps = ctx.deps

            agent: Agent[ResearchDependencies, ConversationalAnswer] = Agent(  # type: ignore[assignment]
                model=get_model(model_config, config),
                output_type=ConversationalAnswer,
                instructions=synthesis_prompt,
                retries=3,
                output_retries=3,
                deps_type=ResearchDependencies,
            )

            context_xml = format_context_for_prompt(
                state.context, include_pending_questions=False
            )
            prompt = (
                f"Answer the question based on the gathered evidence.\n\n{context_xml}"
            )
            agent_deps = ResearchDependencies(
                client=deps.client,
                context=state.context,
            )
            result = await agent.run(prompt, deps=agent_deps)

            # Collect unique citations from qa_responses (dedupe by chunk_id)
            seen_chunks: set[str] = set()
            unique_citations: list[Citation] = []
            for qa in state.context.qa_responses:
                for c in qa.citations:
                    if c.chunk_id not in seen_chunks:
                        seen_chunks.add(c.chunk_id)
                        unique_citations.append(c)

            return ConversationalAnswer(
                answer=result.output.answer,
                citations=unique_citations,
                confidence=result.output.confidence,
            )

    # Build graph edges: iterative loop
    #
    # START -> plan_next -> [decision]
    #                          |
    #          [is_complete or max_iterations] -> synthesize -> END
    #                          |
    #          [has next_question] -> search_one -> plan_next (loop)

    def extract_question(
        ctx: StepContext[ResearchState, ResearchDeps, IterativePlanResult],
    ) -> str:
        """Extract next_question from IterativePlanResult."""
        return ctx.inputs.next_question or ""

    g.add(
        g.edge_from(g.start_node).to(plan_next),
        g.edge_from(plan_next).to(
            g.decision()
            .branch(
                g.match(
                    IterativePlanResult,
                    matches=lambda r: not r.is_complete and r.next_question is not None,
                )
                .label("Continue research")
                .transform(extract_question)
                .to(search_one)
            )
            .branch(
                g.match(IterativePlanResult).label("Done researching").to(synthesize)
            )
        ),
        g.edge_from(search_one).to(plan_next),
        g.edge_from(synthesize).to(g.end_node),
    )

    return g.build()
