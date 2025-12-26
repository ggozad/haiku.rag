import asyncio
from typing import Literal
from uuid import uuid4

from pydantic_ai import Agent, RunContext, format_as_xml
from pydantic_ai.output import ToolOutput
from pydantic_graph.beta import Graph, GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_append

from haiku.rag.config import Config
from haiku.rag.config.models import AppConfig
from haiku.rag.graph.agui.emitter import (
    emit_text_message_end,
    emit_text_message_start,
    emit_tool_call_args,
    emit_tool_call_end,
    emit_tool_call_start,
)
from haiku.rag.graph.research.dependencies import ResearchContext, ResearchDependencies
from haiku.rag.graph.research.models import (
    EvaluationResult,
    RawSearchAnswer,
    ResearchPlan,
    ResearchReport,
    SearchAnswer,
)
from haiku.rag.graph.research.prompts import (
    DECISION_PROMPT,
    PLAN_PROMPT,
    SEARCH_PROMPT,
    SYNTHESIS_PROMPT,
)
from haiku.rag.graph.research.state import ResearchDeps, ResearchState
from haiku.rag.utils import build_prompt, get_model


def format_context_for_prompt(context: ResearchContext) -> str:
    """Format the research context as XML for inclusion in prompts."""
    context_data = {
        "original_question": context.original_question,
        "unanswered_questions": context.sub_questions,
        "qa_responses": [
            {
                "question": qa.query,
                "answer": qa.answer,
                "confidence": qa.confidence,
                "sources": [
                    {
                        "document_uri": c.document_uri,
                        "document_title": c.document_title,
                        "page_numbers": c.page_numbers,
                        "headings": c.headings,
                    }
                    for c in qa.citations
                ],
            }
            for qa in context.qa_responses
        ],
    }
    return format_as_xml(context_data, root_tag="research_context")


def build_research_graph(
    config: AppConfig = Config,
    include_plan: bool = True,
    interactive: bool = False,
) -> Graph[ResearchState, ResearchDeps, None, ResearchReport]:
    """Build the Research graph.

    Args:
        config: AppConfig object (uses config.research for provider, model, and graph parameters)
        include_plan: Whether to include the planning step (False for execute-only mode)
        interactive: Whether to include human decision nodes for HIL

    Returns:
        Configured Research graph
    """
    model_config = config.research.model

    # Build prompts with system_context if configured
    plan_prompt = build_prompt(
        PLAN_PROMPT
        + "\n\nUse the gather_context tool once on the main question before planning.",
        config,
    )
    search_prompt = build_prompt(SEARCH_PROMPT, config)
    decision_prompt = build_prompt(DECISION_PROMPT, config)
    synthesis_prompt = build_prompt(
        config.prompts.synthesis or SYNTHESIS_PROMPT, config
    )
    g = GraphBuilder(
        state_type=ResearchState,
        deps_type=ResearchDeps,
        output_type=ResearchReport,
    )

    @g.step
    async def plan(ctx: StepContext[ResearchState, ResearchDeps, None]) -> None:
        """Create research plan with sub-questions."""
        state = ctx.state
        deps = ctx.deps

        if deps.agui_emitter:
            deps.agui_emitter.start_step("plan")
            deps.agui_emitter.update_activity(
                "planning", {"stepName": "plan", "message": "Creating research plan"}
            )

        try:
            plan_agent = Agent(
                model=get_model(model_config, config),
                output_type=ResearchPlan,
                instructions=plan_prompt,
                retries=3,
                output_retries=3,
                deps_type=ResearchDependencies,
            )

            search_filter = state.search_filter

            @plan_agent.tool
            async def gather_context(
                ctx2: RunContext[ResearchDependencies],
                query: str,
                limit: int | None = None,
            ) -> str:
                results = await ctx2.deps.client.search(
                    query, limit=limit, filter=search_filter
                )
                results = await ctx2.deps.client.expand_context(results)
                return "\n\n".join(r.content for r in results)

            _ = gather_context

            prompt = (
                "Plan a focused approach for the main question.\n\n"
                f"Main question: {state.context.original_question}"
            )

            agent_deps = ResearchDependencies(client=deps.client, context=state.context)
            plan_result = await plan_agent.run(prompt, deps=agent_deps)
            state.context.sub_questions = list(plan_result.output.sub_questions)

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

    @g.step
    async def search_one(
        ctx: StepContext[ResearchState, ResearchDeps, str],
    ) -> SearchAnswer:
        """Answer a single sub-question using the knowledge base."""
        state = ctx.state
        deps = ctx.deps
        sub_q = ctx.inputs
        step_name = f"search: {sub_q}"

        if deps.agui_emitter:
            deps.agui_emitter.start_step(step_name)

        try:
            if deps.semaphore is None:
                deps.semaphore = asyncio.Semaphore(state.max_concurrency)

            async with deps.semaphore:
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
                    parts = [r.format_for_agent() for r in results]
                    if not parts:
                        return f"No relevant information found for: {query}"
                    return "\n\n".join(parts)

                _ = search_and_answer

                agent_deps = ResearchDependencies(
                    client=deps.client, context=state.context
                )

                try:
                    result = await agent.run(sub_q, deps=agent_deps)
                    raw_answer = result.output
                    if raw_answer:
                        answer = SearchAnswer.from_raw(
                            raw_answer, agent_deps.search_results
                        )
                        state.context.add_qa_response(answer)
                        if deps.agui_emitter:
                            deps.agui_emitter.update_state(state)
                            deps.agui_emitter.update_activity(
                                "searching",
                                {
                                    "stepName": "search_one",
                                    "message": f"Found answer with {answer.confidence:.0%} confidence",
                                    "query": sub_q,
                                    "confidence": answer.confidence,
                                },
                            )
                        return answer
                    return SearchAnswer(query=sub_q, answer="", confidence=0.0)
                except Exception as e:
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
                    return SearchAnswer(
                        query=sub_q,
                        answer=f"Search failed: {str(e)}",
                        confidence=0.0,
                    )
        finally:
            if deps.agui_emitter:
                deps.agui_emitter.finish_step()

    @g.step
    async def get_batch(
        ctx: StepContext[ResearchState, ResearchDeps, None | bool | str],
    ) -> list[str] | None:
        """Get all remaining questions for this iteration."""
        state = ctx.state

        if not state.context.sub_questions:
            return None

        batch = list(state.context.sub_questions)
        state.context.sub_questions.clear()
        return batch

    @g.step
    async def decide(
        ctx: StepContext[ResearchState, ResearchDeps, list[SearchAnswer]],
    ) -> bool:
        """Evaluate research sufficiency and decide whether to continue."""
        state = ctx.state
        deps = ctx.deps

        if deps.agui_emitter:
            deps.agui_emitter.start_step("decide")
            deps.agui_emitter.update_activity(
                "evaluating", {"message": "Evaluating research sufficiency"}
            )

        try:
            agent = Agent(
                model=get_model(model_config, config),
                output_type=EvaluationResult,
                instructions=decision_prompt,
                retries=3,
                output_retries=3,
                deps_type=ResearchDependencies,
            )

            context_xml = format_context_for_prompt(state.context)
            prompt_parts = [
                "Assess whether the research now answers the original question with adequate confidence.",
                context_xml,
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

            # Get already-answered questions to avoid duplicates
            answered_queries = {qa.query.lower() for qa in state.context.qa_responses}

            for new_q in output.new_questions:
                # Skip if already in pending or already answered
                if new_q in state.context.sub_questions:
                    continue
                if new_q.lower() in answered_queries:
                    continue
                state.context.sub_questions.append(new_q)

            if deps.agui_emitter:
                deps.agui_emitter.update_state(state)
                sufficient = "Yes" if output.is_sufficient else "No"
                deps.agui_emitter.update_activity(
                    "evaluating",
                    {
                        "stepName": "decide",
                        "message": f"Confidence: {output.confidence_score:.0%}, Sufficient: {sufficient}",
                        "confidence": output.confidence_score,
                        "is_sufficient": output.is_sufficient,
                    },
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
    async def human_decide(
        ctx: StepContext[ResearchState, ResearchDeps, list[SearchAnswer] | None | bool],
    ) -> Literal["search", "synthesize"]:
        """Wait for human decision on whether to continue searching or synthesize."""
        state = ctx.state
        deps = ctx.deps

        if deps.agui_emitter:
            deps.agui_emitter.start_step("human_decide")
            deps.agui_emitter.update_state(state)

        try:
            # Emit tool call for human input wrapped in a message context
            # This makes the tool call appear as if emitted by the LLM
            message_id = str(uuid4())
            tool_call_id = str(uuid4())

            if deps.agui_emitter:
                # Start an assistant message to contain the tool call
                deps.agui_emitter.emit(emit_text_message_start(message_id))
                # Emit tool call with parent message reference
                deps.agui_emitter.emit(
                    emit_tool_call_start(tool_call_id, "human_decision", message_id)
                )
                # Include full state for display
                qa_responses = [
                    {
                        "query": qa.query,
                        "answer": qa.answer,
                        "confidence": qa.confidence,
                        "citations_count": len(qa.citations),
                    }
                    for qa in state.context.qa_responses
                ]
                deps.agui_emitter.emit(
                    emit_tool_call_args(
                        tool_call_id,
                        {
                            "original_question": state.context.original_question,
                            "sub_questions": list(state.context.sub_questions),
                            "qa_responses": qa_responses,
                            "iterations": state.iterations,
                        },
                    )
                )
                deps.agui_emitter.emit(emit_tool_call_end(tool_call_id))
                # End the message after tool call
                deps.agui_emitter.emit(emit_text_message_end(message_id))

            # Wait for human input
            if deps.human_input_queue is None:
                raise RuntimeError("human_input_queue is required for interactive mode")

            decision = await deps.human_input_queue.get()

            # Process decision
            if decision.action == "modify_questions" and decision.questions:
                state.context.sub_questions = list(decision.questions)
            elif decision.action == "add_questions" and decision.questions:
                state.context.sub_questions.extend(decision.questions)

            if deps.agui_emitter:
                deps.agui_emitter.update_state(state)

            if decision.action in ("search", "modify_questions", "add_questions"):
                return "search"
            else:
                return "synthesize"
        finally:
            if deps.agui_emitter:
                deps.agui_emitter.finish_step()

    @g.step
    async def synthesize(
        ctx: StepContext[ResearchState, ResearchDeps, None | bool | str],
    ) -> ResearchReport:
        """Generate final research report."""
        state = ctx.state
        deps = ctx.deps

        if deps.agui_emitter:
            deps.agui_emitter.start_step("synthesize")
            deps.agui_emitter.update_activity(
                "synthesizing", {"message": "Generating final research report"}
            )

        try:
            agent = Agent(
                model=get_model(model_config, config),
                output_type=ResearchReport,
                instructions=synthesis_prompt,
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

    if interactive:
        # Interactive mode: human decides after plan and after evaluation
        if include_plan:
            g.add(
                g.edge_from(g.start_node).to(plan),
                g.edge_from(plan).to(human_decide),
            )
        else:
            g.add(g.edge_from(g.start_node).to(human_decide))

        g.add(
            g.edge_from(human_decide).to(
                g.decision()
                .branch(
                    g.match(str, matches=lambda x: x == "search")
                    .label("Search")
                    .to(get_batch)
                )
                .branch(
                    g.match(str, matches=lambda x: x == "synthesize")
                    .label("Synthesize")
                    .to(synthesize)
                )
            ),
            g.edge_from(get_batch).to(
                g.decision()
                .branch(g.match(list).label("Has questions").map().to(search_one))
                .branch(g.match(type(None)).label("No questions").to(human_decide))
            ),
            g.edge_from(search_one).to(collect_answers),
            # After search, evaluate to suggest new questions, then human decides
            g.edge_from(collect_answers).to(decide),
            g.edge_from(decide).to(human_decide),
            g.edge_from(synthesize).to(g.end_node),
        )
    else:
        # Non-interactive mode: automatic decision based on confidence/iterations
        if include_plan:
            g.add(
                g.edge_from(g.start_node).to(plan),
                g.edge_from(plan).to(get_batch),
            )
        else:
            g.add(g.edge_from(g.start_node).to(get_batch))

        g.add(
            g.edge_from(get_batch).to(
                g.decision()
                .branch(g.match(list).label("Has questions").map().to(search_one))
                .branch(g.match(type(None)).label("No questions").to(synthesize))
            ),
            g.edge_from(search_one).to(collect_answers),
            g.edge_from(collect_answers).to(decide),
        )

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
