from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.format_prompt import format_as_xml
from pydantic_ai.output import ToolOutput
from pydantic_graph.beta import Graph, GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_append

from haiku.rag.graph_common import get_model, log
from haiku.rag.graph_common.models import ResearchPlan, SearchAnswer
from haiku.rag.graph_common.prompts import PLAN_PROMPT, SEARCH_AGENT_PROMPT
from haiku.rag.research.common import (
    format_analysis_for_prompt,
    format_context_for_prompt,
)
from haiku.rag.research.dependencies import ResearchDependencies
from haiku.rag.research.models import (
    EvaluationResult,
    InsightAnalysis,
    ResearchReport,
)
from haiku.rag.research.prompts import (
    DECISION_AGENT_PROMPT,
    INSIGHT_AGENT_PROMPT,
    SYNTHESIS_AGENT_PROMPT,
)
from haiku.rag.research.state import ResearchDeps, ResearchState


def build_research_graph(
    provider: str, model: str
) -> Graph[ResearchState, ResearchDeps, None, ResearchReport]:
    g = GraphBuilder(
        state_type=ResearchState,
        deps_type=ResearchDeps,
        output_type=ResearchReport,
    )

    @g.step
    async def plan(ctx: StepContext[ResearchState, ResearchDeps, None]) -> None:
        state = ctx.state
        deps = ctx.deps

        log(deps, state, "\n[bold cyan]📋 Creating research plan...[/bold cyan]")

        plan_agent = Agent(
            model=get_model(provider, model),
            output_type=ResearchPlan,
            instructions=(
                PLAN_PROMPT
                + "\n\nUse the gather_context tool once on the main question before planning."
            ),
            retries=3,
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
            console=deps.console,
            stream=deps.stream,
        )
        plan_result = await plan_agent.run(prompt, deps=agent_deps)
        state.context.sub_questions = list(plan_result.output.sub_questions)

        log(deps, state, "\n[bold green]✅ Plan Created:[/bold green]")
        log(
            deps,
            state,
            f"   [bold]Main Question:[/bold] {state.context.original_question}",
        )
        log(deps, state, "   [bold]Sub-questions:[/bold]")
        for i, sq in enumerate(state.context.sub_questions, 1):
            log(deps, state, f"      {i}. {sq}")

    @g.step
    async def search_one(
        ctx: StepContext[ResearchState, ResearchDeps, str],
    ) -> SearchAnswer:
        state = ctx.state
        deps = ctx.deps
        sub_q = ctx.inputs

        log(
            deps,
            state,
            f"\n[bold cyan]🔍 Searching & Answering:[/bold cyan] {sub_q}",
        )

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
            console=deps.console,
            stream=deps.stream,
        )
        try:
            result = await agent.run(sub_q, deps=agent_deps)
            answer = result.output
            if answer:
                state.context.add_qa_response(answer)
                preview = answer.answer[:150] + (
                    "…" if len(answer.answer) > 150 else ""
                )
                log(deps, state, f"   [green]✓[/green] {preview}")
            return answer
        except Exception as e:
            log(deps, state, f"[red]Search failed:[/red] {e}")
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

        log(
            deps,
            state,
            "\n[bold cyan]🧭 Synthesizing new insights and gap status...[/bold cyan]",
        )

        agent = Agent(
            model=get_model(provider, model),
            output_type=InsightAnalysis,
            instructions=INSIGHT_AGENT_PROMPT,
            retries=3,
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
            console=deps.console,
            stream=deps.stream,
        )
        result = await agent.run(prompt, deps=agent_deps)
        analysis: InsightAnalysis = result.output

        state.context.integrate_analysis(analysis)
        state.last_analysis = analysis

        if analysis.commentary:
            log(deps, state, f"   Summary: {analysis.commentary}")
        if analysis.highlights:
            log(deps, state, "   [bold]Updated insights:[/bold]")
            for insight in analysis.highlights:
                label = insight.status.value
                log(
                    deps,
                    state,
                    f"   • ({label}) {insight.summary}",
                )
        if analysis.gap_assessments:
            log(deps, state, "   [bold yellow]Gap updates:[/bold yellow]")
            for gap in analysis.gap_assessments:
                status = "resolved" if gap.resolved else "open"
                severity = gap.severity.value
                log(
                    deps,
                    state,
                    f"   • ({severity}/{status}) {gap.description}",
                )
        if analysis.resolved_gaps:
            log(deps, state, "   [green]Resolved gaps:[/green]")
            for resolved in analysis.resolved_gaps:
                log(deps, state, f"   • {resolved}")
        if analysis.new_questions:
            log(deps, state, "   [cyan]Proposed follow-ups:[/cyan]")
            for question in analysis.new_questions:
                log(deps, state, f"   • {question}")

    @g.step
    async def decide(ctx: StepContext[ResearchState, ResearchDeps, None]) -> bool:
        state = ctx.state
        deps = ctx.deps

        log(
            deps,
            state,
            "\n[bold cyan]📊 Evaluating research sufficiency...[/bold cyan]",
        )

        agent = Agent(
            model=get_model(provider, model),
            output_type=EvaluationResult,
            instructions=DECISION_AGENT_PROMPT,
            retries=3,
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
            console=deps.console,
            stream=deps.stream,
        )
        decision_result = await agent.run(prompt, deps=agent_deps)
        output = decision_result.output

        state.last_eval = output
        state.iterations += 1

        for new_q in output.new_questions:
            if new_q not in state.context.sub_questions:
                state.context.sub_questions.append(new_q)

        if output.key_insights:
            log(deps, state, "   [bold]Key insights:[/bold]")
            for insight in output.key_insights:
                log(deps, state, f"   • {insight}")

        if output.gaps:
            log(deps, state, "   [bold yellow]Remaining gaps:[/bold yellow]")
            for gap in output.gaps:
                log(deps, state, f"   • {gap}")

        log(
            deps,
            state,
            f"   Confidence: [yellow]{output.confidence_score:.1%}[/yellow]",
        )
        status = "[green]Yes[/green]" if output.is_sufficient else "[red]No[/red]"
        log(deps, state, f"   Sufficient: {status}")

        should_continue = (
            not output.is_sufficient
            or output.confidence_score < state.confidence_threshold
        ) and state.iterations < state.max_iterations

        if not should_continue:
            log(deps, state, "\n[bold green]✅ Stopping research.[/bold green]")

        return should_continue

    @g.step
    async def synthesize(
        ctx: StepContext[ResearchState, ResearchDeps, None | bool],
    ) -> ResearchReport:
        state = ctx.state
        deps = ctx.deps

        log(
            deps,
            state,
            "\n[bold cyan]📝 Generating final research report...[/bold cyan]",
        )

        agent = Agent(
            model=get_model(provider, model),
            output_type=ResearchReport,
            instructions=SYNTHESIS_AGENT_PROMPT,
            retries=3,
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
            console=deps.console,
            stream=deps.stream,
        )
        result = await agent.run(prompt, deps=agent_deps)

        log(deps, state, "[bold green]✅ Research complete![/bold green]")
        return result.output

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
