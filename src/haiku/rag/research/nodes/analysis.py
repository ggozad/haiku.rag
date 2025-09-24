from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_graph import BaseNode, GraphRunContext

from haiku.rag.research.common import (
    format_analysis_for_prompt,
    format_context_for_prompt,
    get_model,
    log,
)
from haiku.rag.research.dependencies import ResearchDependencies
from haiku.rag.research.models import EvaluationResult, InsightAnalysis, ResearchReport
from haiku.rag.research.nodes.synthesize import SynthesizeNode
from haiku.rag.research.prompts import DECISION_AGENT_PROMPT, INSIGHT_AGENT_PROMPT
from haiku.rag.research.state import ResearchDeps, ResearchState


@dataclass
class AnalyzeInsightsNode(BaseNode[ResearchState, ResearchDeps, ResearchReport]):
    provider: str
    model: str

    async def run(
        self, ctx: GraphRunContext[ResearchState, ResearchDeps]
    ) -> BaseNode[ResearchState, ResearchDeps, ResearchReport]:
        state = ctx.state
        deps = ctx.deps

        log(
            deps,
            state,
            "\n[bold cyan]🧭 Synthesizing new insights and gap status...[/bold cyan]",
        )

        agent = Agent(
            model=get_model(self.provider, self.model),
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

        return DecisionNode(self.provider, self.model)


@dataclass
class DecisionNode(BaseNode[ResearchState, ResearchDeps, ResearchReport]):
    provider: str
    model: str

    async def run(
        self, ctx: GraphRunContext[ResearchState, ResearchDeps]
    ) -> BaseNode[ResearchState, ResearchDeps, ResearchReport]:
        state = ctx.state
        deps = ctx.deps

        log(
            deps,
            state,
            "\n[bold cyan]📊 Evaluating research sufficiency...[/bold cyan]",
        )

        agent = Agent(
            model=get_model(self.provider, self.model),
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

        from haiku.rag.research.nodes.search import SearchDispatchNode

        if (
            output.is_sufficient
            and output.confidence_score >= state.confidence_threshold
        ) or state.iterations >= state.max_iterations:
            log(deps, state, "\n[bold green]✅ Stopping research.[/bold green]")
            return SynthesizeNode(self.provider, self.model)

        return SearchDispatchNode(self.provider, self.model)
