from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_graph import BaseNode, GraphRunContext

from haiku.rag.research.common import format_context_for_prompt, get_model, log
from haiku.rag.research.dependencies import (
    ResearchDependencies,
)
from haiku.rag.research.models import EvaluationResult, ResearchReport
from haiku.rag.research.nodes.synthesize import SynthesizeNode
from haiku.rag.research.prompts import EVALUATION_AGENT_PROMPT
from haiku.rag.research.state import ResearchDeps, ResearchState


@dataclass
class EvaluateNode(BaseNode[ResearchState, ResearchDeps, ResearchReport]):
    provider: str
    model: str

    async def run(
        self, ctx: GraphRunContext[ResearchState, ResearchDeps]
    ) -> BaseNode[ResearchState, ResearchDeps, ResearchReport]:
        state = ctx.state
        deps = ctx.deps

        log(
            deps.console,
            "\n[bold cyan]📊 Analyzing and evaluating research progress...[/bold cyan]",
        )

        agent = Agent(
            model=get_model(self.provider, self.model),
            output_type=EvaluationResult,
            instructions=EVALUATION_AGENT_PROMPT,
            retries=3,
            deps_type=ResearchDependencies,
        )

        context_xml = format_context_for_prompt(state.context)
        prompt = (
            "Analyze gathered information and evaluate completeness for the original question.\n\n"
            f"{context_xml}"
        )
        agent_deps = ResearchDependencies(
            client=deps.client, context=state.context, console=deps.console
        )
        eval_result = await agent.run(prompt, deps=agent_deps)
        output = eval_result.output

        for insight in output.key_insights:
            state.context.add_insight(insight)
        for new_q in output.new_questions:
            if new_q not in state.sub_questions:
                state.sub_questions.append(new_q)

        state.last_eval = output
        state.iterations += 1

        if output.key_insights:
            log(deps.console, "   [bold]Key insights:[/bold]")
            for ins in output.key_insights:
                log(deps.console, f"   • {ins}")
        log(
            deps.console,
            f"   Confidence: [yellow]{output.confidence_score:.1%}[/yellow]",
        )
        status = "[green]Yes[/green]" if output.is_sufficient else "[red]No[/red]"
        log(deps.console, f"   Sufficient: {status}")

        from haiku.rag.research.nodes.search import SearchDispatchNode

        if (
            output.is_sufficient
            and output.confidence_score >= state.confidence_threshold
        ) or state.iterations >= state.max_iterations:
            log(deps.console, "\n[bold green]✅ Stopping research.[/bold green]")
            return SynthesizeNode(self.provider, self.model)

        return SearchDispatchNode(self.provider, self.model)
