from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_graph import BaseNode, End, GraphRunContext

from haiku.rag.research.common import format_context_for_prompt, get_model, log
from haiku.rag.research.dependencies import (
    ResearchDependencies,
)
from haiku.rag.research.models import ResearchReport
from haiku.rag.research.prompts import SYNTHESIS_AGENT_PROMPT
from haiku.rag.research.state import ResearchDeps, ResearchState


@dataclass
class SynthesizeNode(BaseNode[ResearchState, ResearchDeps, ResearchReport]):
    provider: str
    model: str

    async def run(
        self, ctx: GraphRunContext[ResearchState, ResearchDeps]
    ) -> End[ResearchReport]:
        state = ctx.state
        deps = ctx.deps

        log(
            deps,
            state,
            "\n[bold cyan]📝 Generating final research report...[/bold cyan]",
        )

        agent = Agent(
            model=get_model(self.provider, self.model),
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
        return End(result.output)
