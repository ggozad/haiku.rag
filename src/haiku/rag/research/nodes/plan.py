from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_graph import BaseNode, GraphRunContext

from haiku.rag.research.common import get_model, log
from haiku.rag.research.dependencies import ResearchDependencies
from haiku.rag.research.models import ResearchPlan, ResearchReport
from haiku.rag.research.nodes.search import SearchDispatchNode
from haiku.rag.research.prompts import PLAN_PROMPT
from haiku.rag.research.state import ResearchDeps, ResearchState


@dataclass
class PlanNode(BaseNode[ResearchState, ResearchDeps, ResearchReport]):
    provider: str
    model: str

    async def run(
        self, ctx: GraphRunContext[ResearchState, ResearchDeps]
    ) -> BaseNode[ResearchState, ResearchDeps, ResearchReport]:
        state = ctx.state
        deps = ctx.deps

        log(deps, state, "\n[bold cyan]📋 Creating research plan...[/bold cyan]")

        plan_agent = Agent(
            model=get_model(self.provider, self.model),
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
            "Plan a focused research approach for the main question.\n\n"
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

        log(deps, state, "\n[bold green]✅ Research Plan Created:[/bold green]")
        log(
            deps,
            state,
            f"   [bold]Main Question:[/bold] {state.context.original_question}",
        )
        log(deps, state, "   [bold]Sub-questions:[/bold]")
        for i, sq in enumerate(state.context.sub_questions, 1):
            log(deps, state, f"      {i}. {sq}")

        return SearchDispatchNode(self.provider, self.model)
