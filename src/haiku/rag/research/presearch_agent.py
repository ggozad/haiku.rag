from pydantic_ai import RunContext
from pydantic_ai.run import AgentRunResult

from haiku.rag.research.base import BaseResearchAgent
from haiku.rag.research.dependencies import ResearchDependencies
from haiku.rag.research.prompts import PRESEARCH_AGENT_PROMPT


class PresearchSurveyAgent(BaseResearchAgent[str]):
    """Presearch agent that gathers verbatim context and summarizes it."""

    def __init__(self, provider: str, model: str) -> None:
        super().__init__(provider, model, str)

    async def run(
        self, prompt: str, deps: ResearchDependencies, **kwargs
    ) -> AgentRunResult[str]:
        return await super().run(prompt, deps, **kwargs)

    def get_system_prompt(self) -> str:
        return PRESEARCH_AGENT_PROMPT

    def register_tools(self) -> None:
        @self.agent.tool
        async def gather_context(
            ctx: RunContext[ResearchDependencies],
            query: str,
            limit: int = 6,
        ) -> str:
            """Return verbatim concatenation of relevant chunk texts."""
            query = query.replace('"', "")
            results = await ctx.deps.client.search(query, limit=limit)
            expanded = await ctx.deps.client.expand_context(results)
            return "\n\n".join(chunk.content for chunk, _ in expanded)
