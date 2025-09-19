from pydantic import BaseModel, Field
from pydantic_ai.run import AgentRunResult

from haiku.rag.research.base import BaseResearchAgent
from haiku.rag.research.dependencies import (
    ResearchDependencies,
    _format_context_for_prompt,
)
from haiku.rag.research.prompts import SYNTHESIS_AGENT_PROMPT


class ResearchReport(BaseModel):
    """Final research report structure."""

    title: str = Field(description="Concise title for the research")
    executive_summary: str = Field(description="Brief overview of key findings")
    main_findings: list[str] = Field(
        description="Primary research findings with supporting evidence"
    )
    conclusions: list[str] = Field(description="Evidence-based conclusions")
    limitations: list[str] = Field(
        description="Limitations of the current research", default=[]
    )
    recommendations: list[str] = Field(
        description="Actionable recommendations based on findings", default=[]
    )
    sources_summary: str = Field(
        description="Summary of sources used and their reliability"
    )


class SynthesisAgent(BaseResearchAgent[ResearchReport]):
    """Agent specialized in synthesizing research into comprehensive reports."""

    def __init__(self, provider: str, model: str) -> None:
        super().__init__(provider, model, output_type=ResearchReport)

    async def run(
        self, prompt: str, deps: ResearchDependencies, **kwargs
    ) -> AgentRunResult[ResearchReport]:
        console = deps.console
        if console:
            console.print(
                "\n[bold cyan]📝 Generating final research report...[/bold cyan]"
            )

        context_xml = _format_context_for_prompt(deps.context)
        synthesis_prompt = f"""Generate a comprehensive research report based on all gathered information.

{context_xml}

Create a detailed report that synthesizes all findings into a coherent response."""
        result = await super().run(synthesis_prompt, deps, **kwargs)
        if console:
            console.print("[bold green]✅ Research complete![/bold green]")

        return result

    def get_system_prompt(self) -> str:
        return SYNTHESIS_AGENT_PROMPT
