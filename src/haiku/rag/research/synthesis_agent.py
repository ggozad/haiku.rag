from pydantic import BaseModel, Field

from haiku.rag.research.base import BaseResearchAgent
from haiku.rag.research.prompts import SYNTHESIS_AGENT_PROMPT


class ResearchReport(BaseModel):
    """Final research report structure."""

    title: str = Field(description="Concise title for the research")
    executive_summary: str = Field(description="Brief overview of key findings")
    main_findings: list[str] = Field(
        description="Primary research findings with supporting evidence"
    )
    themes: dict[str, str] = Field(description="Major themes and their explanations")
    conclusions: list[str] = Field(description="Evidence-based conclusions")
    limitations: list[str] = Field(description="Limitations of the current research")
    recommendations: list[str] = Field(
        description="Actionable recommendations based on findings"
    )
    sources_summary: str = Field(
        description="Summary of sources used and their reliability"
    )


class SynthesisAgent(BaseResearchAgent[ResearchReport]):
    """Agent specialized in synthesizing research into comprehensive reports."""

    def __init__(self, provider: str, model: str) -> None:
        super().__init__(provider, model, output_type=ResearchReport)

    def get_system_prompt(self) -> str:
        return SYNTHESIS_AGENT_PROMPT

    def register_tools(self) -> None:
        """Register synthesis-specific tools."""
        # The agent will use its LLM capabilities directly for synthesis
        # The structured output will guide the report generation
        pass
