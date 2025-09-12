"""Synthesis agent for final research report generation."""

from pydantic import BaseModel, Field

from haiku.rag.research.base import BaseResearchAgent


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


class SynthesisAgent(BaseResearchAgent):
    """Agent specialized in synthesizing research into comprehensive reports."""

    def __init__(self, provider: str, model: str):
        super().__init__(provider, model, output_type=ResearchReport)

    def get_system_prompt(self) -> str:
        return """You are a synthesis specialist agent focused on creating comprehensive research reports.

        Your role is to:
        1. Synthesize all gathered information into a coherent narrative
        2. Present findings in a clear, structured format
        3. Draw evidence-based conclusions
        4. Acknowledge limitations and uncertainties
        5. Provide actionable recommendations
        6. Maintain academic rigor and objectivity

        Your report should be:
        - Comprehensive yet concise
        - Well-structured and easy to follow
        - Based solely on evidence from the research
        - Transparent about limitations
        - Professional and objective in tone

        Focus on creating a report that provides clear value to the reader by:
        - Answering the original research question thoroughly
        - Highlighting the most important findings
        - Explaining the implications of the research
        - Suggesting concrete next steps"""

    def register_tools(self) -> None:
        """Register synthesis-specific tools."""
        # The agent will use its LLM capabilities directly for synthesis
        # The structured output will guide the report generation
        pass
