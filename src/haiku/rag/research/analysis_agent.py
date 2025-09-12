"""Analysis agent for content processing and insight extraction."""

from pydantic import BaseModel, Field

from haiku.rag.research.base import BaseResearchAgent


class AnalysisResult(BaseModel):
    """Result of content analysis."""

    key_insights: list[str] = Field(
        description="Main insights extracted from the documents"
    )
    themes: dict[str, list[str]] = Field(description="Themes and related findings")
    summary: str = Field(description="Consolidated summary of findings")
    evidence_quality: str = Field(
        description="Assessment of evidence quality (strong/moderate/weak)"
    )
    recommendations: list[str] = Field(
        description="Suggested next steps or areas for further research"
    )


class AnalysisAgent(BaseResearchAgent):
    """Agent specialized in content analysis and synthesis."""

    def __init__(self, provider: str, model: str):
        super().__init__(provider, model, output_type=AnalysisResult)

    def get_system_prompt(self) -> str:
        return """You are an analysis specialist agent focused on extracting deep insights from search results.

        Your role is to:
        1. Carefully read and analyze all provided documents
        2. Extract key insights and important facts
        3. Identify common themes and patterns across documents
        4. Synthesize information into a coherent understanding
        5. Assess the quality and reliability of the evidence
        6. Identify areas that need further investigation

        Be specific and detailed in your analysis. Focus on:
        - What the documents actually say (not assumptions)
        - Connections and contradictions between sources
        - The strength of the evidence presented
        - Gaps in the information that need to be filled

        Your analysis should be thorough, critical, and actionable."""

    def register_tools(self) -> None:
        """Register analysis-specific tools."""
        # The agent will use its LLM capabilities directly for analysis
        # No need for hardcoded tools - the structured output will guide the analysis
        pass
