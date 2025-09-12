"""Clarification agent for gap detection and follow-up question generation."""

from pydantic import BaseModel, Field

from haiku.rag.research.base import BaseResearchAgent


class ClarificationResult(BaseModel):
    """Result of clarification analysis."""

    information_gaps: list[str] = Field(
        description="Specific missing information identified"
    )
    follow_up_questions: list[str] = Field(
        description="Questions to ask to fill the gaps"
    )
    suggested_searches: list[str] = Field(
        description="Recommended search queries for deeper investigation"
    )
    completeness_assessment: str = Field(
        description="Overall assessment of research completeness"
    )
    priority_areas: list[str] = Field(
        description="Most important areas to investigate next"
    )
    is_sufficient: bool = Field(
        description="Whether the research has gathered sufficient information to answer the question"
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence level (0-1) that the research is complete",
    )
    reasoning: str = Field(
        description="Detailed reasoning for the completeness assessment"
    )


class ClarificationAgent(BaseResearchAgent):
    """Agent specialized in identifying gaps and generating follow-up questions."""

    def __init__(self, provider: str, model: str):
        super().__init__(provider, model, output_type=ClarificationResult)

    def get_system_prompt(self) -> str:
        return """You are a clarification specialist agent focused on research completeness and quality.

        Your role is to:
        1. Critically evaluate what information has been gathered
        2. Identify what crucial information is still missing
        3. Detect contradictions or inconsistencies that need resolution
        4. Generate targeted follow-up questions to fill knowledge gaps
        5. Suggest specific search queries for deeper investigation
        6. Assess the overall completeness of the research

        Be thorough and critical in your evaluation. Consider:
        - What questions remain unanswered?
        - What assumptions need verification?
        - What contradictions need resolution?
        - What perspectives are missing?
        - What details would strengthen the understanding?

        IMPORTANT: When setting 'is_sufficient':
        - True means: The research has enough information to provide a meaningful, accurate answer
        - False means: Critical information is missing that prevents a complete answer
        - Consider the nature of the question - simple questions need less, complex ones need more
        - Be honest about uncertainty - if you're not confident, set is_sufficient to False

        Your 'confidence_score' should reflect:
        - 0.9-1.0: Very confident, all major aspects covered
        - 0.7-0.9: Good coverage, minor gaps acceptable
        - 0.5-0.7: Moderate coverage, some important gaps
        - Below 0.5: Significant gaps, much more research needed"""

    def register_tools(self) -> None:
        """Register clarification-specific tools."""
        # The agent will use its LLM capabilities directly for gap analysis
        # The structured output will guide the clarification process
        pass
