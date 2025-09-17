from pydantic import BaseModel, Field

from haiku.rag.research.base import BaseResearchAgent
from haiku.rag.research.prompts import EVALUATION_AGENT_PROMPT


class EvaluationResult(BaseModel):
    """Result of analysis and evaluation."""

    key_insights: list[str] = Field(
        description="Main insights extracted from the research so far"
    )
    new_questions: list[str] = Field(
        description="New sub-questions to add to the research (max 3)", max_length=3
    )
    confidence_score: float = Field(
        description="Confidence level in the completeness of research (0-1)",
        ge=0.0,
        le=1.0,
    )
    is_sufficient: bool = Field(
        description="Whether the research is sufficient to answer the original question"
    )
    reasoning: str = Field(
        description="Explanation of why the research is or isn't complete"
    )


class AnalysisEvaluationAgent(BaseResearchAgent[EvaluationResult]):
    """Agent that analyzes findings and evaluates research completeness."""

    def __init__(self, provider: str, model: str) -> None:
        super().__init__(provider, model, output_type=EvaluationResult)

    def get_system_prompt(self) -> str:
        return EVALUATION_AGENT_PROMPT

    def register_tools(self) -> None:
        """No additional tools needed - uses LLM capabilities directly."""
        pass
