from pydantic import BaseModel, Field
from pydantic_ai.run import AgentRunResult

from haiku.rag.research.base import BaseResearchAgent
from haiku.rag.research.dependencies import (
    ResearchDependencies,
    _format_context_for_prompt,
)
from haiku.rag.research.prompts import EVALUATION_AGENT_PROMPT


class EvaluationResult(BaseModel):
    """Result of analysis and evaluation."""

    key_insights: list[str] = Field(
        description="Main insights extracted from the research so far"
    )
    new_questions: list[str] = Field(
        description="New sub-questions to add to the research (max 3)",
        max_length=3,
        default=[],
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

    async def run(
        self, prompt: str, deps: ResearchDependencies, **kwargs
    ) -> AgentRunResult[EvaluationResult]:
        console = deps.console
        if console:
            console.print(
                "\n[bold cyan]📊 Analyzing and evaluating research progress...[/bold cyan]"
            )

        # Format context for the evaluation agent
        context_xml = _format_context_for_prompt(deps.context)
        evaluation_prompt = f"""Analyze all gathered information and evaluate the completeness of research.

{context_xml}

Evaluate the research progress for the original question and identify any remaining gaps."""

        result = await super().run(evaluation_prompt, deps, **kwargs)
        output = result.output

        # Store insights
        for insight in output.key_insights:
            deps.context.add_insight(insight)

        # Add new questions to the sub-questions list
        for new_q in output.new_questions:
            if new_q not in deps.context.sub_questions:
                deps.context.sub_questions.append(new_q)

        if console:
            if output.key_insights:
                console.print("   [bold]Key insights:[/bold]")
                for insight in output.key_insights:
                    console.print(f"   • {insight}")
            console.print(
                f"   Confidence: [yellow]{output.confidence_score:.1%}[/yellow]"
            )
            status = "[green]Yes[/green]" if output.is_sufficient else "[red]No[/red]"
            console.print(f"   Sufficient: {status}")

        return result

    def get_system_prompt(self) -> str:
        return EVALUATION_AGENT_PROMPT
