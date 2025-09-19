from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai.run import AgentRunResult
from rich.console import Console

from haiku.rag.config import Config
from haiku.rag.research.base import BaseResearchAgent
from haiku.rag.research.dependencies import (
    ResearchContext,
    ResearchDependencies,
)
from haiku.rag.research.evaluation_agent import (
    AnalysisEvaluationAgent,
    EvaluationResult,
)
from haiku.rag.research.presearch_agent import PresearchSurveyAgent
from haiku.rag.research.prompts import ORCHESTRATOR_PROMPT
from haiku.rag.research.search_agent import SearchSpecialistAgent
from haiku.rag.research.synthesis_agent import ResearchReport, SynthesisAgent


class ResearchPlan(BaseModel):
    """Research execution plan."""

    main_question: str = Field(description="The main research question")
    sub_questions: list[str] = Field(
        description="Decomposed sub-questions to investigate (max 3)", max_length=3
    )


class ResearchOrchestrator(BaseResearchAgent[ResearchPlan]):
    """Orchestrator agent that coordinates the research workflow."""

    def __init__(
        self,
        provider: str | None = Config.RESEARCH_PROVIDER,
        model: str | None = None,
    ):
        # Use provided values or fall back to config defaults
        provider = provider or Config.RESEARCH_PROVIDER or Config.QA_PROVIDER
        model = model or Config.RESEARCH_MODEL or Config.QA_MODEL

        super().__init__(provider, model, output_type=ResearchPlan)

        self.search_agent: SearchSpecialistAgent = SearchSpecialistAgent(
            provider, model
        )
        self.presearch_agent: PresearchSurveyAgent = PresearchSurveyAgent(
            provider, model
        )
        self.evaluation_agent: AnalysisEvaluationAgent = AnalysisEvaluationAgent(
            provider, model
        )
        self.synthesis_agent: SynthesisAgent = SynthesisAgent(provider, model)

    def get_system_prompt(self) -> str:
        return ORCHESTRATOR_PROMPT

    def _should_stop_research(
        self,
        evaluation_result: AgentRunResult[EvaluationResult],
        confidence_threshold: float,
    ) -> bool:
        """Determine if research should stop based on evaluation."""

        result = evaluation_result.output
        return result.is_sufficient and result.confidence_score >= confidence_threshold

    async def conduct_research(
        self,
        question: str,
        client: Any,
        max_iterations: int = 3,
        confidence_threshold: float = 0.8,
        verbose: bool = False,
    ) -> ResearchReport:
        """Conduct comprehensive research on a question.

        Args:
            question: The research question to investigate
            client: HaikuRAG client for document operations
            max_iterations: Maximum number of search-analyze-clarify cycles
            confidence_threshold: Minimum confidence level to stop research (0-1)
            verbose: If True, print progress and intermediate results

        Returns:
            ResearchReport with comprehensive findings
        """

        # Initialize context
        context = ResearchContext(original_question=question)
        deps = ResearchDependencies(client=client, context=context)
        if verbose:
            deps.console = Console()

        console = deps.console
        # Create initial research plan
        if console:
            console.print("\n[bold cyan]📋 Creating research plan...[/bold cyan]")

        # Run a simple presearch survey to summarize KB context
        presearch_result = await self.presearch_agent.run(question, deps=deps)
        plan_prompt = (
            "Create a research plan for the main question below.\n\n"
            f"Main question: {question}\n\n"
            "Use this brief presearch summary to inform the plan. Focus the 3 sub-questions "
            "on the most important aspects not already obvious from the current KB context.\n\n"
            f"{presearch_result.output}"
        )

        plan_result: AgentRunResult[ResearchPlan] = await self.run(
            plan_prompt, deps=deps
        )
        context.sub_questions = plan_result.output.sub_questions

        if console:
            console.print("\n[bold green]✅ Research Plan Created:[/bold green]")
            console.print(
                f"   [bold]Main Question:[/bold] {plan_result.output.main_question}"
            )
            console.print("   [bold]Sub-questions:[/bold]")
            for i, sq in enumerate(plan_result.output.sub_questions, 1):
                console.print(f"      {i}. {sq}")

        # Execute research iterations
        for iteration in range(max_iterations):
            if console:
                console.rule(
                    f"[bold yellow]🔄 Iteration {iteration + 1}/{max_iterations}[/bold yellow]"
                )

            # Check if we have questions to search
            if not context.sub_questions:
                if console:
                    console.print(
                        "[yellow]No more questions to explore. Concluding research.[/yellow]"
                    )
                break

            # Use current sub-questions for this iteration
            questions_to_search = context.sub_questions[:]

            # Search phase - answer all questions in this iteration
            if console:
                console.print(
                    f"\n[bold cyan]🔍 Searching & Answering {len(questions_to_search)} questions:[/bold cyan]"
                )

            for search_question in questions_to_search:
                await self.search_agent.run(search_question, deps=deps)

            # Analysis and Evaluation phase

            evaluation_result = await self.evaluation_agent.run("", deps=deps)

            # Check if research is sufficient
            if self._should_stop_research(evaluation_result, confidence_threshold):
                if console:
                    console.print(
                        f"\n[bold green]✅ Stopping research:[/bold green] {evaluation_result.output.reasoning}"
                    )
                break

        # Generate final report
        report_result: AgentRunResult[ResearchReport] = await self.synthesis_agent.run(
            "", deps=deps
        )

        return report_result.output
