from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai.format_prompt import format_as_xml
from pydantic_ai.run import AgentRunResult
from rich.console import Console

from haiku.rag.config import Config
from haiku.rag.research.base import BaseResearchAgent
from haiku.rag.research.dependencies import ResearchContext, ResearchDependencies
from haiku.rag.research.evaluation_agent import (
    AnalysisEvaluationAgent,
    EvaluationResult,
)
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
        self, provider: str | None = Config.RESEARCH_PROVIDER, model: str | None = None
    ):
        # Use provided values or fall back to config defaults
        provider = provider or Config.RESEARCH_PROVIDER or Config.QA_PROVIDER
        model = model or Config.RESEARCH_MODEL or Config.QA_MODEL

        super().__init__(provider, model, output_type=ResearchPlan)

        self.search_agent: SearchSpecialistAgent = SearchSpecialistAgent(
            provider, model
        )
        self.evaluation_agent: AnalysisEvaluationAgent = AnalysisEvaluationAgent(
            provider, model
        )
        self.synthesis_agent: SynthesisAgent = SynthesisAgent(provider, model)

    def get_system_prompt(self) -> str:
        return ORCHESTRATOR_PROMPT

    def register_tools(self) -> None:
        """Register orchestration tools."""
        # Tools are no longer needed - orchestrator directly calls agents
        pass

    def _format_context_for_prompt(self, context: ResearchContext) -> str:
        """Format the research context as XML for inclusion in prompts."""

        context_data = {
            "original_question": context.original_question,
            "unanswered_questions": context.sub_questions,
            "qa_responses": [
                {"question": qa.query, "answer": qa.answer}
                for qa in context.qa_responses
            ],
            "insights": context.insights,
            "gaps": context.gaps,
        }
        return format_as_xml(context_data, root_tag="research_context")

    async def conduct_research(
        self,
        question: str,
        client: Any,
        max_iterations: int = 3,
        confidence_threshold: float = 0.8,
        verbose: bool = False,
        console: Console | None = None,
    ) -> ResearchReport:
        """Conduct comprehensive research on a question.

        Args:
            question: The research question to investigate
            client: HaikuRAG client for document operations
            max_iterations: Maximum number of search-analyze-clarify cycles
            confidence_threshold: Minimum confidence level to stop research (0-1)
            verbose: If True, print progress and intermediate results
            console: Optional Rich console for output

        Returns:
            ResearchReport with comprehensive findings
        """

        # Initialize context
        context = ResearchContext(original_question=question)
        deps = ResearchDependencies(client=client, context=context)

        # Use provided console or create a new one
        console = console or Console() if verbose else None

        # Create initial research plan
        if console:
            console.print("\n[bold cyan]📋 Creating research plan...[/bold cyan]")

        plan_result: AgentRunResult[ResearchPlan] = await self.run(
            f"Create a research plan for: {question}", deps=deps
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
            console.print()

        # Execute research iterations
        for iteration in range(max_iterations):
            if console:
                console.rule(
                    f"[bold yellow]🔄 Iteration {iteration + 1}/{max_iterations}[/bold yellow]"
                )

            # Check if we have questions to search
            if not context.sub_questions:
                # No more questions to explore
                if console:
                    console.print(
                        "[yellow]No more questions to explore. Concluding research.[/yellow]"
                    )
                break

            # Use current sub-questions for this iteration
            questions_to_search = context.sub_questions

            # Search phase - answer all questions in this iteration
            if console:
                console.print(
                    f"\n[bold cyan]🔍 Searching & Answering {len(questions_to_search)} questions:[/bold cyan]"
                )
                for i, q in enumerate(questions_to_search, 1):
                    console.print(f"   {i}. {q}")

            # Run searches for all questions and remove answered ones
            answered_questions = []
            for search_question in questions_to_search:
                try:
                    await self.search_agent.run(search_question, deps=deps)
                except Exception as e:  # pragma: no cover - defensive
                    if console:
                        console.print(
                            f"\n   [red]×[/red] Omitting failed question: {search_question} ({e})"
                        )
                finally:
                    answered_questions.append(search_question)

                if console and context.qa_responses:
                    # Show the last QA response (which should be for this question)
                    latest_qa = context.qa_responses[-1]
                    answer_preview = (
                        latest_qa.answer[:150] + "..."
                        if len(latest_qa.answer) > 150
                        else latest_qa.answer
                    )
                    console.print(
                        f"\n   [green]✓[/green] {search_question[:50]}..."
                        if len(search_question) > 50
                        else f"\n   [green]✓[/green] {search_question}"
                    )
                    console.print(f"      {answer_preview}")

            # Remove answered questions from the list
            for question in answered_questions:
                if question in context.sub_questions:
                    context.sub_questions.remove(question)

            # Analysis and Evaluation phase
            if console:
                console.print(
                    "\n[bold cyan]📊 Analyzing and evaluating research progress...[/bold cyan]"
                )

            # Format context for the evaluation agent
            context_xml = self._format_context_for_prompt(context)
            evaluation_prompt = f"""Analyze all gathered information and evaluate the completeness of research.

{context_xml}

Evaluate the research progress for the original question and identify any remaining gaps."""

            evaluation_result = await self.evaluation_agent.run(
                evaluation_prompt,
                deps=deps,
            )

            if console and evaluation_result.output:
                output = evaluation_result.output
                if output.key_insights:
                    console.print("   [bold]Key insights:[/bold]")
                    for insight in output.key_insights:
                        console.print(f"   • {insight}")
                console.print(
                    f"   Confidence: [yellow]{output.confidence_score:.1%}[/yellow]"
                )
                status = (
                    "[green]Yes[/green]" if output.is_sufficient else "[red]No[/red]"
                )
                console.print(f"   Sufficient: {status}")

            # Store insights
            for insight in evaluation_result.output.key_insights:
                context.add_insight(insight)

            # Add new questions to the sub-questions list
            for new_q in evaluation_result.output.new_questions:
                if new_q not in context.sub_questions:
                    context.sub_questions.append(new_q)

            # Check if research is sufficient
            if self._should_stop_research(evaluation_result, confidence_threshold):
                if console:
                    console.print(
                        f"\n[bold green]✅ Stopping research:[/bold green] {evaluation_result.output.reasoning}"
                    )
                break

        # Generate final report
        if console:
            console.print(
                "\n[bold cyan]📝 Generating final research report...[/bold cyan]"
            )

        # Format context for the synthesis agent
        final_context_xml = self._format_context_for_prompt(context)
        synthesis_prompt = f"""Generate a comprehensive research report based on all gathered information.

{final_context_xml}

Create a detailed report that synthesizes all findings into a coherent response."""

        report_result: AgentRunResult[ResearchReport] = await self.synthesis_agent.run(
            synthesis_prompt, deps=deps
        )

        if console:
            console.print("[bold green]✅ Research complete![/bold green]")

        return report_result.output

    def _should_stop_research(
        self,
        evaluation_result: AgentRunResult[EvaluationResult],
        confidence_threshold: float,
    ) -> bool:
        """Determine if research should stop based on evaluation."""

        result = evaluation_result.output

        # Stop if the agent indicates sufficient information AND confidence exceeds threshold
        return result.is_sufficient and result.confidence_score >= confidence_threshold
