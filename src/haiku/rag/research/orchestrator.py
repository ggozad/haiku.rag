"""Research orchestrator agent that coordinates specialized agents."""

from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from haiku.rag.research.analysis_agent import AnalysisAgent, AnalysisResult
from haiku.rag.research.base import BaseResearchAgent
from haiku.rag.research.clarification_agent import (
    ClarificationAgent,
    ClarificationResult,
)
from haiku.rag.research.dependencies import ResearchContext, ResearchDependencies
from haiku.rag.research.search_agent import SearchSpecialistAgent
from haiku.rag.research.synthesis_agent import ResearchReport, SynthesisAgent


class ResearchPlan(BaseModel):
    """Research execution plan."""

    main_question: str = Field(description="The main research question")
    sub_questions: list[str] = Field(
        description="Decomposed sub-questions to investigate"
    )
    search_strategies: list[str] = Field(
        description="Different search approaches to use"
    )
    success_criteria: list[str] = Field(description="Criteria for successful research")


class ResearchOrchestrator(BaseResearchAgent):
    """Orchestrator agent that coordinates the research workflow."""

    def __init__(self, provider: str, model: str):
        super().__init__(provider, model, output_type=ResearchPlan)

        # Initialize specialized agents
        self.search_agent = SearchSpecialistAgent(provider, model)
        self.analysis_agent = AnalysisAgent(provider, model)
        self.clarification_agent = ClarificationAgent(provider, model)
        self.synthesis_agent = SynthesisAgent(provider, model)

    def get_system_prompt(self) -> str:
        return """You are a research orchestrator responsible for coordinating a comprehensive research workflow.

        Your role is to:
        1. Understand and decompose the research question
        2. Plan a systematic research approach
        3. Coordinate specialized agents to gather and analyze information
        4. Ensure comprehensive coverage of the topic
        5. Iterate based on findings and gaps

        Create a research plan that:
        - Breaks down complex questions into manageable parts
        - Identifies multiple search strategies
        - Defines clear success criteria
        - Ensures thorough investigation"""

    def register_tools(self) -> None:
        """Register orchestration tools."""

        @self.agent.tool
        async def delegate_search(
            ctx: RunContext[ResearchDependencies], queries: list[str], limit: int = 5
        ) -> Any:
            """Delegate search to the search specialist agent."""
            # Pass the context to maintain usage tracking
            result = await self.search_agent.run(
                f"Search for: {', '.join(queries)}", deps=ctx.deps, usage=ctx.usage
            )
            return result

        @self.agent.tool
        async def delegate_analysis(
            ctx: RunContext[ResearchDependencies],
        ) -> AnalysisResult:
            """Delegate analysis to the analysis agent."""
            # Get search results from context
            all_documents = []
            for search in ctx.deps.context.search_results:
                all_documents.extend(search.get("results", []))

            # Pass documents for analysis
            result = await self.analysis_agent.run(
                f"Analyze these {len(all_documents)} documents from our search",
                deps=ctx.deps,
                usage=ctx.usage,
            )

            # Store analysis insights in context
            if hasattr(result, "output") and isinstance(result.output, AnalysisResult):
                for insight in result.output.key_insights:
                    ctx.deps.context.add_insight(insight)

            return result.output if hasattr(result, "output") else result

        @self.agent.tool
        async def delegate_clarification(
            ctx: RunContext[ResearchDependencies],
        ) -> ClarificationResult:
            """Delegate gap analysis to the clarification agent."""
            result = await self.clarification_agent.run(
                f"Evaluate the completeness of research on: {ctx.deps.context.original_question}",
                deps=ctx.deps,
                usage=ctx.usage,
            )

            # Store identified gaps in context
            if hasattr(result, "output") and isinstance(
                result.output, ClarificationResult
            ):
                for gap in result.output.information_gaps:
                    ctx.deps.context.add_gap(gap)
                ctx.deps.context.follow_up_questions.extend(
                    result.output.follow_up_questions
                )

            return result.output if hasattr(result, "output") else result

        @self.agent.tool
        async def generate_report(
            ctx: RunContext[ResearchDependencies],
        ) -> ResearchReport:
            """Generate final research report using synthesis agent."""
            result = await self.synthesis_agent.run(
                f"Create a comprehensive research report for: {ctx.deps.context.original_question}",
                deps=ctx.deps,
                usage=ctx.usage,
            )
            return result.output if hasattr(result, "output") else result

    async def conduct_research(
        self,
        question: str,
        client: Any,
        max_iterations: int = 3,
        confidence_threshold: float = 0.8,
    ) -> ResearchReport:
        """Conduct comprehensive research on a question.

        Args:
            question: The research question to investigate
            client: HaikuRAG client for document operations
            max_iterations: Maximum number of search-analyze-clarify cycles
            confidence_threshold: Minimum confidence level to stop research (0-1)

        Returns:
            ResearchReport with comprehensive findings
        """

        # Initialize context
        context = ResearchContext(original_question=question)
        deps = ResearchDependencies(client=client, context=context)

        # Create initial research plan
        plan_result = await self.run(
            f"Create a research plan for: {question}", deps=deps
        )

        if hasattr(plan_result, "output") and isinstance(
            plan_result.output, ResearchPlan
        ):
            context.sub_questions = plan_result.output.sub_questions

        # Execute research iterations
        for iteration in range(max_iterations):
            # Determine what to search for in this iteration
            if context.follow_up_questions:
                # Use follow-up questions from previous clarification
                search_target = context.follow_up_questions[:3]  # Take top 3 follow-ups
                search_prompt = f"Search for: {', '.join(search_target)}"
            elif iteration < len(context.sub_questions):
                # Use pre-planned sub-questions
                search_prompt = f"Search for: {context.sub_questions[iteration]}"
            else:
                # Fall back to original question with variation
                search_prompt = f"Additional search for: {question}"

            # Search phase
            await self.run(search_prompt, deps=deps)

            # Analysis phase (only if we have results)
            if context.search_results:
                await self.run("Analyze the gathered information", deps=deps)

            # Clarification phase - evaluate completeness
            clarification_result = await self.run(
                f"Evaluate the completeness of research for: {question}. "
                f"Consider all information gathered so far and determine if we have sufficient "
                f"information to provide a comprehensive answer.",
                deps=deps,
            )

            # Check if research is sufficient based on semantic evaluation
            if self._should_stop_research(clarification_result, confidence_threshold):
                # Log the reasoning for stopping
                if hasattr(clarification_result, "output") and isinstance(
                    clarification_result.output, ClarificationResult
                ):
                    print(f"Stopping research: {clarification_result.output.reasoning}")
                break

        # Generate final report
        report_result = await self.run("Generate the final research report", deps=deps)
        return (
            report_result.output if hasattr(report_result, "output") else report_result
        )

    def _should_stop_research(
        self, clarification_result: Any, confidence_threshold: float
    ) -> bool:
        """Determine if research should stop based on semantic completeness evaluation."""

        if not hasattr(clarification_result, "output") or not isinstance(
            clarification_result.output, ClarificationResult
        ):
            # If we can't evaluate, continue researching
            return False

        result = clarification_result.output

        # Use the LLM's semantic evaluation
        # Stop if the agent indicates sufficient information AND confidence exceeds threshold
        return result.is_sufficient and result.confidence_score >= confidence_threshold
