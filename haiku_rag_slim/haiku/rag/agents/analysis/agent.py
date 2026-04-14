from pydantic_ai import Agent, RunContext

from haiku.rag.agents.analysis.dependencies import AnalysisDeps
from haiku.rag.agents.analysis.models import AnalysisResult, CodeExecution
from haiku.rag.agents.analysis.prompts import ANALYSIS_SYSTEM_PROMPT
from haiku.rag.config.models import AppConfig
from haiku.rag.utils import get_model


def create_analysis_agent(config: AppConfig) -> Agent[AnalysisDeps, AnalysisResult]:
    """Create an analysis agent with code execution capability.

    The analysis agent can write and execute Python code in a sandboxed
    environment to solve problems that require computation, aggregation,
    or complex traversal across documents.

    Args:
        config: Application configuration.

    Returns:
        A pydantic-ai Agent configured for analysis execution.
    """
    model = get_model(config.analysis.model, config)

    agent: Agent[AnalysisDeps, AnalysisResult] = Agent(  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        model,
        deps_type=AnalysisDeps,
        output_type=AnalysisResult,
        instructions=ANALYSIS_SYSTEM_PROMPT,
        retries=3,
    )

    @agent.tool
    async def execute_code(ctx: RunContext[AnalysisDeps], code: str) -> CodeExecution:
        """Execute Python code in a sandboxed interpreter.

        The code has access to haiku.rag functions (search, get_context,
        list_documents, get_document, get_docling_document, llm).

        Use print() to output results.

        Args:
            code: Python code to execute.

        Returns:
            Structured result with success status, stdout, and stderr.
        """
        result = await ctx.deps.sandbox.execute(code)

        execution = CodeExecution(
            code=code,
            stdout=result.stdout,
            stderr=result.stderr,
            success=result.success,
        )

        return execution

    return agent
