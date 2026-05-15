from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ToolReturn

from haiku.rag.agents.analysis.dependencies import AnalysisDeps
from haiku.rag.agents.analysis.models import CodeExecution, RawAnalysisResult
from haiku.rag.agents.analysis.prompts import ANALYSIS_SYSTEM_PROMPT
from haiku.rag.config.models import AppConfig
from haiku.rag.utils import get_model


def create_analysis_agent(config: AppConfig) -> Agent[AnalysisDeps, RawAnalysisResult]:
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

    agent: Agent[AnalysisDeps, RawAnalysisResult] = Agent(  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        model,
        deps_type=AnalysisDeps,
        output_type=RawAnalysisResult,
        instructions=ANALYSIS_SYSTEM_PROMPT,
        tool_retries=3,
        output_retries=3,
    )

    @agent.tool
    async def execute_code(
        ctx: RunContext[AnalysisDeps], code: str
    ) -> CodeExecution | ToolReturn:
        """Execute Python code in a sandboxed interpreter.

        The code has access to search(), list_documents(), and show_image()
        external functions, and a virtual filesystem at /documents/ with
        document content and structure. Use print() to output results.

        When the code calls ``show_image(document_id, self_ref)``, the queued
        picture bytes are attached to the tool response as ``BinaryContent``
        so a vision-capable driving model can actually see the image.

        Args:
            code: Python code to execute.

        Returns:
            Structured result with success status, stdout, and stderr. Wrapped
            in a ``ToolReturn`` carrying ``BinaryContent`` parts whenever the
            code called ``show_image``.
        """
        result = await ctx.deps.sandbox.execute(code)

        execution = CodeExecution(
            code=code,
            stdout=result.stdout,
            stderr=result.stderr,
            success=result.success,
        )

        if result.binary_attachments:
            return ToolReturn(
                return_value=execution, content=list(result.binary_attachments)
            )
        return execution

    return agent
