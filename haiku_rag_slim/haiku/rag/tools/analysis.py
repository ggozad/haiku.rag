from pydantic_ai import FunctionToolset

from haiku.rag.agents.rlm.agent import create_rlm_agent
from haiku.rag.agents.rlm.dependencies import RLMContext, RLMDeps
from haiku.rag.agents.rlm.docker_sandbox import DockerSandbox
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.tools.context import ToolContext
from haiku.rag.tools.filters import (
    build_document_filter,
    combine_filters,
    get_session_filter,
)
from haiku.rag.tools.models import AnalysisResult


def create_analysis_toolset(
    client: HaikuRAG,
    config: AppConfig,
    context: ToolContext | None = None,
    base_filter: str | None = None,
    tool_name: str = "analyze",
) -> FunctionToolset:
    """Create a toolset with code analysis capabilities via RLM agent.

    Args:
        client: HaikuRAG client for document operations.
        config: Application configuration.
        context: Optional ToolContext for state accumulation.
            If SessionState is registered, it will be used for dynamic
            document filtering.
        base_filter: Optional base SQL WHERE clause applied to searches.
        tool_name: Name for the analyze tool. Defaults to "analyze".

    Returns:
        FunctionToolset with an analyze tool.
    """

    async def analyze(
        task: str,
        document_name: str | None = None,
    ) -> AnalysisResult:
        """Execute a computational task via code execution.

        Uses the RLM (Recursive Language Model) agent to write and execute
        Python code to answer the task.

        Args:
            task: A specific, actionable instruction describing what to compute.
            document_name: Optional document name/title to focus on.

        Returns:
            AnalysisResult with answer and execution metadata.
        """
        doc_filter = build_document_filter(document_name) if document_name else None
        effective_filter = combine_filters(
            get_session_filter(context, base_filter), doc_filter
        )

        rlm_context = RLMContext(filter=effective_filter)

        async with DockerSandbox(
            client=client,
            config=config.rlm,
            context=rlm_context,
            image=config.rlm.docker_image,
        ) as sandbox:
            deps = RLMDeps(
                sandbox=sandbox,
                context=rlm_context,
            )

            rlm_agent = create_rlm_agent(config)
            result = await rlm_agent.run(task, deps=deps)

            program = result.output.program

            return AnalysisResult(
                answer=result.output.answer,
                code_executed=bool(program),
            )

    toolset = FunctionToolset()
    toolset.add_function(analyze, name=tool_name)
    return toolset
