from pydantic import BaseModel
from pydantic_ai import FunctionToolset

from haiku.rag.agents.rlm.agent import create_rlm_agent
from haiku.rag.agents.rlm.dependencies import RLMContext, RLMDeps
from haiku.rag.agents.rlm.models import CodeExecution
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.tools.context import ToolContext
from haiku.rag.tools.filters import (
    build_document_filter,
    build_multi_document_filter,
    combine_filters,
)
from haiku.rag.tools.models import AnalysisResult
from haiku.rag.tools.session import SESSION_NAMESPACE, SessionState

ANALYSIS_NAMESPACE = "haiku.rag.analysis"


class AnalysisState(BaseModel):
    """State for analysis toolset.

    Tracks code executions across tool invocations.
    """

    code_executions: list[CodeExecution] = []


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
            If provided, code executions are tracked in AnalysisState.
            If SessionState is registered, it will be used for dynamic
            document filtering.
        base_filter: Optional base SQL WHERE clause applied to searches.
        tool_name: Name for the analyze tool. Defaults to "analyze".

    Returns:
        FunctionToolset with an analyze tool.
    """
    # Get or create analysis state if context provided
    state: AnalysisState | None = None
    if context is not None:
        state = context.get_or_create(ANALYSIS_NAMESPACE, AnalysisState)

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
        # Get session filter from session state
        session_filter = None
        if context is not None:
            session_state = context.get_typed(SESSION_NAMESPACE, SessionState)
            if session_state is not None and session_state.document_filter:
                session_filter = build_multi_document_filter(
                    session_state.document_filter
                )

        # Build filter from base_filter, session_filter, and document_name
        doc_filter = build_document_filter(document_name) if document_name else None
        effective_filter = combine_filters(
            combine_filters(base_filter, session_filter), doc_filter
        )

        # Create RLM context and deps
        rlm_context = RLMContext(filter=effective_filter)
        deps = RLMDeps(
            client=client,
            config=config,
            context=rlm_context,
        )

        # Run RLM agent
        rlm_agent = create_rlm_agent(config)
        result = await rlm_agent.run(task, deps=deps)

        # Track code executions in state
        code_executions = rlm_context.code_executions
        if state is not None:
            state.code_executions.extend(code_executions)

        return AnalysisResult(
            answer=result.output.answer,
            code_executed=len(code_executions) > 0,
            execution_count=len(code_executions),
        )

    toolset = FunctionToolset()
    toolset.add_function(analyze, name=tool_name)
    return toolset
