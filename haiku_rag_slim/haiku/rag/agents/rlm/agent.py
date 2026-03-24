from pydantic_ai import Agent, RunContext

from haiku.rag.agents.rlm.dependencies import RLMDeps
from haiku.rag.agents.rlm.models import CodeExecution, RLMResult
from haiku.rag.agents.rlm.prompts import RLM_SYSTEM_PROMPT
from haiku.rag.config.models import AppConfig
from haiku.rag.utils import get_model


def create_rlm_agent(config: AppConfig) -> Agent[RLMDeps, RLMResult]:
    """Create an RLM agent with code execution capability.

    The RLM (Recursive Language Model) agent can write and execute Python code
    in a sandboxed environment to solve problems that require computation,
    aggregation, or complex traversal across documents.

    Args:
        config: Application configuration.

    Returns:
        A pydantic-ai Agent configured for RLM execution.
    """
    model = get_model(config.rlm.model, config)

    agent: Agent[RLMDeps, RLMResult] = Agent(  # type: ignore[invalid-assignment]
        model,
        deps_type=RLMDeps,
        output_type=RLMResult,
        instructions=RLM_SYSTEM_PROMPT,
        retries=3,
    )

    @agent.tool
    async def execute_code(ctx: RunContext[RLMDeps], code: str) -> CodeExecution:
        """Execute Python code in a sandboxed interpreter.

        The code has access to haiku.rag functions (search, list_documents,
        get_document, get_chunk, llm).

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
