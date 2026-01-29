from pydantic_ai import Agent, RunContext

from haiku.rag.agents.rlm.dependencies import RLMDeps
from haiku.rag.agents.rlm.models import CodeExecution, RLMResult
from haiku.rag.agents.rlm.prompts import RLM_SYSTEM_PROMPT
from haiku.rag.agents.rlm.sandbox import REPLEnvironment
from haiku.rag.config.models import AppConfig
from haiku.rag.utils import get_model

_repl_cache: dict[int, REPLEnvironment] = {}


def _get_or_create_repl(ctx) -> REPLEnvironment:
    """Get or create a REPL environment for this context."""
    key = id(ctx.deps)
    if key not in _repl_cache:
        _repl_cache[key] = REPLEnvironment(
            client=ctx.deps.client,
            config=ctx.deps.rlm_config,
            context=ctx.deps.context,
        )
    return _repl_cache[key]


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
    model = get_model(config.qa.model, config)

    agent: Agent[RLMDeps, RLMResult] = Agent(  # type: ignore[invalid-assignment]
        model,
        deps_type=RLMDeps,
        output_type=RLMResult,
        instructions=RLM_SYSTEM_PROMPT,
        retries=3,
    )

    @agent.tool
    async def execute_code(ctx: RunContext[RLMDeps], code: str) -> CodeExecution:
        """Execute Python code in the sandboxed environment.

        The code has access to haiku.rag functions (search, list_documents,
        get_document, get_docling_document, ask) and safe standard library
        modules (json, re, collections, math, statistics, itertools,
        functools, datetime, typing).

        Use print() to output results. Variables persist between executions.

        Args:
            code: Python code to execute.

        Returns:
            Structured result with success status, stdout, and stderr.
        """
        repl = _get_or_create_repl(ctx)

        result = await repl.execute_async(code)

        execution = CodeExecution(
            code=code,
            stdout=result.stdout,
            stderr=result.stderr,
            success=result.success,
        )

        ctx.deps.context.code_executions.append(execution)

        return execution

    return agent
