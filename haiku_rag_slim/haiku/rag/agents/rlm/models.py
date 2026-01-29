from pydantic import BaseModel, Field

from haiku.rag.agents.research.models import Citation


class CodeExecution(BaseModel):
    """Result of executing a code block in the RLM sandbox."""

    code: str = Field(description="The Python code that was executed")
    stdout: str = Field(description="Standard output captured during execution")
    stderr: str = Field(description="Standard error captured during execution")
    success: bool = Field(description="Whether execution completed without error")


class RLMResult(BaseModel):
    """Result from RLM agent execution."""

    answer: str = Field(description="The answer to the user's question")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations for sources used in the answer",
    )
    code_executions: list[CodeExecution] = Field(
        default_factory=list,
        description="History of code executions during the RLM session",
    )
