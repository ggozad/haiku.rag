from pydantic import BaseModel, Field

from haiku.rag.agents.research.models import Citation


class CodeExecution(BaseModel):
    """Result of executing a code block in the analysis sandbox."""

    code: str = Field(description="The Python code that was executed")
    stdout: str = Field(description="Standard output captured during execution")
    stderr: str = Field(description="Standard error captured during execution")
    success: bool = Field(description="Whether execution completed without error")


class RawAnalysisResult(BaseModel):
    """Raw result from the analysis agent (LLM output)."""

    answer: str = Field(description="The answer to the user's question")
    program: str = Field(description="The final consolidated program")


class AnalysisResult(BaseModel):
    """Result from analysis execution with resolved citations.

    The per-execution program(s) are tracked on ``AnalysisState.executions``
    (populated by the analysis skill's ``execute_code`` tool), not on this
    return value. Consumers that need the executed code should pull it from
    the skill state."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
