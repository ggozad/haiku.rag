from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from haiku.rag.store.models import Document, SearchResult

if TYPE_CHECKING:
    from haiku.rag.agents.rlm.docker_sandbox import DockerSandbox
    from haiku.rag.agents.rlm.models import CodeExecution


@dataclass
class RLMContext:
    """Mutable context accumulating data during RLM execution."""

    documents: list[Document] | None = None
    filter: str | None = None
    search_results: list[SearchResult] = field(default_factory=list)
    code_executions: "list[CodeExecution]" = field(default_factory=list)


@dataclass
class RLMDeps:
    """Dependencies for RLM agent."""

    sandbox: "DockerSandbox"
    context: RLMContext = field(default_factory=RLMContext)
