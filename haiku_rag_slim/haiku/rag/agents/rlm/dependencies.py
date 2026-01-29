from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from haiku.rag.store.models import Document, SearchResult

if TYPE_CHECKING:
    from haiku.rag.agents.rlm.models import CodeExecution
    from haiku.rag.client import HaikuRAG
    from haiku.rag.config.models import AppConfig


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

    client: "HaikuRAG"
    config: "AppConfig"
    context: RLMContext = field(default_factory=RLMContext)
