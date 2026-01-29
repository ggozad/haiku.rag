from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel

from haiku.rag.store.models import Document, SearchResult

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG
    from haiku.rag.config.models import AppConfig


class RLMConfig(BaseModel):
    """Configuration for RLM agent sandbox execution."""

    code_timeout: float = 60.0
    max_output_chars: int = 50_000
    max_tool_calls: int = 20


@dataclass
class RLMContext:
    """Mutable context accumulating data during RLM execution."""

    documents: list[Document] | None = None
    search_results: list[SearchResult] = field(default_factory=list)
    code_executions: list[dict] = field(default_factory=list)


@dataclass
class RLMDeps:
    """Dependencies for RLM agent."""

    client: "HaikuRAG"
    config: "AppConfig"
    rlm_config: RLMConfig = field(default_factory=RLMConfig)
    context: RLMContext = field(default_factory=RLMContext)
