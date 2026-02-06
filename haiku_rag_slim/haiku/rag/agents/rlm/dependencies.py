from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from haiku.rag.store.models import Document

if TYPE_CHECKING:
    from haiku.rag.agents.rlm.docker_sandbox import DockerSandbox


@dataclass
class RLMContext:
    """Mutable context accumulating data during RLM execution."""

    documents: list[Document] | None = None
    filter: str | None = None


@dataclass
class RLMDeps:
    """Dependencies for RLM agent."""

    sandbox: "DockerSandbox"
    context: RLMContext = field(default_factory=RLMContext)
