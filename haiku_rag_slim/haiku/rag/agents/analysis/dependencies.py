from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from haiku.rag.store.models import Document

if TYPE_CHECKING:
    from haiku.rag.agents.analysis.sandbox import Sandbox


@dataclass
class AnalysisContext:
    """Mutable context accumulating data during analysis execution."""

    documents: list[Document] | None = None
    filter: str | None = None


@dataclass
class AnalysisDeps:
    """Dependencies for analysis agent."""

    sandbox: "Sandbox"
    context: AnalysisContext = field(default_factory=AnalysisContext)
