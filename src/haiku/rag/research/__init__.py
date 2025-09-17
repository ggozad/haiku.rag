"""Multi-agent research workflow for advanced RAG queries."""

from haiku.rag.research.base import (
    BaseResearchAgent,
    ResearchOutput,
    SearchAnswer,
    SearchResult,
)
from haiku.rag.research.dependencies import ResearchContext, ResearchDependencies
from haiku.rag.research.evaluation_agent import (
    AnalysisEvaluationAgent,
    EvaluationResult,
)
from haiku.rag.research.orchestrator import ResearchOrchestrator, ResearchPlan
from haiku.rag.research.search_agent import SearchSpecialistAgent
from haiku.rag.research.synthesis_agent import ResearchReport, SynthesisAgent

__all__ = [
    # Base classes
    "BaseResearchAgent",
    "ResearchDependencies",
    "ResearchContext",
    "SearchResult",
    "ResearchOutput",
    # Specialized agents
    "SearchAnswer",
    "SearchSpecialistAgent",
    "AnalysisEvaluationAgent",
    "EvaluationResult",
    "SynthesisAgent",
    "ResearchReport",
    # Orchestrator
    "ResearchOrchestrator",
    "ResearchPlan",
]
