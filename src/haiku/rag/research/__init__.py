"""Multi-agent research workflow for advanced RAG queries."""

from haiku.rag.research.analysis_agent import AnalysisAgent, AnalysisResult
from haiku.rag.research.base import BaseResearchAgent, ResearchOutput, SearchResult
from haiku.rag.research.clarification_agent import (
    ClarificationAgent,
    ClarificationResult,
)
from haiku.rag.research.dependencies import ResearchContext, ResearchDependencies
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
    "SearchSpecialistAgent",
    "AnalysisAgent",
    "AnalysisResult",
    "ClarificationAgent",
    "ClarificationResult",
    "SynthesisAgent",
    "ResearchReport",
    # Orchestrator
    "ResearchOrchestrator",
    "ResearchPlan",
]
