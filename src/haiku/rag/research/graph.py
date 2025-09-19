from pydantic_graph import Graph

from haiku.rag.research.models import ResearchReport
from haiku.rag.research.nodes.evaluate import EvaluateNode
from haiku.rag.research.nodes.plan import PlanNode
from haiku.rag.research.nodes.search import SearchDispatchNode
from haiku.rag.research.nodes.synthesize import SynthesizeNode
from haiku.rag.research.state import ResearchDeps, ResearchState

__all__ = [
    "PlanNode",
    "SearchDispatchNode",
    "EvaluateNode",
    "SynthesizeNode",
    "ResearchState",
    "ResearchDeps",
    "build_research_graph",
]


def build_research_graph() -> Graph[ResearchState, ResearchDeps, ResearchReport]:
    return Graph(
        nodes=[
            PlanNode,
            SearchDispatchNode,
            EvaluateNode,
            SynthesizeNode,
        ]
    )
