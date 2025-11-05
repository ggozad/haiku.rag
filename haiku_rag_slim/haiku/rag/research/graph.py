from pydantic_graph import Graph

from haiku.rag.graph.nodes.analysis import AnalyzeInsightsNode, DecisionNode
from haiku.rag.graph.nodes.plan import PlanNode
from haiku.rag.graph.nodes.search import SearchDispatchNode
from haiku.rag.graph.nodes.synthesize import SynthesizeNode
from haiku.rag.research.models import ResearchReport
from haiku.rag.research.state import ResearchDeps, ResearchState


def build_research_graph() -> Graph[ResearchState, ResearchDeps, ResearchReport]:
    return Graph(
        nodes=[
            PlanNode,
            SearchDispatchNode,
            AnalyzeInsightsNode,
            DecisionNode,
            SynthesizeNode,
        ]
    )
