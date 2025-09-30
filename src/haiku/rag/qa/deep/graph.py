from pydantic_graph import Graph

from haiku.rag.qa.deep.models import DeepAnswer
from haiku.rag.qa.deep.nodes import (
    DeepDecisionNode,
    DeepPlanNode,
    DeepSearchDispatchNode,
    DeepSynthesizeNode,
)
from haiku.rag.qa.deep.state import DeepQADeps, DeepQAState


def build_deep_qa_graph() -> Graph[DeepQAState, DeepQADeps, DeepAnswer]:
    return Graph(
        nodes=[
            DeepPlanNode,
            DeepSearchDispatchNode,
            DeepDecisionNode,
            DeepSynthesizeNode,
        ]
    )
