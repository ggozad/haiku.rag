from pydantic_graph import Graph

from haiku.rag.qa.deep.models import DeepQAAnswer
from haiku.rag.qa.deep.nodes import (
    DeepQADecisionNode,
    DeepQAPlanNode,
    DeepQASearchDispatchNode,
    DeepQASynthesizeNode,
)
from haiku.rag.qa.deep.state import DeepQADeps, DeepQAState


def build_deep_qa_graph() -> Graph[DeepQAState, DeepQADeps, DeepQAAnswer]:
    return Graph(
        nodes=[
            DeepQAPlanNode,
            DeepQASearchDispatchNode,
            DeepQADecisionNode,
            DeepQASynthesizeNode,
        ]
    )
