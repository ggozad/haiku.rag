import asyncio

from haiku.rag.research.dependencies import ResearchContext
from haiku.rag.research.graph import ResearchState, build_research_graph


def test_build_graph_and_state():
    graph = build_research_graph()
    assert graph is not None

    state = ResearchState(
        context=ResearchContext(
            original_question="What are the key features of haiku.rag?"
        ),
        max_iterations=1,
        confidence_threshold=0.8,
    )
    assert state.iterations == 0
    assert state.context.sub_questions == []


def test_async_loop_available():
    # Ensure an event loop can be created in test env
    loop = asyncio.new_event_loop()
    loop.close()
