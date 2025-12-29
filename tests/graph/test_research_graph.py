import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.graph.agui.stream import stream_graph
from haiku.rag.graph.research.dependencies import ResearchContext
from haiku.rag.graph.research.graph import build_research_graph
from haiku.rag.graph.research.state import ResearchDeps, ResearchState


@pytest.mark.vcr()
async def test_graph_end_to_end(allow_model_requests, temp_db_path, qa_corpus):
    """Test research graph with real LLM calls recorded via VCR."""
    graph = build_research_graph()

    client = HaikuRAG(temp_db_path, create=True)
    doc = qa_corpus[0]
    await client.create_document(
        content=doc["document_extracted"], uri=doc["document_id"]
    )

    state = ResearchState(
        context=ResearchContext(original_question=doc["question"]),
        max_iterations=1,
        confidence_threshold=0.5,
        max_concurrency=1,
    )

    deps = ResearchDeps(client=client)

    events = []
    result = None
    async for event in stream_graph(graph, state, deps):
        events.append(event)
        if event["type"] == "RUN_FINISHED":
            result = event["result"]
        elif event["type"] == "RUN_ERROR":
            pytest.fail(f"Graph execution failed: {event['message']}")

    assert result is not None, (
        f"No result. Events collected: {[e['type'] for e in events]}"
    )
    assert isinstance(result, dict)
    assert "title" in result
    assert "executive_summary" in result

    event_types = [e["type"] for e in events]
    assert "RUN_STARTED" in event_types
    assert "RUN_FINISHED" in event_types

    client.close()
