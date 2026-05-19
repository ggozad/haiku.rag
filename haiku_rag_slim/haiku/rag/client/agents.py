from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from haiku.rag.agents.analysis.models import AnalysisResult
    from haiku.rag.agents.research.models import Citation, ResearchReport
    from haiku.rag.client import HaikuRAG


async def ask(
    client: "HaikuRAG",
    question: str,
    filter: str | None = None,
) -> "tuple[str, list[Citation]]":
    """Ask a question against the knowledge base via the rag skill.

    Args:
        client: The HaikuRAG client.
        question: The question to ask.
        filter: SQL WHERE clause to filter documents.

    Returns:
        Tuple of (answer text, list of resolved citations).
    """
    from haiku.rag.skills.rag import RAGState, create_skill
    from haiku.rag.utils import get_model
    from haiku.skills import run_skill

    skill = create_skill(db_path=client.store.db_path, config=client._config)
    state = RAGState(document_filter=filter)
    model = get_model(client._config.qa.model, client._config)
    answer, _, _ = await run_skill(model, skill, question, state=state)
    citations = [
        state.citation_index[cid]
        for cid in state.citations
        if cid in state.citation_index
    ]
    return answer, citations


async def research(
    client: "HaikuRAG",
    question: str,
    *,
    filter: str | None = None,
    max_iterations: int | None = None,
) -> "ResearchReport":
    """Run multi-agent research to investigate a question.

    Args:
        client: The HaikuRAG client.
        question: The research question to investigate.
        filter: SQL WHERE clause to filter documents.
        max_iterations: Override max iterations (None uses config default).

    Returns:
        ResearchReport with structured findings.
    """
    from haiku.rag.agents.research.dependencies import ResearchContext
    from haiku.rag.agents.research.graph import build_research_graph
    from haiku.rag.agents.research.state import ResearchDeps, ResearchState

    graph = build_research_graph(config=client._config)
    context = ResearchContext(original_question=question)
    state = ResearchState.from_config(
        context=context, config=client._config, max_iterations=max_iterations
    )
    state.search_filter = filter
    deps = ResearchDeps(client=client)

    return await graph.run(state=state, deps=deps)


async def analyze(
    client: "HaikuRAG",
    question: str,
    documents: list[str] | None = None,
    filter: str | None = None,
) -> "AnalysisResult":
    """Answer a question using the analysis agent with code execution.

    The analysis agent can write and execute Python code in a sandboxed
    environment to solve problems that require computation, aggregation, or
    complex traversal across documents.

    Args:
        client: The HaikuRAG client.
        question: The question to answer.
        documents: Optional list of document IDs or titles to pre-load.
        filter: SQL WHERE clause to filter documents during searches.

    Returns:
        AnalysisResult with the answer and the final consolidated program.
    """
    from haiku.rag.agents.analysis import (
        AnalysisContext,
        AnalysisDeps,
        Sandbox,
        create_analysis_agent,
    )
    from haiku.rag.agents.analysis.models import AnalysisResult
    from haiku.rag.agents.research.models import Citation

    context = AnalysisContext(filter=filter)

    if documents:
        loaded_docs = []
        for doc_ref in documents:
            doc = await client.resolve_document(doc_ref)
            if doc:
                loaded_docs.append(doc)
        context.documents = loaded_docs if loaded_docs else None

    sandbox = Sandbox(
        db_path=client.store.db_path,
        config=client._config,
        context=context,
    )
    deps = AnalysisDeps(
        sandbox=sandbox,
        context=context,
    )

    agent = create_analysis_agent(client._config)
    result = await agent.run(question, deps=deps)

    output = result.output
    seen: set[str] = set()
    citations: list[Citation] = []
    for sr in sandbox._search_results:
        if sr.chunk_id and sr.chunk_id not in seen:
            seen.add(sr.chunk_id)
            citations.append(
                Citation(
                    index=len(seen),
                    document_id=sr.document_id or "",
                    chunk_id=sr.chunk_id,
                    document_uri=sr.document_uri or "",
                    document_title=sr.document_title,
                    page_numbers=sr.page_numbers,
                    headings=sr.headings,
                    content=sr.content,
                )
            )
    return AnalysisResult(
        answer=output.answer,
        program=output.program,
        citations=citations,
    )
