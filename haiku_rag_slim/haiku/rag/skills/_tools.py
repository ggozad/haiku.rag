from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_ai import RunContext

from haiku.rag.agents.research.models import Citation
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models.chunk import SearchResult
from haiku.rag.tools.document import DocumentInfo
from haiku.rag.tools.filters import combine_filters
from haiku.rag.tools.qa import QAHistoryEntry
from haiku.skills.state import SkillRunDeps


class ResearchEntry(BaseModel):
    question: str
    title: str
    executive_summary: str


class AnalysisEntry(BaseModel):
    question: str
    answer: str
    program: str | None = None


async def find_relevant_prior_qa(
    qa_history: list[QAHistoryEntry],
    query: str,
    config: AppConfig,
) -> list[QAHistoryEntry]:
    from haiku.rag.embeddings import get_embedder
    from haiku.rag.tools.qa import PRIOR_ANSWER_RELEVANCE_THRESHOLD
    from haiku.rag.utils import cosine_similarity

    if not qa_history:
        return []

    embedder = get_embedder(config)
    query_embedding = await embedder.embed_query(query)

    to_embed = []
    to_embed_indices = []
    for i, qa in enumerate(qa_history):
        if qa.question_embedding is None:
            to_embed.append(qa.question)
            to_embed_indices.append(i)

    if to_embed:
        new_embeddings = await embedder.embed_documents(to_embed)
        for i, idx in enumerate(to_embed_indices):
            qa_history[idx].question_embedding = new_embeddings[i]

    matches = []
    for qa in qa_history:
        if qa.question_embedding is not None:
            similarity = cosine_similarity(query_embedding, qa.question_embedding)
            if similarity >= PRIOR_ANSWER_RELEVANCE_THRESHOLD:
                matches.append(qa)

    return matches


async def skill_search(
    db_path: Path,
    config: AppConfig,
    query: str,
    limit: int | None = None,
    document_filter: str | None = None,
) -> tuple[str, list[SearchResult]]:
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(db_path, config=config, read_only=True) as rag:
        results = await rag.search(
            query,
            limit=limit,
            filter=document_filter,
        )
        results = await rag.expand_context(results)

    formatted = "\n\n---\n\n".join(
        r.format_for_agent(rank=i + 1, total=len(results))
        for i, r in enumerate(results)
    )
    return formatted, list(results)


async def skill_list_documents(
    db_path: Path,
    config: AppConfig,
    limit: int | None = None,
    offset: int | None = None,
    filter: str | None = None,
) -> list[dict[str, Any]]:
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(db_path, config=config, read_only=True) as rag:
        documents = await rag.list_documents(limit, offset, filter=filter)
        return [
            {
                "id": doc.id,
                "title": doc.title,
                "uri": doc.uri,
                "metadata": doc.metadata,
                "created_at": str(doc.created_at),
                "updated_at": str(doc.updated_at),
            }
            for doc in documents
        ]


async def skill_get_document(
    db_path: Path,
    config: AppConfig,
    query: str,
) -> dict[str, Any] | None:
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(db_path, config=config, read_only=True) as rag:
        document = await rag.resolve_document(query)
        if document is None:
            return None
        return {
            "id": document.id,
            "content": document.content,
            "title": document.title,
            "uri": document.uri,
            "metadata": document.metadata,
            "created_at": str(document.created_at),
            "updated_at": str(document.updated_at),
        }


async def skill_ask(
    db_path: Path,
    config: AppConfig,
    question: str,
    qa_history: list[QAHistoryEntry] | None = None,
    document_filter: str | None = None,
) -> tuple[str, list[Citation]]:
    from haiku.rag.client import HaikuRAG
    from haiku.rag.utils import format_citations

    ask_question = question
    if qa_history:
        matches = await find_relevant_prior_qa(qa_history, question, config)
        if matches:
            prior_parts = []
            for qa in matches:
                part = f"Q: {qa.question}\nA: {qa.answer}"
                if qa.citations:
                    part += "\n" + format_citations(qa.citations)
                prior_parts.append(part)
            ask_question = (
                "Context from prior questions in this session:\n\n"
                + "\n\n---\n\n".join(prior_parts)
                + "\n\n---\n\nCurrent question: "
                + question
            )

    async with HaikuRAG(db_path, config=config, read_only=True) as rag:
        answer, citations = await rag.ask(
            ask_question,
            filter=document_filter,
        )

    return answer, citations


async def skill_research(
    db_path: Path,
    config: AppConfig,
    question: str,
    document_filter: str | None = None,
) -> tuple[str, str, str]:
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(db_path, config=config, read_only=True) as rag:
        report = await rag.research(question, filter=document_filter)

    parts = [
        f"# {report.title}",
        f"\n## Executive Summary\n{report.executive_summary}",
    ]
    if report.main_findings:
        parts.append("\n## Main Findings")
        for finding in report.main_findings:
            parts.append(f"- {finding}")
    if report.conclusions:
        parts.append("\n## Conclusions")
        for conclusion in report.conclusions:
            parts.append(f"- {conclusion}")
    if report.limitations:
        parts.append("\n## Limitations")
        for limitation in report.limitations:
            parts.append(f"- {limitation}")
    if report.recommendations:
        parts.append("\n## Recommendations")
        for rec in report.recommendations:
            parts.append(f"- {rec}")
    parts.append(f"\n## Sources\n{report.sources_summary}")

    formatted = "\n".join(parts)
    return formatted, report.title, report.executive_summary


async def skill_analyze(
    db_path: Path,
    config: AppConfig,
    question: str,
    document: str | None = None,
    filter: str | None = None,
) -> tuple[str, str, str | None]:
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(db_path, config=config, read_only=True) as rag:
        documents = [document] if document else None
        result = await rag.rlm(question, documents=documents, filter=filter)
        output = result.answer
        if result.program:
            output += f"\n\nProgram:\n{result.program}"

    return output, result.answer, result.program


def update_documents_state(
    documents_state: list[DocumentInfo],
    doc_dicts: list[dict[str, Any]],
) -> None:
    for doc_dict in doc_dicts:
        doc_info = DocumentInfo(
            id=str(doc_dict["id"]),
            title=doc_dict["title"] or "Untitled",
            uri=doc_dict.get("uri") or "",
            created=doc_dict.get("created_at", ""),
        )
        if not any(d.id == doc_info.id for d in documents_state):
            documents_state.append(doc_info)


def _get_state(ctx: RunContext[SkillRunDeps], state_type: type[BaseModel]) -> Any:
    if ctx.deps and ctx.deps.state and isinstance(ctx.deps.state, state_type):
        return ctx.deps.state
    return None


def create_skill_extras(
    db_path: Path,
    config: AppConfig,
) -> dict[str, Any]:
    """Create non-tool utility functions bound to a specific database.

    Returns a dict of values that can be attached to a Skill's extras:

    Keys:
    - 'db_path': path to the LanceDB used to configure the skill
    - 'config': config passed to (or derived for) the skill
    - 'list_documents': returns info for documents in the database
    - 'visualize_chunk': returns visualizations for chunks in the database
    """

    async def visualize_chunk(chunk_id: str) -> list:
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(db_path, config=config, read_only=True) as rag:
            chunk = await rag.get_chunk_by_id(chunk_id)
            if chunk is None:
                return []
            return await rag.visualize_chunk(chunk)

    async def list_documents(
        limit: int | None = None,
        offset: int | None = None,
        filter: str | None = None,
    ) -> list[dict[str, Any]]:
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(db_path, config=config, read_only=True) as rag:
            documents = await rag.list_documents(limit, offset, filter=filter)
            return [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "uri": doc.uri,
                    "metadata": doc.metadata,
                    "created_at": str(doc.created_at),
                    "updated_at": str(doc.updated_at),
                }
                for doc in documents
            ]

    return {
        "db_path": db_path,
        "config": config,
        "visualize_chunk": visualize_chunk,
        "list_documents": list_documents,
    }


def create_skill_tools(
    db_path: Path,
    config: AppConfig,
    state_type: type[BaseModel],
    tool_names: list[str],
) -> dict[str, Any]:
    """Create tool closures for a skill.

    Returns a dict mapping tool name to async callable.
    Each tool extracts state from RunContext, calls the shared implementation,
    and updates state.
    """
    tools: dict[str, Any] = {}

    if "search" in tool_names:

        async def search(
            ctx: RunContext[SkillRunDeps], query: str, limit: int | None = None
        ) -> str:
            """Search the knowledge base using hybrid search (vector + full-text).

            Returns ranked results with content and metadata.

            Args:
                query: The search query.
                limit: Maximum number of results.
            """
            state = _get_state(ctx, state_type)
            formatted, results = await skill_search(
                db_path,
                config,
                query,
                limit=limit,
                document_filter=state.document_filter if state else None,
            )
            if state:
                state.searches[query] = results
            return formatted

        tools["search"] = search

    if "list_documents" in tool_names:

        async def list_documents(
            ctx: RunContext[SkillRunDeps],
            limit: int | None = None,
            offset: int | None = None,
        ) -> list[dict[str, Any]]:
            """List documents in the knowledge base with optional pagination.

            Args:
                limit: Maximum number of documents to return.
                offset: Number of documents to skip.
            """
            state = _get_state(ctx, state_type)
            result = await skill_list_documents(
                db_path,
                config,
                limit,
                offset,
                filter=state.document_filter if state else None,
            )
            if state:
                update_documents_state(state.documents, result)
            return result

        tools["list_documents"] = list_documents

    if "get_document" in tool_names:

        async def get_document(
            ctx: RunContext[SkillRunDeps], query: str
        ) -> dict[str, Any] | None:
            """Retrieve a document by ID, title, or URI.

            Args:
                query: Document ID, title, or URI to look up.
            """
            result = await skill_get_document(db_path, config, query)
            if result is not None:
                state = _get_state(ctx, state_type)
                if state:
                    update_documents_state(state.documents, [result])
            return result

        tools["get_document"] = get_document

    if "ask" in tool_names:

        async def ask(ctx: RunContext[SkillRunDeps], question: str) -> str:
            """Ask a question and get an answer with citations from the knowledge base.

            Args:
                question: The question to ask.
            """
            from haiku.rag.utils import format_citations

            state = _get_state(ctx, state_type)
            answer, citations = await skill_ask(
                db_path,
                config,
                question,
                qa_history=state.qa_history if state else None,
                document_filter=state.document_filter if state else None,
            )

            if state:
                next_index = len(state.citations) + 1
                for citation in citations:
                    citation.index = next_index
                    next_index += 1
                state.citations.extend(citations)
                state.qa_history.append(
                    QAHistoryEntry(
                        question=question, answer=answer, citations=citations
                    )
                )

            if citations:
                answer += "\n\n" + format_citations(citations)

            return answer

        tools["ask"] = ask

    if "research" in tool_names:

        async def research(ctx: RunContext[SkillRunDeps], question: str) -> str:
            """Conduct deep multi-agent research on a question.

            Iteratively searches, analyzes, and synthesizes information from the
            knowledge base to produce a comprehensive research report.
            Only use when the user explicitly requests deep research.

            Args:
                question: The research question to investigate.
            """
            state = _get_state(ctx, state_type)
            formatted, title, executive_summary = await skill_research(
                db_path,
                config,
                question,
                document_filter=state.document_filter if state else None,
            )

            if state:
                state.reports.append(
                    ResearchEntry(
                        question=question,
                        title=title,
                        executive_summary=executive_summary,
                    )
                )
                state.qa_history.append(
                    QAHistoryEntry(question=question, answer=executive_summary)
                )

            return formatted

        tools["research"] = research

    if "analyze" in tool_names:

        async def analyze(
            ctx: RunContext[SkillRunDeps],
            question: str,
            document: str | None = None,
            filter: str | None = None,
        ) -> str:
            """Answer complex analytical questions using code execution.

            Use this for questions requiring computation, aggregation, or
            data traversal across documents.

            Args:
                question: The question to answer.
                document: Optional document ID or title to pre-load for analysis.
                filter: Optional SQL WHERE clause to filter documents.
            """
            state = _get_state(ctx, state_type)
            state_filter = state.document_filter if state else None
            effective_filter = combine_filters(state_filter, filter)
            output, answer, program = await skill_analyze(
                db_path, config, question, document=document, filter=effective_filter
            )
            if state:
                state.analyses.append(
                    AnalysisEntry(
                        question=question,
                        answer=answer,
                        program=program,
                    )
                )

            return output

        tools["analyze"] = analyze

    return tools
