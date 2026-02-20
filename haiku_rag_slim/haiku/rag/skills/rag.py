import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_ai import RunContext

from haiku.rag.store.models.chunk import SearchResult
from haiku.rag.tools.document import DocumentInfo
from haiku.rag.tools.qa import QAHistoryEntry
from haiku.skills.models import Skill, SkillSource
from haiku.skills.parser import parse_skill_md
from haiku.skills.state import SkillRunDeps


class ResearchEntry(BaseModel):
    question: str
    title: str
    executive_summary: str


class RAGState(BaseModel):
    citations: list[Any] = []
    qa_history: list[QAHistoryEntry] = []
    document_filter: str | None = None
    searches: dict[str, list[SearchResult]] = {}
    documents: list[DocumentInfo] = []
    reports: list[ResearchEntry] = []


def create_skill(
    db_path: Path | None = None,
    config: Any = None,
) -> Skill:
    """Create a RAG skill for searching and analyzing documents.

    Args:
        db_path: Path to the LanceDB database. Resolved from:
            1. This argument
            2. HAIKU_RAG_DB environment variable
            3. haiku.rag default (config.storage.data_dir / "haiku.rag.lancedb")
        config: haiku.rag AppConfig instance. If None, uses get_config().
    """
    from haiku.rag.config import get_config

    if config is None:
        config = get_config()

    if db_path is None:
        env_db = os.environ.get("HAIKU_RAG_DB")
        if env_db:
            db_path = Path(env_db).expanduser()
        else:
            db_path = config.storage.data_dir / "haiku.rag.lancedb"

    path = Path(__file__).parent / "rag"
    metadata, instructions = parse_skill_md(path / "SKILL.md")

    async def _find_relevant_prior_qa(
        state: RAGState, query: str
    ) -> list[QAHistoryEntry]:
        from haiku.rag.embeddings import get_embedder
        from haiku.rag.tools.qa import PRIOR_ANSWER_RELEVANCE_THRESHOLD
        from haiku.rag.utils import cosine_similarity

        if not state.qa_history:
            return []

        embedder = get_embedder(config)
        query_embedding = await embedder.embed_query(query)

        to_embed = []
        to_embed_indices = []
        for i, qa in enumerate(state.qa_history):
            if qa.question_embedding is None:
                to_embed.append(qa.question)
                to_embed_indices.append(i)

        if to_embed:
            new_embeddings = await embedder.embed_documents(to_embed)
            for i, idx in enumerate(to_embed_indices):
                state.qa_history[idx].question_embedding = new_embeddings[i]

        matches = []
        for qa in state.qa_history:
            if qa.question_embedding is not None:
                similarity = cosine_similarity(query_embedding, qa.question_embedding)
                if similarity >= PRIOR_ANSWER_RELEVANCE_THRESHOLD:
                    matches.append(qa)

        return matches

    async def search(
        ctx: RunContext[SkillRunDeps], query: str, limit: int | None = None
    ) -> str:
        """Search the knowledge base using hybrid search (vector + full-text).

        Returns ranked results with content and metadata.

        Args:
            query: The search query.
            limit: Maximum number of results.
        """
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(db_path, config=config, read_only=True) as rag:
            results = await rag.search(query, limit=limit)
            results = await rag.expand_context(results)

        if ctx.deps and ctx.deps.state and isinstance(ctx.deps.state, RAGState):
            ctx.deps.state.searches[query] = list(results)

        return "\n\n---\n\n".join(
            r.format_for_agent(rank=i + 1, total=len(results))
            for i, r in enumerate(results)
        )

    async def list_documents(
        ctx: RunContext[SkillRunDeps],
        limit: int | None = None,
        offset: int | None = None,
        filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """List documents in the knowledge base with optional pagination and filtering.

        Args:
            limit: Maximum number of documents to return.
            offset: Number of documents to skip.
            filter: Optional SQL WHERE clause to filter documents.
        """
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(db_path, config=config, read_only=True) as rag:
            documents = await rag.list_documents(limit, offset, filter)
            result = [
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

        if ctx.deps and ctx.deps.state and isinstance(ctx.deps.state, RAGState):
            for doc_dict in result:
                doc_info = DocumentInfo(
                    id=str(doc_dict["id"]),
                    title=doc_dict["title"] or "Untitled",
                    uri=doc_dict.get("uri") or "",
                    created=doc_dict.get("created_at", ""),
                )
                if not any(d.id == doc_info.id for d in ctx.deps.state.documents):
                    ctx.deps.state.documents.append(doc_info)

        return result

    async def get_document(
        ctx: RunContext[SkillRunDeps], query: str
    ) -> dict[str, Any] | None:
        """Retrieve a document by ID, title, or URI.

        Args:
            query: Document ID, title, or URI to look up.
        """
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(db_path, config=config, read_only=True) as rag:
            document = await rag.resolve_document(query)
            if document is None:
                return None
            result = {
                "id": document.id,
                "content": document.content,
                "title": document.title,
                "uri": document.uri,
                "metadata": document.metadata,
                "created_at": str(document.created_at),
                "updated_at": str(document.updated_at),
            }

        if ctx.deps and ctx.deps.state and isinstance(ctx.deps.state, RAGState):
            doc_info = DocumentInfo(
                id=str(result["id"]),
                title=result["title"] or "Untitled",
                uri=result.get("uri") or "",
                created=result.get("created_at", ""),
            )
            if not any(d.id == doc_info.id for d in ctx.deps.state.documents):
                ctx.deps.state.documents.append(doc_info)

        return result

    async def ask(ctx: RunContext[SkillRunDeps], question: str) -> str:
        """Ask a question and get an answer with citations from the knowledge base.

        Args:
            question: The question to ask.
        """
        from haiku.rag.client import HaikuRAG
        from haiku.rag.utils import format_citations

        state = (
            ctx.deps.state
            if ctx.deps and ctx.deps.state and isinstance(ctx.deps.state, RAGState)
            else None
        )

        ask_question = question
        if state:
            matches = await _find_relevant_prior_qa(state, question)
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
            answer, citations = await rag.ask(ask_question)

        if ctx.deps and ctx.deps.state and isinstance(ctx.deps.state, RAGState):
            next_index = len(ctx.deps.state.citations) + 1
            for citation in citations:
                citation.index = next_index
                next_index += 1
            ctx.deps.state.citations.extend(citations)
            ctx.deps.state.qa_history.append(
                QAHistoryEntry(question=question, answer=answer, citations=citations)
            )

        if citations:
            answer += "\n\n" + format_citations(citations)

        return answer

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
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(db_path, config=config, read_only=True) as rag:
            documents = [document] if document else None
            result = await rag.rlm(question, documents=documents, filter=filter)
            output = result.answer
            if result.program:
                output += f"\n\nProgram:\n{result.program}"

        if ctx.deps and ctx.deps.state and isinstance(ctx.deps.state, RAGState):
            ctx.deps.state.qa_history.append(
                QAHistoryEntry(question=question, answer=output)
            )

        return output

    async def research(ctx: RunContext[SkillRunDeps], question: str) -> str:
        """Conduct deep multi-agent research on a question.

        Iteratively searches, analyzes, and synthesizes information from the
        knowledge base to produce a comprehensive research report.
        Only use when the user explicitly requests deep research.

        Args:
            question: The research question to investigate.
        """
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(db_path, config=config, read_only=True) as rag:
            report = await rag.research(question)

        if ctx.deps and ctx.deps.state and isinstance(ctx.deps.state, RAGState):
            ctx.deps.state.reports.append(
                ResearchEntry(
                    question=question,
                    title=report.title,
                    executive_summary=report.executive_summary,
                )
            )
            ctx.deps.state.qa_history.append(
                QAHistoryEntry(question=question, answer=report.executive_summary)
            )

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

        return "\n".join(parts)

    return Skill(
        metadata=metadata,
        source=SkillSource.ENTRYPOINT,
        path=path,
        instructions=instructions,
        tools=[
            search,
            list_documents,
            get_document,
            ask,
            analyze,
            research,
        ],
        state_type=RAGState,
        state_namespace="rag",
    )
