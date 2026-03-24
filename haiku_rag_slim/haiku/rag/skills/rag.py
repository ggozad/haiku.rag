import os
from functools import cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from haiku.rag.agents.research.models import Citation
from haiku.rag.store.models.chunk import SearchResult
from haiku.rag.tools.document import DocumentInfo
from haiku.rag.tools.qa import QAHistoryEntry
from haiku.skills.models import Skill, SkillMetadata, SkillSource, StateMetadata
from haiku.skills.parser import parse_skill_md
from haiku.skills.state import SkillRunDeps

AGENT_PREAMBLE = """You are a helpful research assistant powered by haiku.rag, a knowledge base system.

CRITICAL RULES:
1. For greetings or casual chat: respond directly WITHOUT using any tools
2. NEVER make up information - always use skills to get facts from the knowledge base
3. When a skill returns citations, always include them in your response
"""


class ResearchEntry(BaseModel):
    question: str
    title: str
    executive_summary: str


class RAGState(BaseModel):
    citations: list[Citation] = Field(default_factory=list)
    qa_history: list[QAHistoryEntry] = Field(default_factory=list)
    document_filter: str | None = None
    searches: dict[str, list[SearchResult]] = Field(default_factory=dict)
    documents: list[DocumentInfo] = Field(default_factory=list)
    reports: list[ResearchEntry] = Field(default_factory=list)


STATE_TYPE = RAGState
STATE_NAMESPACE = "rag"

_skill_path = Path(__file__).parent / "rag"


@cache
def skill_metadata() -> SkillMetadata:
    metadata, _ = parse_skill_md(_skill_path / "SKILL.md")
    return metadata


@cache
def instructions() -> str | None:
    _, instr = parse_skill_md(_skill_path / "SKILL.md")
    return instr


def state_metadata() -> StateMetadata:
    return StateMetadata(
        namespace=STATE_NAMESPACE,
        type=STATE_TYPE,
        schema=STATE_TYPE.model_json_schema(),
    )


def _get_state(ctx: RunContext[SkillRunDeps]) -> RAGState | None:
    if ctx.deps and ctx.deps.state and isinstance(ctx.deps.state, RAGState):
        return ctx.deps.state
    return None


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

    async def search(
        ctx: RunContext[SkillRunDeps], query: str, limit: int | None = None
    ) -> str:
        """Search the knowledge base using hybrid search (vector + full-text).

        Returns ranked results with content and metadata.

        Args:
            query: The search query.
            limit: Maximum number of results.
        """
        from haiku.rag.skills._tools import skill_search

        state = _get_state(ctx)
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
        from haiku.rag.skills._tools import (
            skill_list_documents,
            update_documents_state,
        )

        result = await skill_list_documents(db_path, config, limit, offset)
        state = _get_state(ctx)
        if state:
            update_documents_state(state.documents, result)
        return result

    async def get_document(
        ctx: RunContext[SkillRunDeps], query: str
    ) -> dict[str, Any] | None:
        """Retrieve a document by ID, title, or URI.

        Args:
            query: Document ID, title, or URI to look up.
        """
        from haiku.rag.skills._tools import (
            skill_get_document,
            update_documents_state,
        )

        result = await skill_get_document(db_path, config, query)
        if result is not None:
            state = _get_state(ctx)
            if state:
                update_documents_state(state.documents, [result])
        return result

    async def ask(ctx: RunContext[SkillRunDeps], question: str) -> str:
        """Ask a question and get an answer with citations from the knowledge base.

        Args:
            question: The question to ask.
        """
        from haiku.rag.skills._tools import skill_ask
        from haiku.rag.utils import format_citations

        state = _get_state(ctx)
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
                QAHistoryEntry(question=question, answer=answer, citations=citations)
            )

        if citations:
            answer += "\n\n" + format_citations(citations)

        return answer

    async def research(ctx: RunContext[SkillRunDeps], question: str) -> str:
        """Conduct deep multi-agent research on a question.

        Iteratively searches, analyzes, and synthesizes information from the
        knowledge base to produce a comprehensive research report.
        Only use when the user explicitly requests deep research.

        Args:
            question: The research question to investigate.
        """
        from haiku.rag.skills._tools import skill_research

        state = _get_state(ctx)
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

    return Skill(
        metadata=skill_metadata(),
        source=SkillSource.ENTRYPOINT,
        path=_skill_path,
        instructions=instructions(),
        tools=[
            search,
            list_documents,
            get_document,
            ask,
            research,
        ],
        state_type=STATE_TYPE,
        state_namespace=STATE_NAMESPACE,
    )
