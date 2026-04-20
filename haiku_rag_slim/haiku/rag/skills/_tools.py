from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_ai import RunContext

from haiku.rag.agents.research.models import Citation
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models.chunk import SearchResult
from haiku.skills.state import SkillRunDeps


class CodeExecutionEntry(BaseModel):
    code: str
    stdout: str
    stderr: str = ""
    success: bool = True


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
    filter: str | None = None,
) -> list[dict[str, Any]]:
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(db_path, config=config, read_only=True) as rag:
        documents = await rag.list_documents(filter=filter)
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


def _get_state(ctx: RunContext[SkillRunDeps], state_type: type[BaseModel]) -> Any:
    if ctx.deps and ctx.deps.state and isinstance(ctx.deps.state, state_type):
        return ctx.deps.state
    return None


def _register_citations(state: Any, citations: "list[Citation]") -> None:
    """Add citations to the index and record the turn's chunk IDs."""
    chunk_ids = []
    next_index = len(state.citation_index) + 1
    for citation in citations:
        cid = citation.chunk_id
        if cid not in state.citation_index:
            citation.index = next_index
            next_index += 1
            state.citation_index[cid] = citation
        chunk_ids.append(cid)
    state.citations.append(chunk_ids)


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
        max_searches = config.qa.max_searches
        search_counts: dict[str, int] = {}

        async def search(
            ctx: RunContext[SkillRunDeps], query: str, limit: int | None = None
        ) -> str:
            """Search the knowledge base using hybrid search (vector + full-text).

            Returns ranked results with content and metadata.

            Args:
                query: The search query.
                limit: Maximum number of results.
            """
            rid = ctx.run_id or ""
            search_counts[rid] = search_counts.get(rid, 0) + 1
            if search_counts[rid] > max_searches:
                return (
                    "Search limit reached. Answer the question using "
                    "the results you already have."
                )

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
        ) -> list[dict[str, Any]]:
            """List all documents in the knowledge base."""
            state = _get_state(ctx, state_type)
            result = await skill_list_documents(
                db_path,
                config,
                filter=state.document_filter if state else None,
            )
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
            return await skill_get_document(db_path, config, query)

        tools["get_document"] = get_document

    if "execute_code" in tool_names:

        async def execute_code(ctx: RunContext[SkillRunDeps], code: str) -> str:
            """Execute Python code in a sandboxed interpreter.

            The code has access to search(), list_documents(), llm() functions
            and a virtual filesystem at /documents/ with document content and
            structure (metadata.json, content.txt, items.jsonl per document).

            Use print() to output results. Each call runs in a fresh
            interpreter — variables do not persist between calls.

            Args:
                code: Python code to execute.
            """
            from haiku.rag.agents.analysis.dependencies import AnalysisContext
            from haiku.rag.agents.analysis.sandbox import Sandbox

            state = _get_state(ctx, state_type)
            doc_filter = state.document_filter if state else None
            context = AnalysisContext(filter=doc_filter)
            sandbox = Sandbox(db_path=db_path, config=config, context=context)
            result = await sandbox.execute(code)

            state = _get_state(ctx, state_type)
            if state and sandbox._search_results:
                existing = state.searches.get("_sandbox", [])
                seen = {r.chunk_id for r in existing}
                for sr in sandbox._search_results:
                    if sr.chunk_id not in seen:
                        existing.append(sr)
                        seen.add(sr.chunk_id)
                state.searches["_sandbox"] = existing

            if state:
                state.executions.append(
                    CodeExecutionEntry(
                        code=code,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        success=result.success,
                    )
                )

            if result.success:
                return result.stdout if result.stdout else "No output."
            return f"Error: {result.stderr}\n\nOutput: {result.stdout}"

        tools["execute_code"] = execute_code

    if "cite" in tool_names:

        async def cite(ctx: RunContext[SkillRunDeps], chunk_ids: list[str]) -> str:
            """Register chunk IDs as citations for your answer.

            Call this after searching, with the chunk_id values from search
            results that support your answer.

            Args:
                chunk_ids: List of chunk_id values from search results.
            """
            from haiku.rag.agents.research.models import resolve_citations

            state = _get_state(ctx, state_type)
            if not state:
                return "No state available."

            all_results = []
            for results_list in state.searches.values():
                all_results.extend(results_list)

            citations = resolve_citations(chunk_ids, all_results)
            if citations:
                _register_citations(state, citations)
            return f"Registered {len(citations)} citation(s)."

        tools["cite"] = cite

    return tools
