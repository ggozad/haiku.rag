import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_ai import RunContext

from haiku.skills.models import Skill, SkillSource
from haiku.skills.parser import parse_skill_md
from haiku.skills.state import SkillRunDeps


class AnalysisEntry(BaseModel):
    question: str
    answer: str
    program: str | None = None


class RLMState(BaseModel):
    analyses: list[AnalysisEntry] = []


def create_skill(
    db_path: Path | None = None,
    config: Any = None,
) -> Skill:
    """Create an RLM analysis skill for computational document analysis.

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

    path = Path(__file__).parent / "rag-rlm"
    metadata, instructions = parse_skill_md(path / "SKILL.md")

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

        if ctx.deps and ctx.deps.state and isinstance(ctx.deps.state, RLMState):
            ctx.deps.state.analyses.append(
                AnalysisEntry(
                    question=question,
                    answer=result.answer,
                    program=result.program,
                )
            )

        return output

    return Skill(
        metadata=metadata,
        source=SkillSource.ENTRYPOINT,
        path=path,
        instructions=instructions,
        tools=[analyze],
        state_type=RLMState,
        state_namespace="rlm",
    )
