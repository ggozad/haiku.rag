import os
from functools import cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_ai import RunContext

from haiku.skills.models import Skill, SkillMetadata, SkillSource, StateMetadata
from haiku.skills.parser import parse_skill_md
from haiku.skills.state import SkillRunDeps


class AnalysisEntry(BaseModel):
    question: str
    answer: str
    program: str | None = None


class RLMState(BaseModel):
    analyses: list[AnalysisEntry] = []


STATE_TYPE = RLMState
STATE_NAMESPACE = "rlm"

_skill_path = Path(__file__).parent / "rag-rlm"


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
        metadata=skill_metadata(),
        source=SkillSource.ENTRYPOINT,
        path=_skill_path,
        instructions=instructions(),
        tools=[analyze],
        state_type=STATE_TYPE,
        state_namespace=STATE_NAMESPACE,
    )
