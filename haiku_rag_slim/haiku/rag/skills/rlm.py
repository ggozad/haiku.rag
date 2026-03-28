import os
from functools import cache
from pathlib import Path

from pydantic import BaseModel

from haiku.rag.config.models import AppConfig
from haiku.rag.skills._tools import AnalysisEntry
from haiku.skills.models import Skill, SkillMetadata, SkillSource, StateMetadata
from haiku.skills.parser import parse_skill_md


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
    config: AppConfig | None = None,
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
    from haiku.rag.skills._tools import create_skill_extras, create_skill_tools

    if config is None:
        config = get_config()

    if db_path is None:
        env_db = os.environ.get("HAIKU_RAG_DB")
        if env_db:
            db_path = Path(env_db).expanduser()
        else:
            db_path = config.storage.data_dir / "haiku.rag.lancedb"

    tools = create_skill_tools(db_path, config, RLMState, ["analyze"])
    extras = create_skill_extras(db_path, config)

    return Skill(
        metadata=skill_metadata(),
        source=SkillSource.ENTRYPOINT,
        path=_skill_path,
        instructions=instructions(),
        tools=list(tools.values()),
        extras=extras,
        state_type=STATE_TYPE,
        state_namespace=STATE_NAMESPACE,
    )
