import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

from haiku.rag.utils import get_default_data_dir

load_dotenv()


class AppConfig(BaseModel):
    ENV: str = "production"

    LANCEDB_API_KEY: str = ""
    LANCEDB_URI: str = ""
    LANCEDB_REGION: str = ""

    DEFAULT_DATA_DIR: Path = get_default_data_dir()
    MONITOR_DIRECTORIES: list[Path] = []

    EMBEDDINGS_PROVIDER: str = "ollama"
    EMBEDDINGS_MODEL: str = "mxbai-embed-large"
    EMBEDDINGS_VECTOR_DIM: int = 1024

    RERANK_PROVIDER: str = ""
    RERANK_MODEL: str = ""

    QA_PROVIDER: str = "ollama"
    QA_MODEL: str = "gpt-oss"

    # Research defaults (fallback to QA if not provided via env)
    RESEARCH_PROVIDER: str = "ollama"
    RESEARCH_MODEL: str = "gpt-oss"

    CHUNK_SIZE: int = 256
    CONTEXT_CHUNK_RADIUS: int = 0

    # Optional dotted path or file path to a callable that preprocesses
    # markdown content before chunking. Examples:
    MARKDOWN_PREPROCESSOR: str = ""

    OLLAMA_BASE_URL: str = "http://localhost:11434"

    VLLM_EMBEDDINGS_BASE_URL: str = ""
    VLLM_RERANK_BASE_URL: str = ""
    VLLM_QA_BASE_URL: str = ""
    VLLM_RESEARCH_BASE_URL: str = ""

    # Provider keys
    VOYAGE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    COHERE_API_KEY: str = ""

    # If true, refuse to auto-create a new LanceDB database or tables
    # and error out when the database does not already exist.
    DISABLE_DB_AUTOCREATE: bool = False

    @field_validator("MONITOR_DIRECTORIES", mode="before")
    @classmethod
    def parse_monitor_directories(cls, v):
        if isinstance(v, str):
            if not v.strip():
                return []
            return [
                Path(path.strip()).absolute() for path in v.split(",") if path.strip()
            ]
        return v


# Expose Config object for app to import
Config = AppConfig.model_validate(os.environ)
if Config.OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
if Config.VOYAGE_API_KEY:
    os.environ["VOYAGE_API_KEY"] = Config.VOYAGE_API_KEY
if Config.ANTHROPIC_API_KEY:
    os.environ["ANTHROPIC_API_KEY"] = Config.ANTHROPIC_API_KEY
if Config.COHERE_API_KEY:
    os.environ["CO_API_KEY"] = Config.COHERE_API_KEY
