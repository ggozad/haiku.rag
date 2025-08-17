import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

from haiku.rag.utils import get_default_data_dir

load_dotenv()


class AppConfig(BaseModel):
    ENV: str = "production"

    DEFAULT_DATA_DIR: Path = get_default_data_dir()
    MONITOR_DIRECTORIES: list[Path] = []

    EMBEDDINGS_PROVIDER: str = "ollama"
    EMBEDDINGS_MODEL: str = "mxbai-embed-large"
    EMBEDDINGS_VECTOR_DIM: int = 1024

    RERANK_PROVIDER: str = "ollama"
    RERANK_MODEL: str = "qwen3"

    QA_PROVIDER: str = "ollama"
    QA_MODEL: str = "qwen3"

    CHUNK_SIZE: int = 256
    CONTEXT_CHUNK_RADIUS: int = 0

    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Provider keys
    VOYAGE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    COHERE_API_KEY: str = ""

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
