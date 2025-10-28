from pathlib import Path

from pydantic import BaseModel, Field

from haiku.rag.utils import get_default_data_dir


class StorageConfig(BaseModel):
    data_dir: Path = Field(default_factory=get_default_data_dir)
    monitor_directories: list[Path] = []
    disable_autocreate: bool = False
    vacuum_retention_seconds: int = 60


class LanceDBConfig(BaseModel):
    uri: str = ""
    api_key: str = ""
    region: str = ""


class EmbeddingsConfig(BaseModel):
    provider: str = "ollama"
    model: str = "qwen3-embedding"
    vector_dim: int = 4096


class RerankingConfig(BaseModel):
    provider: str = ""
    model: str = ""


class QAConfig(BaseModel):
    provider: str = "ollama"
    model: str = "gpt-oss"


class ResearchConfig(BaseModel):
    provider: str = "ollama"
    model: str = "gpt-oss"


class ProcessingConfig(BaseModel):
    chunk_size: int = 256
    context_chunk_radius: int = 0
    markdown_preprocessor: str = ""


class OllamaConfig(BaseModel):
    base_url: str = Field(
        default_factory=lambda: __import__("os").environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
    )


class VLLMConfig(BaseModel):
    embeddings_base_url: str = ""
    rerank_base_url: str = ""
    qa_base_url: str = ""
    research_base_url: str = ""


class ProvidersConfig(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)


class A2AConfig(BaseModel):
    max_contexts: int = 1000


class AppConfig(BaseModel):
    environment: str = "production"
    storage: StorageConfig = Field(default_factory=StorageConfig)
    lancedb: LanceDBConfig = Field(default_factory=LanceDBConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    qa: QAConfig = Field(default_factory=QAConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    a2a: A2AConfig = Field(default_factory=A2AConfig)
