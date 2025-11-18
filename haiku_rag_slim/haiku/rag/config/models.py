from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from haiku.rag.utils import get_default_data_dir


class StorageConfig(BaseModel):
    data_dir: Path = Field(default_factory=get_default_data_dir)
    vacuum_retention_seconds: int = 86400


class MonitorConfig(BaseModel):
    directories: list[Path] = []
    ignore_patterns: list[str] = []
    include_patterns: list[str] = []
    delete_orphans: bool = False


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
    max_sub_questions: int = 3
    max_iterations: int = 2
    max_concurrency: int = 1


class ResearchConfig(BaseModel):
    provider: str = "ollama"
    model: str = "gpt-oss"
    max_iterations: int = 3
    confidence_threshold: float = 0.8
    max_concurrency: int = 1


class ConversionOptions(BaseModel):
    """Options for document conversion."""

    # OCR options
    do_ocr: bool = True
    force_ocr: bool = False
    ocr_lang: list[str] = []

    # Table options
    do_table_structure: bool = True
    table_mode: Literal["fast", "accurate"] = "accurate"
    table_cell_matching: bool = True

    # Image options
    images_scale: float = 2.0


class ProcessingConfig(BaseModel):
    chunk_size: int = 256
    context_chunk_radius: int = 0
    markdown_preprocessor: str = ""
    converter: str = "docling-local"
    chunker: str = "docling-local"
    chunker_type: str = "hybrid"
    chunking_tokenizer: str = "Qwen/Qwen3-Embedding-0.6B"
    chunking_merge_peers: bool = True
    chunking_use_markdown_tables: bool = False
    conversion_options: ConversionOptions = Field(default_factory=ConversionOptions)


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


class DoclingServeConfig(BaseModel):
    base_url: str = "http://localhost:5001"
    api_key: str = ""
    timeout: int = 300


class ProvidersConfig(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    docling_serve: DoclingServeConfig = Field(default_factory=DoclingServeConfig)


class AGUIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]
    cors_credentials: bool = True
    cors_methods: list[str] = ["GET", "POST", "OPTIONS"]
    cors_headers: list[str] = ["*"]


class AppConfig(BaseModel):
    environment: str = "production"
    storage: StorageConfig = Field(default_factory=StorageConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    lancedb: LanceDBConfig = Field(default_factory=LanceDBConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    qa: QAConfig = Field(default_factory=QAConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    agui: AGUIConfig = Field(default_factory=AGUIConfig)
