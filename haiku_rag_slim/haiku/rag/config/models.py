from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from haiku.rag.utils import get_default_data_dir


class ModelConfig(BaseModel):
    """Configuration for a language model.

    Attributes:
        provider: Model provider (ollama, openai, anthropic, etc.)
        name: Model name/identifier
        base_url: Optional base URL for OpenAI-compatible servers (vLLM, LM Studio, etc.)
        enable_thinking: Control reasoning behavior (true/false/None for default)
        temperature: Sampling temperature (0.0 to 1.0+)
        max_tokens: Maximum tokens to generate
    """

    provider: str = "ollama"
    name: str = "gpt-oss"
    base_url: str | None = None

    enable_thinking: bool | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class EmbeddingModelConfig(BaseModel):
    """Configuration for an embedding model.

    Attributes:
        provider: Model provider (ollama, openai, voyageai, cohere, sentence-transformers)
        name: Model name/identifier
        vector_dim: Vector dimensions produced by the model
        base_url: Optional base URL for OpenAI-compatible servers (vLLM, LM Studio, etc.)
    """

    provider: str = "ollama"
    name: str = "qwen3-embedding:4b"
    vector_dim: int = 2560
    base_url: str | None = None


class MultimodalEmbeddingModelConfig(BaseModel):
    """Configuration for a multimodal embedding model (text + image).

    This is used for image-to-image and text-to-image retrieval.

    Notes:
    - This feature is opt-in and may require external services (e.g., vLLM).
    - Keep vector_dim stable for a database once created, just like text embeddings.
    """

    provider: Literal["vllm"] = "vllm"
    name: str = "Qwen/Qwen3-VL-Embedding-2B"
    vector_dim: int = 2048

    # Base URL for the multimodal embedding service.
    # vLLM default is typically http://localhost:8000
    base_url: str = "http://localhost:8000"

    # HTTP client settings
    timeout: int = 60

    # Optional: request server-side output dimension (if the backend supports it).
    # For OpenAI-compatible /v1/embeddings this is usually called "dimensions".
    #
    # Note: Some servers/models (including vLLM + Qwen3-VL-Embedding-2B) reject the
    # presence of this field unless the model explicitly supports matryoshka output.
    # Keep this unset by default; haiku will validate the returned dimension via
    # `vector_dim`.
    dimensions: int | None = None

    # OpenAI-compatible embeddings response format.
    # vLLM/OpenAI supports "float"; some servers also support "base64".
    encoding_format: Literal["float", "base64"] = "float"


class MultimodalConfig(BaseModel):
    """Opt-in multimodal embedding configuration."""

    enabled: bool = False
    index_pictures: bool = True

    # Crop padding (in pixels) for bbox-based image crops.
    # Kept here (rather than conversion options) because it affects embedding, not conversion.
    image_crop_padding_px: int = 8

    # Image resize guardrail before embedding (to keep payload sizes manageable and avoid
    # backend limits). Set to 0 to disable resizing.
    image_max_side_px: int = 1024

    # Max number of images to send per /v1/embeddings call for mm_assets indexing.
    # (Large PDFs can contain many figures; batching avoids oversized requests.)
    embed_batch_size: int = 8

    model: MultimodalEmbeddingModelConfig = Field(
        default_factory=MultimodalEmbeddingModelConfig
    )


class StorageConfig(BaseModel):
    data_dir: Path = Field(default_factory=get_default_data_dir)
    auto_vacuum: bool = True
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
    model: EmbeddingModelConfig = Field(default_factory=EmbeddingModelConfig)


class RerankingConfig(BaseModel):
    model: ModelConfig | None = None


class QAConfig(BaseModel):
    model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            provider="ollama",
            name="gpt-oss",
            enable_thinking=False,
        )
    )
    max_sub_questions: int = 3
    max_iterations: int = 2
    max_concurrency: int = 1


class ResearchConfig(BaseModel):
    model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            provider="ollama",
            name="gpt-oss",
            enable_thinking=False,
        )
    )
    max_iterations: int = 3
    confidence_threshold: float = 0.8
    max_concurrency: int = 1


class PictureDescriptionConfig(BaseModel):
    """Configuration for VLM-based picture description."""

    enabled: bool = False
    model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            provider="ollama",
            name="ministral-3",
        )
    )
    timeout: int = 90
    max_tokens: int = 200


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
    generate_picture_images: bool = False

    # VLM picture description
    picture_description: PictureDescriptionConfig = Field(
        default_factory=PictureDescriptionConfig
    )


class ProcessingConfig(BaseModel):
    chunk_size: int = 256
    converter: str = "docling-local"
    chunker: str = "docling-local"
    chunker_type: str = "hybrid"
    chunking_tokenizer: str = "Qwen/Qwen3-Embedding-0.6B"
    chunking_merge_peers: bool = True
    chunking_use_markdown_tables: bool = False
    conversion_options: ConversionOptions = Field(default_factory=ConversionOptions)


class SearchConfig(BaseModel):
    limit: int = 5
    context_radius: int = 0
    max_context_items: int = 10
    max_context_chars: int = 10000
    vector_index_metric: Literal["cosine", "l2", "dot"] = "cosine"
    vector_refine_factor: int = 30


class OllamaConfig(BaseModel):
    base_url: str = Field(
        default_factory=lambda: __import__("os").environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
    )


class DoclingServeConfig(BaseModel):
    base_url: str = "http://localhost:5001"
    api_key: str = ""


class ProvidersConfig(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    docling_serve: DoclingServeConfig = Field(default_factory=DoclingServeConfig)


class AGUIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]
    cors_credentials: bool = True
    cors_methods: list[str] = ["GET", "POST", "OPTIONS"]
    cors_headers: list[str] = ["*"]


class PromptsConfig(BaseModel):
    domain_preamble: str = ""
    qa: str | None = None
    synthesis: str | None = None
    picture_description: str = (
        "Describe this image for a blind user. "
        "State the image type (screenshot, chart, photo, etc.), "
        "what it depicts, any visible text, and key visual details. "
        "Be concise and accurate."
    )


class AppConfig(BaseModel):
    environment: str = "production"
    storage: StorageConfig = Field(default_factory=StorageConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    lancedb: LanceDBConfig = Field(default_factory=LanceDBConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    multimodal: MultimodalConfig = Field(default_factory=MultimodalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    qa: QAConfig = Field(default_factory=QAConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    agui: AGUIConfig = Field(default_factory=AGUIConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
