from pathlib import Path
from typing import Annotated, Literal

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
        vision: True if the model can interpret images. Default False.
        extra_body: Raw dict forwarded verbatim to the model SDK as
            `ModelSettings.extra_body`. Provider-side escape hatch for
            keys haiku.rag doesn't model explicitly (e.g. vLLM's
            `chat_template_kwargs.enable_thinking: false` for Qwen3).
            Honored by openai/ollama/anthropic/groq; ignored by gemini/bedrock.
    """

    provider: str = "ollama"
    name: str = "gpt-oss"
    base_url: str | None = None

    enable_thinking: bool | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    vision: bool = False
    extra_body: dict | None = None


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


class StorageConfig(BaseModel):
    data_dir: Path = Field(default_factory=get_default_data_dir)
    auto_vacuum: bool = True
    vacuum_retention_seconds: int = 86400


class LanceDBConfig(BaseModel):
    uri: str = ""
    api_key: str = ""
    region: str = ""
    storage_options: dict[str, str] = Field(default_factory=dict)


class EmbeddingsConfig(BaseModel):
    model: EmbeddingModelConfig = Field(default_factory=EmbeddingModelConfig)
    batch_size: int = 512


class RerankingConfig(BaseModel):
    model: ModelConfig | None = None


class QAConfig(BaseModel):
    model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            provider="ollama",
            name="gpt-oss",
            enable_thinking=True,
            temperature=0.3,
        )
    )
    max_searches: int = 5


class AnalysisConfig(BaseModel):
    """Driving model + sandbox limits for the analysis skill.

    ``model`` defaults to ``None``, meaning "no override — use ``qa.model``."
    Consumers resolve via ``config.analysis.model or config.qa.model``. Set
    explicitly when the analysis workload wants a different model from QA
    (e.g. a stronger model for computational tasks)."""

    model: ModelConfig | None = None
    code_timeout: float = 60.0
    max_output_chars: int = 50_000


class PictureDescriptionConfig(BaseModel):
    """How the VLM runs over each picture when it runs at all.

    Activation lives on ``ProcessingConfig.pictures`` — these fields only
    describe *how* the VLM runs once ``pictures == "description"``.
    """

    model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            provider="ollama",
            name="ministral-3",
            temperature=0.0,
        )
    )
    timeout: int = 90
    max_tokens: int = 200


class ConversionOptions(BaseModel):
    """Options for document conversion."""

    # OCR options
    do_ocr: bool = True
    force_ocr: bool = False
    ocr_engine: Literal[
        "auto", "easyocr", "ocrmac", "rapidocr", "tesserocr", "tesseract"
    ] = "auto"
    ocr_lang: list[str] = []

    # Table options
    do_table_structure: bool = True
    table_mode: Literal["fast", "accurate"] = "accurate"
    table_cell_matching: bool = True

    # Image options
    images_scale: float = 2.0
    generate_page_images: bool = True

    # Fetch images referenced by URL in HTML and Markdown inputs.
    # docling-local only — docling-serve cannot fetch external images.
    fetch_remote_images: bool = True

    picture_description: PictureDescriptionConfig = Field(
        default_factory=PictureDescriptionConfig
    )


PicturesMode = Literal["none", "description", "image"]


class ProcessingConfig(BaseModel):
    chunk_size: int = 256
    converter: str = "docling-local"
    chunker: str = "docling-local"
    chunker_type: str = "hybrid"
    chunking_tokenizer: str = "Qwen/Qwen3-Embedding-0.6B"
    chunking_merge_peers: bool = True
    chunking_use_markdown_tables: bool = False
    conversion_options: ConversionOptions = Field(default_factory=ConversionOptions)
    pictures: PicturesMode = "image"
    """How embedded pictures are handled at ingest.

    - ``"none"``: docling skips picture-image generation; ``label="picture"``
      rows still exist as structure but carry no bytes or description. Use
      this when you don't need picture content and want to keep RAM and DB
      size low on large documents.
    - ``"description"``: docling generates picture images, the configured
      VLM produces text descriptions woven into chunk text, AND the bytes
      are retained in ``document_items.picture_data`` so a vision-capable
      QA model or multimodal embedder can be enabled later without
      reingesting.
    - ``"image"``: docling generates picture images and stores them in
      ``document_items.picture_data``; no VLM runs at ingest.
    """
    auto_title: bool = False
    title_model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            provider="ollama",
            name="gpt-oss",
            enable_thinking=False,
            temperature=0.3,
            max_tokens=100,
        )
    )


class SearchConfig(BaseModel):
    limit: int = 5
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


class PromptsConfig(BaseModel):
    domain_preamble: str = ""
    picture_description: str = (
        "Describe this image for a blind user. "
        "State the image type (screenshot, chart, photo, etc.), "
        "what it depicts, any visible text, and key visual details. "
        "Be concise and accurate."
    )


class EvaluationsConfig(BaseModel):
    """Settings consumed only by the `evaluations` package."""

    judge: ModelConfig | None = Field(
        default=None,
        description=(
            "Judge model for `evaluations run`'s LLM-as-judge step. "
            "ModelConfig's base_url lets the judge point at any "
            "OpenAI-compatible endpoint."
        ),
    )


class QueueConfig(BaseModel):
    """SQLite queue for the production ingester."""

    path: Path = Field(
        default_factory=lambda: get_default_data_dir() / "ingester.db",
        description="Location of the ingester's SQLite queue file.",
    )


class RetryPolicyConfig(BaseModel):
    """Per-job retry policy. Per-source override is allowed under
    SourceConfig.retry so a flaky source doesn't drag the rest of the queue."""

    max_attempts: int = 5
    base_delay_s: float = 2.0
    max_delay_s: float = 300.0
    jitter: float = Field(default=0.25, ge=0.0, le=1.0)


class CircuitBreakerConfig(BaseModel):
    """Per-source breaker over discover() failures. Stops the ingester from
    hammering a source that's persistently failing."""

    failure_threshold: int = Field(
        default=5, description="Consecutive failures before the breaker opens."
    )
    cooldown_s: float = Field(
        default=600.0,
        description="How long the breaker stays open before allowing a probe.",
    )


class WorkerConfig(BaseModel):
    worker_count: int = 4
    max_concurrent: int = 4
    poll_idle_interval_s: float = 1.0
    claim_timeout_s: int = 1800
    reaper_interval_s: int = 60
    retry: RetryPolicyConfig = Field(default_factory=RetryPolicyConfig)


class APIConfig(BaseModel):
    """HTTP control plane settings for the ingester."""

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8765
    auth_token: str | None = None


class _SourceBase(BaseModel):
    """Fields common to every source. `id` is optional; if omitted the source
    derives a deterministic id from its target (root path / bucket+prefix /
    user-supplied tag)."""

    id: str | None = None
    delete_orphans: bool = True
    poll_interval_s: float = Field(
        default=300.0,
        description="How often discover() runs. FS additionally uses watchfiles "
        "for push events between sweeps.",
    )
    retry: RetryPolicyConfig | None = Field(
        default=None,
        description="Override the worker's default retry policy for jobs from "
        "this source. None = inherit from WorkerConfig.retry.",
    )
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)


class FSSourceConfig(_SourceBase):
    type: Literal["fs"]
    root: Path
    ignore_patterns: list[str] = []
    include_patterns: list[str] = []


class HTTPSourceConfig(_SourceBase):
    type: Literal["http"]
    urls: list[str] = []
    headers: dict[str, str] = Field(default_factory=dict)


class S3SourceConfig(_SourceBase):
    type: Literal["s3"]
    uri: str
    storage_options: dict[str, str] = Field(default_factory=dict)
    ignore_patterns: list[str] = []
    include_patterns: list[str] = []


SourceConfig = Annotated[
    FSSourceConfig | HTTPSourceConfig | S3SourceConfig,
    Field(discriminator="type"),
]


class IngesterConfig(BaseModel):
    """Production ingester settings."""

    sources: list[SourceConfig] = []
    queue: QueueConfig = Field(default_factory=QueueConfig)
    workers: WorkerConfig = Field(default_factory=WorkerConfig)
    api: APIConfig = Field(default_factory=APIConfig)


class AppConfig(BaseModel):
    environment: str = "production"
    storage: StorageConfig = Field(default_factory=StorageConfig)
    lancedb: LanceDBConfig = Field(default_factory=LanceDBConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    qa: QAConfig = Field(default_factory=QAConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    ingester: IngesterConfig = Field(default_factory=IngesterConfig)
    evaluations: "EvaluationsConfig" = Field(
        default_factory=lambda: EvaluationsConfig()
    )
