from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
        provider: Model provider (ollama, openai, voyageai, cohere, sentence-transformers, vllm)
        name: Model name/identifier
        vector_dim: Vector dimensions produced by the model
        base_url: Optional base URL for OpenAI-compatible servers (vLLM, LM Studio, etc.)
        multimodal: Whether the model embeds images into the same vector space as
            text. Supported on the vllm, voyageai, and cohere providers; other
            providers raise when this is set.
    """

    provider: str = "ollama"
    name: str = "qwen3-embedding:4b"
    vector_dim: int = 2560
    base_url: str | None = None
    multimodal: bool = False


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
    max_executions: int = 15


class DuplicateDetectionConfig(BaseModel):
    """Thresholds for doctor's near-duplicate document detection.

    Detection clusters documents that share most of their chunks (revisions of
    one another). ``containment_threshold`` is the decision knob; the others
    tune the cheap centroid pre-filter, what counts as a shared chunk, and which
    tiny documents to skip.
    """

    containment_threshold: float = 0.75
    candidate_threshold: float = 0.85
    twin_similarity: float = 0.95
    min_chunks: int = 3


class DoctorConfig(BaseModel):
    duplicates: DuplicateDetectionConfig = Field(
        default_factory=DuplicateDetectionConfig
    )


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
    split_pages: int = Field(
        default=0,
        ge=0,
        description=(
            "If >0, PDFs are split into N-page slices, each converted "
            "independently, and merged via DoclingDocument.concatenate. 0 "
            "disables splitting (single-pass conversion). Recommended: 10 "
            "for memory-bound or large (>100 page) PDFs."
        ),
    )
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
    extract_pdf_attachments: bool = True
    """When a PDF carries `/EmbeddedFiles`, ingest each attachment as a separate
    Document linked back to the wrapper via ``metadata.parent_uri``. Cap depth
    at 3 to bound nested-attachment recursion."""
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
    """docling-serve endpoints. Accepts a single URL or a list — when a list
    is given, the client round-robins jobs across the URLs. Each job's
    submit/poll/result trio stays on the same instance (task IDs are
    instance-local). The round-robin counter is per-process; for true load
    balancing or failover, put an LB in front."""

    base_url: str | list[str] = "http://localhost:5001"
    api_key: str = ""

    @property
    def base_urls(self) -> list[str]:
        """Always-a-list view of base_url. Empty input falls back to localhost."""
        if isinstance(self.base_url, str):
            return [self.base_url]
        return list(self.base_url) or ["http://localhost:5001"]


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
    """Job queue for the production ingester. Defaults to a filesystem SQLite
    file; set `dburi` to point it at a database server instead."""

    path: Path = Field(
        default_factory=lambda: get_default_data_dir() / "ingester.db",
        description="SQLite queue file. Used when dburi is unset.",
    )
    dburi: str | None = Field(
        default=None,
        description="SQLAlchemy async URL for the queue, e.g. "
        "postgresql+asyncpg://user:pw@host/db. Overrides path when set.",
    )
    retention_days: int | None = Field(
        default=30,
        description="Delete succeeded/dead jobs whose completed_at is older "
        "than this many days. The reaper enforces it on reaper_interval_s. "
        "None disables pruning (keep all terminal rows).",
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
    # Reject unknown keys so a renamed/removed setting (e.g. the former
    # claim_timeout_s) fails loudly instead of being silently ignored.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    worker_count: int = Field(
        default=4,
        description="Number of async worker tasks pulling from the queue. "
        "Each worker holds at most one job at a time, so worker_count is "
        "also the maximum number of concurrent in-flight jobs. Size to the "
        "slowest shared downstream — typically the docling-serve fleet or "
        "the embedding endpoint's request budget.",
    )
    poll_idle_interval_s: float = Field(
        default=1.0,
        description="How long an idle worker waits between empty claim_next "
        "polls. Lower = lower latency picking up new jobs, higher = less "
        "queue churn when the queue is usually empty.",
    )
    lease_ttl_s: int = Field(
        default=120,
        gt=0,
        description="A `claimed` job whose lease has not been renewed within "
        "this window is presumed dead and reset to `queued` by the reaper. A "
        "live worker renews its lease every heartbeat_interval_s while "
        "processing, so this need not exceed job duration — it only bounds how "
        "long a crashed worker's job stays stuck before another worker takes "
        "it over.",
    )
    heartbeat_interval_s: int = Field(
        default=30,
        gt=0,
        description="How often a worker renews the lease on its in-flight "
        "jobs. Must be comfortably shorter than lease_ttl_s so scheduler "
        "jitter or a slow DB round-trip can't let a live job's lease lapse.",
    )
    reaper_interval_s: int = Field(
        default=60,
        description="How often the reaper scans for stale claims. Shorter "
        "lowers the worst-case recovery time after a worker crash.",
    )
    retry: RetryPolicyConfig = Field(default_factory=RetryPolicyConfig)
    shutdown_grace_s: float = Field(
        default=60.0,
        description="On SIGINT/SIGTERM, how long to wait for in-flight jobs to "
        "finish before forcing cancellation. Cancelled jobs are released back "
        "to `queued` for immediate re-claim.",
    )

    @model_validator(mode="after")
    def _check_heartbeat_cadence(self) -> "WorkerConfig":
        if self.heartbeat_interval_s > self.lease_ttl_s / 3:
            raise ValueError(
                "heartbeat_interval_s must be <= lease_ttl_s / 3 so a live "
                "worker renews its lease several times before it could expire"
            )
        return self


class APIConfig(BaseModel):
    """HTTP control plane settings for the ingester."""

    # Validate on assignment so CLI overrides (e.g. --root-path) run the same
    # normalization as values parsed from the config file.
    model_config = ConfigDict(validate_assignment=True)

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8765
    auth_token: str | None = None
    root_path: str = Field(
        default="",
        description=(
            "Base path the control plane is served under when reverse-proxied "
            "behind a sub-path (e.g. '/ingester'). Empty serves at the root. "
            "Forwarded to FastAPI/uvicorn as root_path and used to set the "
            "dashboard's <base href> so its fetches are prefix-aware."
        ),
    )

    @field_validator("root_path")
    @classmethod
    def _normalize_root_path(cls, value: str) -> str:
        """Normalize to '' (root) or a single leading-slash, no-trailing-slash
        prefix, so 'ingester', '/ingester/' and '/' become '/ingester',
        '/ingester' and ''."""
        trimmed = value.strip().rstrip("/")
        if trimmed and not trimmed.startswith("/"):
            trimmed = "/" + trimmed
        return trimmed


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
    max_file_size: int | None = Field(
        default=None,
        description="Maximum file size in bytes to fetch. Files larger than "
        "this are rejected with a PermanentError. None = no limit.",
    )
    metadata_provider: str | None = Field(
        default=None,
        description="Name of a metadata provider registered under the "
        "'haiku.rag.metadata_providers' entry-point group. When set, the "
        "provider is called per document with (source_id, uri) and its result "
        "is attached as document metadata. None = no provider.",
    )


class FSSourceConfig(_SourceBase):
    type: Literal["fs"]
    root: Path
    ignore_patterns: list[str] = []
    include_patterns: list[str] = []


class HTTPSourceConfig(_SourceBase):
    type: Literal["http"]
    # HTTP sources have no natural key to derive an id from (a list of urls
    # has no canonical representation), so require one.
    id: str
    urls: list[str] = []
    headers: dict[str, str] = Field(default_factory=dict)


class S3SourceConfig(_SourceBase):
    type: Literal["s3"]
    uri: str
    storage_options: dict[str, str] = Field(default_factory=dict)
    ignore_patterns: list[str] = []
    include_patterns: list[str] = []


class WebDAVSourceConfig(_SourceBase):
    """A WebDAV collection (Nextcloud, ownCloud, Apache mod_dav, etc.). Files
    are discovered via PROPFIND on `base_url`; fetch is plain HTTP GET."""

    type: Literal["webdav"]
    # base_url can be deep + opaque (long URL paths, credentials embedded);
    # require an explicit short id for the queue and logs.
    id: str
    base_url: str
    username: str | None = None
    password: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    ignore_patterns: list[str] = []
    include_patterns: list[str] = []


class PluginSourceConfig(_SourceBase):
    """A source provided by an external package registered under the
    `haiku.rag.sources` entry-point group. `plugin` names the entry point;
    `options` is passed through to the plugin's factory, which validates it."""

    type: Literal["plugin"]
    # No natural key to derive an id from; require an explicit one.
    id: str
    plugin: str = Field(
        description="Name of a source factory registered under the "
        "'haiku.rag.sources' entry-point group."
    )
    options: dict[str, Any] = Field(default_factory=dict)


SourceConfig = Annotated[
    FSSourceConfig
    | HTTPSourceConfig
    | S3SourceConfig
    | WebDAVSourceConfig
    | PluginSourceConfig,
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
    doctor: DoctorConfig = Field(default_factory=DoctorConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    ingester: IngesterConfig = Field(default_factory=IngesterConfig)
    evaluations: "EvaluationsConfig" = Field(
        default_factory=lambda: EvaluationsConfig()
    )
