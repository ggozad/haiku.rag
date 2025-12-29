# Changelog
## [Unreleased]

## [0.23.1] - 2025-12-29

### Added

- **Contextualized FTS Search**: Full-text search now includes section headings
  - New `content_fts` column stores contextualized content (headings + body text)
  - FTS index now searches `content_fts` for better keyword matching on section context
  - Original `content` column preserved for display and context expansion
  - Migration automatically populates `content_fts` for existing databases
- **GitHub Actions CI**: Test workflow runs pytest, pyright, and ruff on push/PR to main
- **VCR Cassette Recording**: Integration tests use recorded HTTP responses for deterministic CI runs
  - LLM tests (QA, embeddings, research graph) replay from cassettes without real API calls
  - docling-serve tests run without Docker container in CI
  - Uses pytest-recording with custom JSON body serializer

## [0.23.0] - 2025-12-26

### Added

- **Prompt Customization**: Configure agent prompts via `prompts` config section
  - `domain_preamble`: Prepended to all agent prompts for domain context
  - `qa`: Full replacement for QA agent prompt
  - `synthesis`: Full replacement for research synthesis prompt

### Changed

- **Embeddings**: Migrated to pydantic-ai's embeddings module
  - Uses pydantic-ai v1.39.0+ embeddings with instrumentation and token counting support
  - Explicit `embed_query()` and `embed_documents()` API for query/document distinction
  - New providers available: Cohere (`cohere:`), SentenceTransformers (`sentence-transformers:`)
  - VoyageAI refactored to extend pydantic-ai's `EmbeddingModel` base class
- **Configuration**: Added `base_url` to `ModelConfig` and `EmbeddingModelConfig`
  - Enables custom endpoints for OpenAI-compatible providers (vLLM, LM Studio, etc.)
  - Model-level `base_url` takes precedence over provider config

### Deprecated

- **vLLM and LM Studio providers**: Use `openai` provider with `base_url` instead
  - `provider: vllm` → `provider: openai` with `base_url: http://localhost:8000/v1`
  - `provider: lm_studio` → `provider: openai` with `base_url: http://localhost:1234/v1`

### Removed

- Deleted obsolete embedder implementations: `ollama.py`, `openai.py`, `vllm.py`, `lm_studio.py`, `base.py`
- Removed `VLLMConfig` and `LMStudioConfig` from configuration (use `base_url` in model config instead)

## [0.22.0] - 2025-12-19

### Added

- **Read-Only Mode**: Global `--read-only` CLI flag for safe database access without modifications
  - Blocks all write operations at the Store layer
  - Skips database upgrades and settings saves on open
  - Excludes write tools (`add_document_*`, `delete_document`) from MCP server
  - Disables file monitor with warning when `--read-only` is used with `serve --monitor`
- **Time Travel**: Query the database as it existed at a previous point in time
  - Global `--before` CLI flag accepts datetime strings (ISO 8601 or date-only)
  - Automatically enables read-only mode when time-traveling
  - New `history` command shows version history for database tables
  - Useful for debugging and auditing
  - Supported throughout: CLI, Client, App, Inspector

### Fixed

- **File Monitor Path Validation**: Monitor now validates directories exist before watching ([#204](https://github.com/ggozad/haiku.rag/issues/204))
  - Provides clear error message pointing to `haiku.rag.yaml` configuration
  - Prevents cryptic `FileNotFoundError: No path was found` from watchfiles
- **Docker Documentation**: Improved Docker setup instructions
  - Added volume mount examples for config file and documents directory
  - Clarified that `monitor.directories` must use container paths, not host paths

### Changed

- **Dependencies**: Updated core dependencies
  - `pydantic-ai-slim`: 1.27.0 → 1.36.0 (FileSearchTool, web chat UI, GPT-5.2 support, prompt caching)
  - `lancedb`: 0.25.3 → 0.26.0
  - `docling`: 2.64.0 → 2.65.0
  - `docling-core`: 2.54.0 → 2.57.0

## [0.21.0] - 2025-12-18

### Added

- **Interactive Research Mode**: Human-in-the-loop research using graph-based decision nodes
  - `haiku-rag research --interactive` starts conversational CLI chat
  - Natural language interpretation for user commands (search, modify questions, synthesize)
  - Chat with assistant before starting research, and during decision points
  - Review collected answers and pending questions at each decision point
  - Add, remove, or modify sub-questions through natural conversation
  - New `human_decide` graph node emits AG-UI tool calls (`TOOL_CALL_START/ARGS/END`) for frontend integration
  - New `emit_tool_call_start()`, `emit_tool_call_args()`, `emit_tool_call_end()` AG-UI event helpers
  - New `AGUIEmitter.emit()` method for direct event emission
- **AG-UI Research Example**: Human-in-the-loop research with client-side tool calling
  - Frontend handles `human_decision` tool calls via AG-UI `TOOL_CALL_*` events
  - Tool results sent directly to backend `/v1/research/stream` endpoint
  - Backend queues decisions and continues the research graph
- **HotpotQA Evaluation**: Added HotpotQA dataset adapter for multi-hop QA benchmarks
  - Extracts unique documents from validation set context paragraphs
  - Uses MAP for retrieval evaluation (multiple supporting documents per question)
  - Run with `evaluations hotpotqa`
- **Plain Text Format**: Added `format="plain"` for text conversion
  - Use when content is plain text without markdown/HTML structure
  - Falls back gracefully when docling cannot detect markdown format in content
  - Supported in `create_document()`, `convert()`, and all converter classes

### Changed

- **AG-UI Events**: Replaced custom event classes with `ag_ui.core` types
  - Removed `haiku.rag.graph.agui.events` module
  - Event factory functions (`emit_*`) now wrap official `ag_ui.core` event classes
- **Chunker Sets Order**: Chunkers now set `chunk.order` directly
- **Unified Research Graph**: Simplified and unified research and deep QA into a single configurable graph
  - Removed `analyze_insights` node - graph now flows directly from `collect_answers` to `decide`
  - Simplified `EvaluationResult` to: `is_sufficient`, `confidence_score`, `reasoning`, `new_questions`
  - Simplified `ResearchContext` - removed insight/gap tracking methods
  - `ask --deep` now uses research graph with `max_iterations=2`, `confidence_threshold=0.0`
  - `ask --deep` output now shows executive summary, key findings, and sources
  - Added `include_plan` parameter to `build_research_graph()` for plan-less execution
  - Added `max_iterations` and `confidence_threshold` overrides to `ResearchState.from_config()`
- **Improved Synthesis Prompt**: Updated synthesis agent prompt to produce direct answers
  - Executive summary now directly answers the question instead of describing the report
  - Added explicit examples of good vs bad output style
- **Evaluations Vacuum Strategy**: `populate_db` now uses periodic vacuum to prevent disk exhaustion with large datasets
  - Disables auto_vacuum during population, vacuums every N documents with retention=0
  - New `--vacuum-interval` CLI option (default: 100) to control vacuum frequency
  - Prevents disk space issues when building databases with thousands of documents (e.g., HotpotQA)
- **Benchmarks Documentation**: Restructured benchmarks.md for clarity
  - Added dedicated Methodology section explaining MRR, MAP, and QA Accuracy metrics
  - Organized results by dataset with retrieval and QA subsections

### Removed

- **Deep QA Graph**: Removed `haiku.rag.graph.deep_qa` module entirely
  - Use `build_research_graph()` with appropriate parameters instead
  - `ask --deep` CLI command now uses research graph internally
- **Insight/Gap Tracking**: Removed over-engineered insight and gap tracking from research graph
  - Removed `InsightRecord`, `GapRecord`, `InsightAnalysis`, `InsightStatus`, `GapSeverity` models
  - Removed `format_analysis_for_prompt()` helper
  - Removed `INSIGHT_AGENT_PROMPT` from prompts

## [0.20.2] - 2025-12-12

### Fixed

- **LLM Schema Compliance**: Improved prompts to prevent LLMs from returning objects instead of plain strings for `list[str]` fields
  - All graph prompts now explicitly state that list fields must contain plain strings only
  - Added missing `query` and `confidence` fields to search agent output format documentation
  - Fixes validation errors with less capable models that ignore JSON schema constraints
- **AG-UI Frontend Types**: Fixed TypeScript interfaces in ag-ui-research example to match backend Python models
  - `EvaluationResult`: `confidence` → `confidence_score`, `should_continue` → `is_sufficient`, `gaps_identified` → `gaps`, `follow_up_questions` → `new_questions`, added `key_insights`
  - `ResearchReport`: `question` → `title`, `summary` → `executive_summary`, `findings` → `main_findings`, removed `insights_used`/`methodology`, added `limitations`/`recommendations`/`sources_summary`
  - Updated Final Report UI to display new fields (Recommendations, Limitations, Sources)
- **Citation Formatting**: Citations in CLI now render properly with Rich panels
  - Content is rendered as markdown with proper code block formatting
  - No longer truncates or flattens newlines in citation content

## [0.20.1] - 2025-12-11

### Added

- **Search Filter for Graphs**: Research and Deep QA graphs now support `search_filter` parameter to restrict searches to specific documents
  - Set `state.search_filter` to a SQL WHERE clause (e.g., `"id IN ('doc1', 'doc2')"`) before running the graph
  - Enables document-scoped research workflows
  - CLI: `haiku-rag research "question" --filter "uri LIKE '%paper%'"`
  - CLI: `haiku-rag ask "question" --filter "title = 'My Doc'"`
  - Python: `client.ask(question, filter="...")` and `agent.answer(question, filter="...")`
- **AG-UI Research Example**: Added bidirectional state demonstration with document filter
  - New `/api/documents` endpoint to list available documents
  - Frontend document selector component with search and multi-select
  - Demonstrates client-to-server state flow via AG-UI protocol
- **Inspector Info Modal**: New `i` keyboard shortcut opens a modal displaying database information

### Changed

- **Inspector Lazy Loading**: Chunks panel now loads chunks in batches of 50 with infinite scroll
  - Fixes unresponsive UI when viewing documents with large numbers of chunks
  - New `ChunkRepository.get_by_document_id()` pagination with `limit` and `offset` parameters
  - New `ChunkRepository.count_by_document_id()` method

## [0.20.0] - 2025-12-10

### Added

- **DoclingDocument Storage**: Full DoclingDocument JSON is now stored with each document, enabling rich context and visual grounding
  - Documents store the complete DoclingDocument structure (JSON) and schema version
  - Chunks store metadata with JSON pointer references (`doc_item_refs`), semantic labels, section headings, and page numbers
  - New `ChunkMetadata` model for structured chunk provenance: `doc_item_refs`, `headings`, `labels`, `page_numbers`
  - `Document.get_docling_document()` method to parse stored DoclingDocument
  - `ChunkMetadata.resolve_doc_items()` to resolve JSON pointer refs to actual DocItem objects
  - `ChunkMetadata.resolve_bounding_boxes()` for visual grounding with page coordinates
  - LRU cache (100 documents) for parsed DoclingDocument objects to avoid repeated JSON parsing
- **Enhanced Search Results**: `search()` and `expand_context()` now return full provenance information
  - `SearchResult` includes `page_numbers`, `headings`, `labels`, and `doc_item_refs`
  - QA and research agents use provenance for better citations (page numbers, section headings)
- **Type-Aware Context Expansion**: `expand_context()` now uses document structure for intelligent expansion
  - Structural content (tables, code blocks, lists) expands to complete structures regardless of chunking
  - Text content uses radius-based expansion via `text_context_radius` setting
  - `max_context_items` and `max_context_chars` settings control expansion limits
  - `SearchResult.format_for_agent()` method formats expanded results with metadata for LLM consumption
- **Visual Grounding**: View page images with highlighted bounding boxes for chunks
  - Inspector modal with keyboard navigation between pages
  - CLI command: `haiku-rag visualize <chunk_id>`
  - Requires `textual-image` dependency and terminal with image support
- **Processing Primitives**: New methods for custom document processing pipelines
  - `convert()` - Convert files, URLs, or text to DoclingDocument
  - `chunk()` - Chunk a DoclingDocument into Chunk objects
  - `contextualize()` - Prepend section headings to chunk content for embedding
  - `embed_chunks()` - Generate embeddings for chunks
- **New `import_document()` Method**: Import pre-processed documents with custom chunks
  - Accepts `DoclingDocument` directly for rich metadata (visual grounding, page numbers)
  - Use when document conversion, chunking, or embedding were done externally
  - Chunks without embeddings are automatically embedded
- **Automatic Chunk Embedding**: `import_document()` and `update_document()` automatically embed chunks that don't have embeddings
  - Pass chunks with or without embeddings - missing embeddings are generated
  - Chunks with pre-computed embeddings are stored as-is
- **Format Parameter for Text Conversion**: New `format` parameter for `convert()` and `create_document()` to specify content type
  - Supports `"md"` (default) for markdown and `"html"` for HTML content
  - HTML format preserves document structure (headings, lists, sections) in DoclingDocument
  - Enables proper parsing of HTML content that was previously treated as plain text
- **Inspector Context Modal**: Press `c` in the inspector to view expanded context for the selected chunk
- **Auto-Vacuum Configuration**: New `storage.auto_vacuum` setting to control automatic vacuuming behavior
  - When `true` (default), vacuum runs automatically after document create/update operations and rebuilds
  - When `false`, vacuum only runs via explicit `haiku-rag vacuum` command
  - Disabling can help avoid potential crashes in high-concurrency scenarios due to LanceDB race conditions

### Changed

- **BREAKING: `create_document()` API**: Removed `chunks` parameter
  - `create_document()` now always processes content (converts, chunks, embeds)
  - Use `import_document()` for pre-processed documents with custom chunks
- **BREAKING: `update_document()` API**: Unified with `update_document_fields()`
  - Old: `update_document(document)` - pass modified Document object
  - New: `update_document(document_id, content=, metadata=, chunks=, title=, docling_document=)`
  - `content` and `docling_document` are mutually exclusive
- **BREAKING: Chunker Interface**: `DocumentChunker.chunk()` now returns `list[Chunk]` instead of `list[str]`
  - Chunks include structured metadata (doc_item_refs, labels, headings, page_numbers)
- **Search Config**: New settings in `search` section for search behavior and context expansion
  - `search.limit` - Default number of search results (default: 5). Used by CLI, MCP server, and API when no limit specified
  - `search.context_radius` - DocItems before/after to include for text content expansion (default: 0)
  - `search.max_context_items` - Maximum items in expanded context (default: 10)
  - `search.max_context_chars` - Maximum characters in expanded context (default: 10000)
- **Rebuild Performance**: Batched database writes during `rebuild` command reduce LanceDB versions by ~98%
  - All rebuild modes (FULL, RECHUNK, EMBED_ONLY) now batch writes across documents
  - Eliminates redundant per-document chunk deletions and vacuum calls
  - Significantly reduces storage overhead and improves rebuild speed for large databases
- **Embedding Architecture**: Moved embedding generation from `ChunkRepository` to client layer
  - Repository is now a pure persistence layer
  - Client handles embedding via `_ensure_chunks_embedded()`
- **Chunk Text Storage**: Chunks store raw text; headings prepended only at embedding time
  - Stored chunk content stays clean without duplicate heading prefixes
  - Local and serve chunkers now produce identical output
- **Citation Models**: Introduced `RawSearchAnswer` for LLM output, `SearchAnswer` with resolved citations
- **Page Image Generation**: Always enabled for local docling converter (required for visual grounding)
- **Download Models Progress**: `haiku-rag download-models` now shows real-time progress with Rich progress bars for Ollama model downloads

### Removed

- **BREAKING: `markdown_preprocessor` Config Option**: Use processing primitives (`convert()`, `chunk()`, `embed_chunks()`) for custom pipelines
- **`update_document_fields()`**: Merged into `update_document()`

### Migration

This release requires a database rebuild to populate the new DoclingDocument fields:

```bash
haiku-rag rebuild
```

Existing documents without DoclingDocument data will work but won't have provenance information.

## [0.19.6] - 2025-12-03

### Changed

- **BREAKING: Explicit Database Creation**: Databases must now be explicitly created before use
  - New `haiku-rag init` command creates a new empty database
  - Python API: `HaikuRAG(path, create=True)` to create database programmatically
  - Operations on non-existent databases raise `FileNotFoundError`
- **BREAKING: Embeddings Configuration**: Restructured to nested `EmbeddingModelConfig`
  - Config path changed from `embeddings.{provider, model, vector_dim}` to `embeddings.model.{provider, name, vector_dim}`
  - Automatic migration upgrades existing databases to new format
- **Database Migrations**: Always run when opening an existing database

## [0.19.5] - 2025-12-01

### Changed

- **Rebuild Performance**: Optimized `rebuild --embed-only` to use batch updates via LanceDB's `merge_insert` instead of individual chunk updates, and skip chunks with unchanged embeddings

## [0.19.4] - 2025-11-28

### Added

- **Rebuild Modes**: New options for `rebuild` command to control what gets rebuilt
  - `--embed-only`: Only regenerate embeddings, keeping existing chunks (fastest option when changing embedding model)
  - `--rechunk`: Re-chunk from existing document content without accessing source files
  - Default (no flag): Full rebuild with source file re-conversion
  - Python API: `rebuild_database(mode=RebuildMode.EMBED_ONLY | RECHUNK | FULL)`

## [0.19.3] - 2025-11-27

### Changed

- **Async Chunker**: `DoclingServeChunker` now uses `httpx.AsyncClient` instead of sync `requests`

### Fixed

- **OCR Options**: Fixed `DoclingLocalConverter` using base `OcrOptions` class which docling's OCR factory doesn't recognize. Now uses `OcrAutoOptions` for automatic OCR engine selection.
- **Dependencies**: Added `opencv-python-headless` to the `docling` optional dependency for table structure detection.

## [0.19.2] - 2025-11-27

### Changed

- **Async Converters**: Made document converters fully async
  - `BaseConverter.convert_file()` and `convert_text()` are now async methods
  - `DoclingLocalConverter` wraps blocking Docling operations with `asyncio.to_thread()`
  - `DoclingServeConverter` now uses `httpx.AsyncClient` instead of sync `requests`
- **Async Model Prefetch**: `prefetch_models()` is now async
  - Uses `httpx.AsyncClient` for Ollama model pulls
  - Wraps blocking Docling and HuggingFace downloads with `asyncio.to_thread()`

## [0.19.1] - 2025-11-26

### Added

- **LM Studio Provider**: Added support for LM Studio as a provider for embeddings and QA/research models
  - Configure with `provider: lm_studio` in embeddings, QA, or research model settings
  - Supports thinking control for reasoning models (gpt-oss, etc.)
  - Default base URL: `http://localhost:1234`

### Fixed

- **Configuration**: Fixed `init-config` command generating invalid configuration files (#165)
  - Refactored `generate_default_config()` to use Pydantic model serialization instead of manual dict construction
  - Updated `qa`, `research`, and `reranking` sections to use new `ModelConfig` structure

## [0.19.0] - 2025-11-25

### Added

- **Model Customization**: Added support for per-model configuration settings
  - New `enable_thinking` parameter to control reasoning behavior (true/false/None)
  - Support for `temperature` and `max_tokens` settings on QA and research models
  - All settings apply to any provider that supports them
- **Database Inspector**: New `inspect` CLI command launches interactive TUI for browsing documents and chunks & searching
- **Evaluations**: Added `evaluations` CLI script for running benchmarks (replaces `python -m evaluations.benchmark`)
- **Evaluations**: Added `--db` option to override evaluation database path
  - Default database location moved to haiku.rag data directory:
    - macOS: `~/Library/Application Support/haiku.rag/evaluations/dbs/`
    - Linux: `~/.local/share/haiku.rag/evaluations/dbs/`
    - Windows: `C:/Users/<USER>/AppData/Roaming/haiku.rag/evaluations/dbs/`
  - Previously stored in `evaluations/data/` within the repository
- **Evaluations**: Added comprehensive experiment metadata tracking for better reproducibility
  - Records dataset name, test case count, and all model configurations
  - Tracks embedder settings: provider, model, and vector dimensions
  - Tracks QA model: provider and model name
  - Tracks judge model: provider and model name for LLM evaluation
  - Tracks processing parameters: `chunk_size` and `context_chunk_radius`
  - Tracks retrieval configuration: `retrieval_limit` for number of chunks retrieved
  - Tracks reranking configuration: `rerank_provider` and `rerank_model`
  - Enables comparison of evaluation runs with different configurations in Logfire
- **Evaluations**: Refactored retrieval evaluation to use pydantic-ai experiment framework
  - New `evaluators` module with `MRREvaluator` (Mean Reciprocal Rank) and `MAPEvaluator` (Mean Average Precision)
  - Retrieval benchmarks now use `Dataset.evaluate()` with full Logfire experiment tracking
  - Dataset specifications now declare their retrieval evaluator (MRR for RepliQA, MAP for Wix)
  - Replaced Recall@K and Success@K with industry-standard MRR and MAP metrics
  - Unified evaluation framework for both retrieval and QA benchmarks
- **AG-UI Events**: Enhanced ActivitySnapshot events with richer structured data
  - Added `stepName` field to identify which graph node emitted each activity
  - Added structured fields to activity content while preserving backward-compatible `message` field:
    - **Planning**: `sub_questions` - list of sub-question strings
    - **Searching**: `query` - the search query, `confidence` - answer confidence (on success), `error` - error message (on failure)
    - **Analyzing** (research): `insights` - list of insight objects, `gaps` - list of gap objects, `resolved_gaps` - list of resolved gap strings
    - **Evaluating** (research): `confidence` - confidence score, `is_sufficient` - sufficiency flag
    - **Evaluating** (deep QA): `is_sufficient` - sufficiency flag, `iterations` - iteration count

### Changed

- **Evaluations**: Renamed `--qa-limit` CLI parameter to `--limit`, now applies to both retrieval and QA benchmarks
- **Evaluations**: Retrieval evaluator selection moved from runtime logic to dataset configuration

## [0.18.0] - 2025-11-21

### Added

- **Manual Vector Indexing**: New `create-index` CLI command for explicit vector index creation
  - Creates IVF_PQ indexes
  - Requires minimum 256 chunks (LanceDB training data requirement)
  - New `search.vector_index_metric` config option: `cosine` (default), `l2`, or `dot`
  - New `search.vector_refine_factor` config option (default: 30) for accuracy/speed tradeoff
  - Indexes not created automatically during ingestion to avoid performance degradation
  - Manual rebuilding required after adding significant new data
- **Enhanced Info Command**: `haiku-rag info` now shows storage sizes and vector index statistics
  - Displays storage size for documents and chunks tables in human-readable format
  - Shows vector index status (exists/not created)
  - Shows indexed and unindexed chunk counts for monitoring index staleness

### Changed

- **BREAKING: Default Embedding Model**: Changed default embedding model from `qwen3-embedding` to `qwen3-embedding:4b` with vector dimension 2560 (previously 4096)
  - New installations will use the smaller, more efficient 4B parameter model by default
  - **Action required**: Existing databases created with the old default will be incompatible. Users must either:
    - Explicitly set `embeddings.model: "qwen3-embedding"` and `embeddings.vector_dim: 4096` in their config to maintain compatibility with existing databases
    - Or run `haiku-rag rebuild` to re-embed all documents with the new default
  - This change provides better performance for most use cases while reducing resource requirements
- **Evaluations**: Improved evaluation dataset naming and simplified evaluator configuration
  - `EvalDataset` now accepts dataset name for better organization in Logfire
  - Added `--name` CLI parameter to override evaluation run names
  - Removed `IsInstance` evaluator, using only `LLMJudge` for QA evaluation
- **Search Accuracy**: Applied `refine_factor` to vector and hybrid searches for improved accuracy
  - Retrieves `refine_factor * limit` candidates and re-ranks in memory
  - Higher values increase accuracy but slow down queries

### Fixed

- **AG-UI Activity Events**: Activity events now correctly use structured dict content instead of strings
- **Graph Configuration**: Graph builder functions now properly accept and use non-global config (#149)
  - `build_research_graph()` and `build_deep_qa_graph()` now pass config to all agents and model creation
  - `get_model()` utility function accepts `config` parameter (defaults to global Config)
  - Allows creating multiple graphs with different configurations in the same application


## [0.17.2] - 2025-11-19

### Added

- **Document Update API**: New `update_document_fields()` method for partial document updates
  - Update individual fields (content, metadata, title, chunks) without fetching full document
  - Support for custom chunks or auto-generation from content

### Changed

- **Chunk Creation**: `ChunkRepository.create()` now accepts both single chunks and lists for batch insertion
  - Batch insertion reduces LanceDB version creation when adding multiple chunks with custom chunks
  - Batch embedding generation for improved performance with multiple chunks
- Updated core dependencies

## [0.17.1] - 2025-11-18

### Added

- **Conversion Options**: Fine-grained control over document conversion for both local and remote converters
  - New `conversion_options` config section in `ProcessingConfig`
  - OCR settings: `do_ocr`, `force_ocr`, `ocr_lang` for controlling OCR behavior
  - Table extraction: `do_table_structure`, `table_mode` (fast/accurate), `table_cell_matching`
  - Image settings: `images_scale` to control image resolution
  - Options work identically with both `docling-local` and `docling-serve` converters

### Changed

- Increase reranking candidate retrieval multiplier from 3x to 10x for improved result quality
- **Docker Images**: Main `haiku.rag` image no longer automatically built and published
- **Conversion Options**: Removed the legacy `pdf_backend` setting; docling now chooses the optimal backend automatically

## [0.17.0] - 2025-11-17

### Added

- **Remote Processing**: Support for docling-serve as remote document processing and chunking service
  - New `converter` config option: `docling-local` (default) or `docling-serve`
  - New `chunker` config option: `docling-local` (default) or `docling-serve`
  - New `providers.docling_serve` config section with `base_url`, `api_key`, and `timeout`
  - Comprehensive error handling for connection, timeout, and authentication issues
- **Chunking Strategies**: Support for both hybrid and hierarchical chunking
  - New `chunker_type` config option: `hybrid` (default) or `hierarchical`
  - Hybrid chunking: Structure-aware splitting that respects document boundaries
  - Hierarchical chunking: Preserves document hierarchy for nested documents
- **Table Serialization Control**: Configurable table representation in chunks
  - New `chunking_use_markdown_tables` config option (default: `false`)
  - `false`: Tables serialized as narrative text ("Value A, Column 2 = Value B")
  - `true`: Tables preserved as markdown format with structure
- **Chunking Configuration**: Additional chunking control options
  - New `chunking_merge_peers` config option (default: `true`) to merge undersized successive chunks
- **Docker Images**: Two Docker images for different deployment scenarios
  - `haiku.rag`: Full image with all dependencies for self-contained deployments
  - `haiku.rag-slim`: Minimal image designed for use with external docling-serve
  - Multi-platform support (linux/amd64, linux/arm64)
  - Docker Compose examples with docling-serve integration
  - Automated CI/CD workflows for both images
  - Build script (`scripts/build-docker-images.sh`) for local multi-platform builds

### Changed

- **BREAKING: Chunking Tokenizer**: Switched from tiktoken to HuggingFace tokenizers for consistency with docling-serve
  - Default tokenizer changed from tiktoken "gpt-4o" to "Qwen/Qwen3-Embedding-0.6B"
  - New `chunking_tokenizer` config option in `ProcessingConfig` for customization
  - `download-models` CLI command now also downloads the configured HuggingFace tokenizer
- **Docker Examples**: Updated examples to demonstrate remote processing
  - `examples/docker` now uses slim image with docling-serve
  - `examples/ag-ui-research` backend uses slim image with docling-serve
  - Configuration examples include remote processing setup

## [0.16.1] - 2025-11-14

### Changed

- **Evaluations**: Refactored QA benchmark to run entire dataset as single evaluation for better Logfire experiment tracking
- **Evaluations**: Added `.env` file loading support via `python-dotenv` dependency

## [0.16.0] - 2025-11-13

### Added

- **AG-UI Protocol Support**: Full AG-UI (Agent-UI) protocol implementation for graph execution with event streaming
  - New `AGUIEmitter` class for emitting AG-UI events from graphs
  - Support for all AG-UI event types: lifecycle events (`RUN_STARTED`, `RUN_FINISHED`, `RUN_ERROR`), step events (`STEP_STARTED`, `STEP_FINISHED`), state updates (`STATE_SNAPSHOT`, `STATE_DELTA`), activity narration (`ACTIVITY_SNAPSHOT`), and text messages (`TEXT_MESSAGE_CHUNK`)
  - `AGUIConsoleRenderer` for rendering AG-UI event streams to terminal with Rich formatting
  - `stream_graph()` utility function for executing graphs with AG-UI event emission
  - State diff computation for efficient state synchronization
  - **Delta State Updates**: AG-UI emitter now supports incremental state updates via JSON Patch operations (`STATE_DELTA` events) to reduce bandwidth, configurable via `use_deltas` parameter (enabled by default)
- **AG-UI Server**: Starlette-based HTTP server for serving graphs via AG-UI protocol
  - Server-Sent Events (SSE) streaming endpoint at `/v1/agent/stream`
  - Health check endpoint at `/health`
  - Full CORS support configurable via `agui` config section
  - `create_agui_server()` function for programmatic server creation
- **Deep QA AG-UI Support**: Deep QA graph now fully supports AG-UI event streaming
  - Integration with `AGUIEmitter` for progress tracking
  - Step-by-step execution visibility via AG-UI events
- **CLI AG-UI Flag**: New `--agui` flag for `serve` command to start AG-UI server
- **Graph Module**: New unified `haiku.rag.graph` module containing all graph-related functionality
- **Common Graph Nodes**: New factory functions (`create_plan_node`, `create_search_node`) in `haiku.rag.graph.common.nodes` for reusable graph components
- **AG-UI Research Example**: New full-stack example (`examples/ag-ui-research`) demonstrating agent+graph architecture with CopilotKit frontend
  - Pydantic AI agent with research tool that invokes the research graph
  - Custom AG-UI streaming endpoint with anyio memory streams
  - React/Next.js frontend with split-pane UI showing live research state
  - Real-time progress tracking of questions, answers, insights, and gaps
  - Docker Compose setup for easy local development

### Changed

- **Vacuum Retention**: Default `vacuum_retention_seconds` increased from 60 seconds to 86400 seconds (1 day) for better version retention in typical workflows
- **BREAKING**: Major refactoring of graph-related code into unified `haiku.rag.graph` module structure:
  - `haiku.rag.research` → `haiku.rag.graph.research`
  - `haiku.rag.qa.deep` → `haiku.rag.graph.deep_qa`
  - `haiku.rag.agui` → `haiku.rag.graph.agui`
  - `haiku.rag.graph_common` → `haiku.rag.graph.common`
- **BREAKING**: Research and Deep QA graphs now use AG-UI event protocol instead of direct console logging
  - Removed `console` and `stream` parameters from graph dependencies
  - All progress updates now emit through `AGUIEmitter`
- **BREAKING**: `ResearchState` converted from dataclass to Pydantic `BaseModel` for JSON serialization and AG-UI compatibility
- Research and Deep QA graphs now emit detailed execution events for better observability
- CLI research command now uses AG-UI event rendering for `--verbose` output
- Improved graph execution visibility with step-by-step progress tracking
- Updated all documentation to reflect new import paths and AG-UI usage
- Updated examples (ag-ui-research, a2a-server) to use new import paths

### Fixed

- **Document Creation**: Optimized `create_document` to skip unnecessary DoclingDocument conversion when chunks are pre-provided
- **FileReader**: Error messages now include both original exception details and file path for easier debugging
- **Database Auto-creation**: Read operations (search, list, get, ask, research) no longer auto-create empty databases. Write operations (add, add-src, delete, rebuild) still create the database as needed. This prevents the confusing scenario where a search query creates an empty database. Fixes issue #137.

### Removed

- **BREAKING**: Removed `disable_autocreate` config option - the behavior is now automatic based on operation type
- **BREAKING**: Removed legacy `ResearchStream` and `ResearchStreamEvent` classes (replaced by AG-UI event protocol)

## [0.15.0] - 2025-11-07

### Added

- **File Monitor**: Orphan deletion feature - automatically removes documents from database when source files are deleted (enabled via `monitor.delete_orphans` config option, default: false)

### Changed

- **Configuration**: All CLI commands now properly support `--config` parameter for specifying custom configuration files
- Configuration loading consolidated across CLI, app, and client with consistent resolution order
- `HaikuRAGApp` and MCP server now accept `config` parameter for programmatic configuration
- Updated CLI documentation to clarify global vs per-command options
- **BREAKING**: Standardized configuration filename to `haiku.rag.yaml` in user directories (was incorrectly using `config.yaml`). Users with existing `config.yaml` in their user directory will need to rename it to `haiku.rag.yaml`

### Fixed

- **File Monitor**: Fixed incorrect "Updated document" logging for unchanged files - monitor now properly skips files when MD5 hash hasn't changed

### Removed

- **BREAKING**: A2A (Agent-to-Agent) protocol support has been moved to a separate self-contained package in `examples/a2a-server/`. The A2A server is no longer part of the main haiku.rag package. Users who need A2A functionality can install and run it from the examples directory with `cd examples/a2a-server && uv sync`.
- **BREAKING**: Removed deprecated `.env`-based configuration system. The `haiku-rag init-config --from-env` command and `load_config_from_env()` function have been removed. All configuration must now be done via YAML files. Environment variables for API keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) and service URLs (e.g., `OLLAMA_BASE_URL`) are still supported and can be set via `.env` files.

## [0.14.1] - 2025-11-06

### Added

- Migrated research and deep QA agents to use Pydantic Graph beta API for better graph execution
- Automatic semaphore-based concurrency control for parallel sub-question processing
- `max_concurrency` parameter for controlling parallel execution in research and deep QA (default: 1)

### Changed

- **BREAKING**: Research and Deep QA graphs now use `pydantic_graph.beta` instead of the class-based graph implementation
- Refactored graph common patterns into `graph_common` module
- Sub-questions now process using `.map()` for true parallel execution
- Improved graph structure with cleaner node definitions and flow control
- Pinned critical dependencies: `docling-core`, `lancedb`, `docling`

## [0.14.0] - 2024-11-05

### Added

- New `haiku.rag-slim` package with minimal dependencies for users who want to install only what they need
- Evaluations package (`haiku.rag-evals`) for internal benchmarking and testing
- Improved search filtering performance by using pandas DataFrames for joins instead of SQL WHERE IN clauses

### Changed

- **BREAKING**: Restructured project into UV workspace with three packages:
  - `haiku.rag-slim` - Core package with minimal dependencies
  - `haiku.rag` - Full package with all extras (recommended for most users)
  - `haiku.rag-evals` - Internal benchmarking and evaluation tools
- Migrated from `pydantic-ai` to `pydantic-ai-slim` with extras system
- Docling is now an optional dependency (install with `haiku.rag-slim[docling]`)
- Package metadata checks now use `haiku.rag-slim` (always present) instead of `haiku.rag`
- Docker image optimized: removed evaluations package, reducing installed packages from 307 to 259
- Improved vector search performance through optimized score normalization

### Fixed

- ImportError now properly raised when optional docling dependency is missing

## [0.13.3] - 2024-11-04

### Added

- Support for Zero Entropy reranker
- Filter parameter to `search()` for filtering documents before search
- Filter parameter to CLI `search` command
- Filter parameter to CLI `list` command for filtering document listings
- Config option to pass custom configuration files to evaluation commands
- Document filtering now respects configured include/exclude patterns when using `add-src` with directories
- Max retries to insight_agent when producing structured output

### Fixed

- CLI now loads `.env` files at startup
- Info command no longer attempts to use deprecated `.env` settings
- Documentation typos

## [0.13.2] - 2024-11-04

### Added

- Gitignore-style pattern filtering for file monitoring using pathspec
- Include/exclude pattern documentation for FileMonitor

### Changed

- Moved monitor configuration to its own section in config
- Improved configuration documentation
- Updated dependencies

## [0.13.1] - 2024-11-03

### Added

- Initial version tracking

[Unreleased]: https://github.com/ggozad/haiku.rag/compare/0.14.0...HEAD
[0.14.0]: https://github.com/ggozad/haiku.rag/compare/0.13.3...0.14.0
[0.13.3]: https://github.com/ggozad/haiku.rag/compare/0.13.2...0.13.3
[0.13.2]: https://github.com/ggozad/haiku.rag/compare/0.13.1...0.13.2
[0.13.1]: https://github.com/ggozad/haiku.rag/releases/tag/0.13.1
