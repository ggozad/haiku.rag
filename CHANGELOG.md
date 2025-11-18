# Changelog
## [Unreleased]

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
