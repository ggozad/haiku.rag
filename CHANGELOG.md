# Changelog
## [Unreleased]

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
