# haiku.rag Project Guide

Agentic RAG system built on LanceDB with hybrid search, multiple embedding providers, reranking, and native Pydantic AI RAG + analysis capabilities.

## Quick Reference

```bash
uv sync                                    # Install dependencies
pytest                                     # Run tests
ty check                                   # Type checking
ruff check && ruff format                  # Lint and format Python
cd app/frontend && biome check             # Lint and format frontend (TypeScript/JSX)
haiku-rag <command> --db /tmp/test.lancedb # CLI with temp DB (ALWAYS use temp DB for testing)
haiku-rag mcp --stdio                      # MCP server for Claude Desktop
```

## Project Structure

**Two PyPI packages:**
- `haiku.rag` - Full package with all extras (docling, voyageai, cohere, zeroentropy, tui, cross-encoder, jina)
- `haiku.rag-slim` - Core with optional extras

**Layout:**
```
haiku_rag_slim/haiku/rag/   # Source code
├── store/                  # LanceDB persistence
│   ├── engine.py           # Store class: connection, locks, migrations, vacuum, tags
│   ├── schema.py           # Table records, Arrow schemas, index_specs/ensure_indexes
│   ├── info.py             # gather_database_info, get_database_stats, DatabaseInfo
│   ├── models/             # Domain models (Document, Chunk, SearchResult)
│   ├── repositories/       # CRUD (DocumentRepository, ChunkRepository, SettingsRepository)
│   ├── upgrades/           # Version migrations (v0_20_0 … v0_75_0)
│   └── exceptions.py       # ReadOnlyError, AmbiguousDatabaseError, UnknownDatabaseError,
│                           # AmbiguousCitationError, SourceUnavailableError, ConfigMismatchError
├── embeddings/             # VoyageAI, Cohere, vLLM (ollama/openai via pydantic-ai)
├── reranking/              # cross-encoder, Cohere, Zero Entropy, Jina, Jina-local, vLLM
├── sandbox/                # pydantic-monty sandbox used by AnalysisCapability
│   ├── sandbox.py          # Sandbox, SandboxResult, recovery_hint
│   ├── dependencies.py     # AnalysisContext (per-invocation filter)
│   └── models.py           # AnalysisResult (returned by client.analyze)
├── providers/              # External service providers
│   └── docling_serve.py    # Docling serve provider
├── capabilities/           # Native Pydantic AI capabilities
│   ├── _base.py            # Per-run state, resources, question identity and epochs
│   ├── _tools.py           # Shared capability primitives
│   ├── ledger.py           # CapabilityEvidenceRecord, citation_status
│   ├── evidence.py         # discover_evidence(), DiscoveredEvidence
│   ├── compaction.py       # EvidenceCompactionCapability, build_capsule, compact_history
│   ├── policy.py           # CitationPolicyCapability, CitationPolicyState
│   ├── rag.py              # RAGCapability and RAGState
│   ├── analysis.py         # AnalysisCapability and AnalysisState
│   └── instructions/       # Deferred model instructions
├── tools/                  # Reusable pydantic-ai FunctionToolsets
│   ├── context.py          # RAGDeps protocol
│   ├── filters.py          # SQL filter builders
│   ├── search.py           # create_search_toolset()
│   └── document.py         # create_document_toolset()
├── chunkers/               # docling-local, docling-serve
├── converters/             # docling-local, docling-serve, text_utils.py
├── config/                 # models.py (all config classes), loader.py
├── chat/                   # Chat TUI (app.py, widgets/)
├── inspector/              # Inspector TUI (app.py, widgets/)
├── client/                 # HaikuRAG high-level API
│   ├── __init__.py         # HaikuRAG class, RebuildMode
│   ├── scope.py            # DatabaseScope, DatabaseRef (which databases an operation covers)
│   ├── session.py          # SingleDatabaseSession, FederatedSession
│   ├── documents.py        # create/import/update_document(s), create_document_from_source, DocumentImport
│   ├── processing.py       # convert, chunk
│   ├── titles.py           # generate_title
│   ├── search.py           # search, expand_context, visualize_chunk
│   ├── agents.py           # ask, analyze
│   ├── rebuild.py          # rebuild_database
│   └── downloads.py        # download_models
├── ingester/               # haiku-ingester service (continuous ingestion; own CLI + [ingester] extra)
│   ├── cli.py, app.py      # haiku-ingester entry point + application layer
│   ├── queue/              # db.py, migrations.py, repository.py (JobRepo/SyncStateRepo), models.py
│   ├── workers/            # pool.py (WorkerPool), pipeline.py (run_job), retry.py
│   ├── pollers/            # base, periodic, fs, manager, factory
│   ├── api/                # FastAPI control plane (server.py, auth.py, schemas.py)
│   │   ├── routes/         # health, sources, jobs, dlq, stats, config, database, dashboard, providers
│   │   └── static/index.html # Self-contained vanilla-JS dashboard served at /
│   └── exceptions.py       # PermanentError, TransientError
├── sources/                # fs, http, s3, webdav adapters + registry, filter, walk_files
│                           # (used by the ingester AND one-shot client ingestion)
├── app.py                  # HaikuRAGApp CLI application layer
├── cli.py                  # Typer CLI entry point
├── mcp.py                  # FastMCP server (create_mcp_server)
├── doctor.py               # Database health checks, provider connectivity probes, near-duplicate detection
├── circuit_breaker.py      # CircuitBreaker (shared by ingester pollers + docling-serve provider)
├── logging.py              # Logging configuration
├── uri.py                  # is_local_uri, uri_to_path (file:// and bare-path handling)
└── utils.py                # Utilities (get_model, format_citations_rich, raise_missing_extra)
tests/                      # Mirrors source structure
├── conftest.py             # Fixtures: temp_db_path, qa_corpus, temp_yaml_config
├── cassettes/              # VCR recorded HTTP responses
├── json_body_serializer.py # Custom VCR serializer for JSON bodies
├── store/                  # Store-specific tests
├── data/                   # Test data files
└── utils/                  # Test utilities
evaluations/                # Benchmarking workspace
├── evaluations/
│   ├── benchmark.py        # Typer CLI: run, download, upload
│   ├── population.py       # populate_db
│   ├── retrieval.py        # run_retrieval_benchmark
│   ├── qa.py               # run_qa_benchmark, run_live_qa_benchmark, Target
│   ├── artifacts.py        # download_dataset_db, upload_dataset_db, HF_REPO_ID
│   ├── experiment.py       # build_experiment_metadata
│   ├── config.py           # DatasetSpec, DocumentPayload, RetrievalSample
│   ├── capability_runner.py # native capability evaluation runner
│   ├── datasets/           # mtrag, open_rag_bench, hotpotqa, t2_ragbench
│   └── evaluators/         # judge.py (LLMJudge), map.py, citation.py
└── tests/                  # benchmark and capability-runner tests
docs/                       # Documentation (zensical, not mkdocs; nav lives in zensical.toml)
examples/                   # Working examples
├── docker/                 # Docker deployment
└── samples/                # Sample documents
app/                        # Conversational RAG application (see below)
```

## Architecture

**Key Classes:**

| Class | Location | Purpose |
|-------|----------|---------|
| `HaikuRAG` | client/__init__.py | High-level API (main entry point) |
| `HaikuRAGApp` | app.py | CLI application layer |
| `Store` | store/engine.py | Async LanceDB connection, tables, upgrades |
| `DatabaseScope` | client/scope.py | The databases an operation covers, resolved once |
| `SingleDatabaseSession` | client/session.py | One database: store, repositories, lifecycle. Reads and writes |
| `FederatedSession` | client/session.py | Several, composed lazily. Reads only |
| `DocumentRepository` | store/repositories/document.py | Document CRUD |
| `ChunkRepository` | store/repositories/chunk.py | Chunk CRUD + search |
| `SettingsRepository` | store/repositories/settings.py | Config persistence |

**Factory Functions:**
- `get_embedder(config)` → `EmbedderWrapper` (embeddings/__init__.py)
- `get_reranker(config)` → `RerankerBase | None` (reranking/__init__.py)
- `get_converter(config)` → `DocumentConverter` (converters/__init__.py)
- `get_chunker(config)` → `DocumentChunker` (chunkers/__init__.py)
- `create_capability(db_path?, config?, rag=None, defer_loading=True, request_limit=20|30, vision=None)` → `RAGCapability | AnalysisCapability` (capabilities/rag.py, capabilities/analysis.py). `request_limit` is a per-question cap: at the limit only the exhausted capability's tools are removed, except its cite tool, which survives `CITATION_GRACE_REQUESTS` (2) further requests that call one of that capability's tools so an exhausted run can still register citations. Turns spent on another capability do not count against that window. A spent search or code budget does not withdraw the tool — it keeps failing, because withdrawing a tool the model calls anyway costs the agent's unknown-tool retries and aborts the run. `vision` gates image attachment on search results and must reflect the model the hosting agent runs; defaults to the configured model's `vision` flag. Pass `rag` to lend the capability an open client instead of letting it open its own: it lands in `borrowed_rag`, which `_ensure_rag` prefers and `_close` never closes, unlike the `rag` it opens itself. Each capability also overrides `from_spec`, delegating here so pydantic-ai agent specs can declare it; `db_path` accepts a `str`.
- `create_capability()` → `EvidenceCompactionCapability` (capabilities/compaction.py). Optional; registering it is the only switch. Replaces earlier questions' evidence on the *request* with a capsule of what was cited (pictures included, fetched through the owning capability), other earlier evidence returns with a receipt. No config, no budget: cited evidence is kept whole, so this reduces a request without bounding it.
- `create_capability()` → `CitationPolicyCapability` (capabilities/policy.py). Optional; requires every answer to declare its grounding. Asks once per question, records failures in `CitationPolicyState.violations`, and never asks the model to change its answer. Enforced whenever the question has something to declare: it retrieved evidence, or the conversation already cited something.
- `create_search_toolset(config)` → `FunctionToolset[RAGDeps]` (tools/search.py)
- `create_document_toolset(config)` → `FunctionToolset[RAGDeps]` (tools/document.py)

**Base Classes:**
- `EmbedderWrapper` (embeddings/__init__.py) — wraps pydantic-ai `Embedder`
- `RerankerBase` (reranking/base.py)
- `DocumentConverter` (converters/base.py)
- `DocumentChunker` (chunkers/base.py)

**Domain vs Persistence:**
- Domain: `Document`, `Chunk`, `SearchResult` (Pydantic BaseModel in store/models/)
- Persistence: `DocumentRecord`, `ChunkRecord`, `SettingsRecord` (LanceModel in store/engine.py)

## HaikuRAG Client API

Key methods on `HaikuRAG` (client/__init__.py):

```python
async with HaikuRAG(db_path, config, create=True) as rag:
    # Document operations
    doc = await rag.create_document(content, uri, title, metadata)
    doc = await rag.create_document_from_source(path_or_url, title, metadata)
    doc = await rag.import_document(docling_doc, uri, title, metadata)
    docs = await rag.import_documents([DocumentImport(docling_doc, chunks, uri, title, metadata), ...])  # batch: one table version per table
    doc = await rag.get_document_by_id(id)
    doc = await rag.get_document_by_uri(uri)
    docs = await rag.list_documents(limit, offset, filter)
    count = await rag.count_documents(filter)
    await rag.update_document(id, content, uri, title, metadata)
    await rag.delete_document(id)

    # Title generation
    title = await rag.generate_title(doc)

    # Search & QA
    results = await rag.search(query, limit, filter)
    expanded = await rag.expand_context(results)
    answer, citations = await rag.ask(question, filter=None, images=None)  # images: Sequence[bytes], needs vision: true on the model
    result = await rag.analyze(question, filter=None, images=None)  # AnalysisResult

    # Document resolution
    doc = await rag.resolve_document(id_or_title)
    chunk = await rag.get_chunk_by_id(chunk_id)

    # Maintenance
    async for doc_id in rag.rebuild_database(mode=RebuildMode.FULL|TITLE_ONLY): ...
    await rag.vacuum()
    async for progress in rag.download_models(): ...

    # Tags (Store-level; TagInfo has tables/missing_tables/complete)
    await rag.store.create_tag("release-1")
    tags = await rag.store.list_tags()
    safety_tag = await rag.store.restore_tag("release-1")
    await rag.store.delete_tag("release-1")

    # Visualization
    images = await rag.visualize_chunk(chunk)
    # Coverage (see "Multiple Databases")
    rag.covers_multiple      # more than one database
    rag.source_names         # configured names covered, in order
    rag.source               # the one name, or None while covering a set
    rag.location             # configured URI or path, None while covering a set
    owner = await rag.reader_for("papers")          # the client reading that database
    papers, wiki = await rag.clients_for(["papers", "wiki"])
    covering = await rag.clients_covering(sources)  # honours a per-query selection
```

## Multiple Databases

`lancedb.databases` maps a name to a location. The name is the only identity that
leaves the configuration: it travels in `SearchResult.source`, `Citation.source`,
`Document.source` and errors, where a location must not.

- **Resolution happens once.** `DatabaseScope.resolve(config, database_name=, database_path=)`
  is the single place selection happens, and it never mutates config. Every
  database has a name: the `lancedb.databases` key, or the path's stem. A name
  selects one configured entry; a path places a database only where the
  configuration places none, and beside `lancedb.databases` raises
  `AmbiguousDatabaseError`; neither selector covers the configured set, a set of
  one included. Nothing configured is the one entry `haiku.rag` at
  `storage.data_dir / "haiku.rag.lancedb"`, selectable by name. The CLI's `--db`
  is the one human override: `DatabaseScope.at(path)` consults no configuration.
  `DatabaseRef.given` marks a caller-given path whose errors may name it; a
  configured or default database's errors name the database and never its
  location (`SourceUnavailableError`, with the remedy for a missing one).
- **Two sessions, two search paths.** `SingleDatabaseSession` reads and writes;
  `FederatedSession` composes single sessions lazily and reads only. A selection of
  one runs the ordinary single-database search — fusion would replace the
  database's hybrid scores with ranks, and embedding up front would embed for a
  filter the repository can see matches nothing.
- **Fusion.** With a reranker, it scores the union directly. Without one, the
  union is ordered by cosine similarity to the query vector — the selection
  shares an embedder, so similarity in that one space is comparable across
  databases by construction; breadth is not guaranteed. Full-text-only
  searches (no query vector) order by retrieval score. Ties resolve by
  within-database rank, and only a tie on both falls to configured order.
  Fused results carry the ordering score, so downstream re-sorts (context
  expansion) preserve fused order. `Chunk.embedding` is populated only when
  cosine fusion will read it (`with_vectors` mirrors `_fuse`'s cosine
  condition; an image query with a reranker configured still takes cosine).
  Over-fetch (`limit * 10`) only when a reranker is configured. Image queries
  are vector-only and skip the reranker: its interface takes a text query,
  and `reranking.multimodal` is candidate-side pictures only.
- **One reranker per set.** A client covering a database for another borrows the
  lender's (`_lender`, `_own_reranker`); only the owner closes it.
- **Collisions.** Chunk ids repeat between copies of a database. In-memory
  identity uses `qualified_id(source, id)`; serialized structures cannot represent
  it, so a cited id retrieved from, or previously cited from, more than one
  selected database raises `AmbiguousCitationError`. A copy the search never
  returned grounded nothing, so one retrieved result resolves normally.
- **Model context** names the database as a `Collection:` line, only when the
  search spans more than one. Storage vocabulary stays "database"; the model
  boundary says "collection".

## Configuration

**Config classes** (config/models.py):
- `AppConfig` - Root config
- `StorageConfig` - data_dir, auto_vacuum, vacuum_retention_seconds
- `EmbeddingsConfig` / `EmbeddingModelConfig` - provider, name, vector_dim, multimodal (bool, gates image embedding)
- `RerankingConfig` - optional reranker model
- `ModelConfig` - base model configuration (model name)
- `QAConfig` - model, max_searches (default 5; bounds the search tool of **both** capabilities, and `qa.model` is the analysis capability's fallback model). `max_searches` counts search *units*: searches emitted in one model response (same `RunContext.run_step`) share a unit, up to `FREE_SIBLINGS_PER_ROUND` (3) per unit — searches 1, 4, 7… of a round increment `search_count` — so total searches per run cap at `max_searches × 3` and a sequential searcher pays one unit per search. A budget-rejected round fails its remaining siblings. A separate `analysis.max_searches` was tried and reverted: raising it bought no accuracy for +44% wall-clock
- `AnalysisConfig` - analysis capability settings (model — defaults to None, falls back to qa.model; code_timeout, max_output_chars, max_executions). Searching from inside `analysis_execute_code` does not count against `qa.max_searches`, so code execution outlives a spent search budget as a way to reach new evidence
- `PictureDescriptionConfig` - picture description model settings
- `ProcessingConfig` - chunk_size, converter, chunker, conversion_options, auto_title, title_model
- `SearchConfig` - limit, max_context_chars, vector_index_metric, vector_refine_factor, vector_nprobes
- `DoctorConfig` - duplicates (DuplicateDetectionConfig: similarity_threshold 0.97, min_chunks 3)
- `ProvidersConfig` - ollama, docling_serve settings (OllamaConfig, DoclingServeConfig)
- `PromptsConfig` - prompt configuration
- `IngesterConfig` - sources, queue (QueueConfig), workers (WorkerConfig), api (APIConfig)
- `LanceDBConfig` - databases (name → local path or URI), api_key, region, storage_options, read_consistency_interval_seconds, index_cache_size_bytes, metadata_cache_size_bytes

Every model inherits `ConfigModel` (`extra="forbid"`), so an unknown or misspelled key raises at load; converter, chunker and chunker_type are `Literal`s and numeric fields carry bounds, while provider fields stay unrestricted `str` (`get_model` passes an unknown provider through to pydantic-ai). Read config through `get_config()` — there is no module-level `Config` singleton, and capturing the config in a default argument freezes it before `set_config()` runs.

**Search order:** CLI `--config` → `./haiku.rag.yaml` → platform user directory

**Global CLI options:** `--db-name NAME` selects one entry from
`lancedb.databases` and is global, so it precedes the command; `--db PATH` is
per-subcommand and follows it. They are mutually exclusive. `search`, `ask`,
`analyze`, `chat` and `mcp` cover the configured set; everything else works on
one database. `settings`, `init-config` and `download-models` resolve no scope at all,
so a name is nothing to them.

## CLI Commands

```
# Document management
add           Add document from text
add-src       Add from file/URL/directory
get           Get document by ID
delete/rm     Delete document by ID
list          List documents (with optional filter)

# Search & QA
search        Hybrid search (vector + full-text)
ask           QA via the RAG capability (always shows citations; --image PATH repeatable)
analyze       Analysis via the analysis capability (Python sandbox with document VFS; --image PATH repeatable)

# Maintenance
init          Initialize new database
rebuild       Re-chunk and re-embed all documents (--title-only, --embed-only, --rechunk)
vacuum        Optimize and clean up tables
migrate       Database migration
create-index  Create vector index for similarity search
info          Show database info (read-only)
history       Show version history for tables (tagged versions annotated)
tag           Manage database tags: tag create/list/delete/restore NAME
doctor        Check database health, probe provider connectivity, detect near-duplicate documents (exits 1 on failure; --duplicates-out exports groups to YAML)

# Configuration
settings      Display current config
init-config   Generate YAML config file

# Server
mcp           Run the MCP server (--stdio, --host, --port); read-only, covers the configured set
              # Continuous ingestion is the separate `haiku-ingester` service

# Tools
visualize     Show visual grounding for a chunk
inspect       Launch TUI to inspect database
chat          Launch TUI for conversational RAG
download-models  Download Docling and Ollama models
```

**Tags:** `haiku-rag tag create/list/delete NAME` name database states across all five tables; `tag restore NAME` brings the live database back to a tagged state (creates a `before-restore-*` safety tag first; requires stopped writers; never migrates). Tagged versions survive vacuum: it retains the oldest tagged version and everything newer, while versions older than the oldest tag stay eligible for cleanup.

## MCP Server Tools

Exposed via FastMCP (mcp.py). `create_mcp_server(db_path?, config?)` resolves a
scope; `_covering(scope, config)` takes one already resolved. The server opens
one read-only client for its lifetime and covers the configured database set, so
every tool takes `sources` (or `source`) by name. There are no write tools:
ingest with `haiku-rag add`/`add-src`/`delete` or `haiku-ingester`.

- `search_documents(query, limit?, include_images?, filter?, sources?)` → `ToolResult`
- `search_documents_by_image(image_base64, limit?, include_images?, filter?, sources?)` → `ToolResult` (registered only when the embedder `supports_images`)
- `get_document(document_id, source?)` → Document
- `get_document_outline(document_id, source?)` → list[OutlineNode]
- `get_document_section(document_id, section_id, source?)` → DocumentSection
- `list_documents(limit?, offset?, filter?)` → list[DocumentInfo]
- `execute_code(code, filter?, sources?)` → the program's stdout

**Search results are text, not JSON.** Both search tools expand through
`HaikuRAG.expand_context` and return the `format_for_agent` rendering (rank,
`Document ID`, `Collection` over several databases, matched chunk metadata,
passage) plus one `ImageContent` per distinct picture. They deliberately carry
no structured content — see the gotcha below. Every other tool returns a
pydantic model, whose JSON is the payload.

**`execute_code`** runs one program per call in the same Monty sandbox the
analysis capability uses, over the documents `filter` and `sources` select, and
returns what it printed. One sandbox per call: a session outliving the call
would hit Monty's cumulative duration budget and never see documents ingested
after its first mount. Where the client is already a model the server's value is
the sandbox, not a second model, so there is no `ask_question` or `analyze` tool
and `haiku.rag.utils` has no `format_citations` (`format_citations_rich` and the
`_citation_*` helpers stay, for the CLI).

**One error contract.** `mask_error_details=False`, passed explicitly since the
setting is also read from the environment: every failure reaches the client as
an MCP error carrying its message, host errors inside a program included. A bad
filter fails in the read itself with the query engine's own message and an
unknown collection with `UnknownDatabaseError`'s, so there is no pre-check
count query and no `ToolError` translation for either. Explicit domain errors
stay (`No document with id`, `No section`, `Invalid base64 image`).

**Plugin bundle.** `plugins/haiku-rag/` holds one `.mcp.json` (`haiku-rag mcp
--stdio`) and one Agent Skill, `skills/haiku-rag/SKILL.md`, under two manifests:
`.claude-plugin/plugin.json` and `.codex-plugin/plugin.json`. Marketplaces are
`.claude-plugin/marketplace.json` (Claude Code) and `.agents/plugins/marketplace.json`
(Codex). Both manifests carry the package version, `scripts/bump_version.py`
rewrites both, and `tests/test_bump_version.py` pins them to
`haiku_rag_slim/pyproject.toml`. A test asserts the skill's `allowed-tools`,
prefix stripped, equal the tool set the server registers, so a renamed tool
fails until the skill follows. The shared `.mcp.json` uses Claude's
`mcpServers` key, which Codex's documentation does not list but its loader
accepts (proven by a real install) — don't "fix" it to `mcp_servers`, the file
serves both. `allowed-tools` is Claude Code's pre-approval and every other
Agent Skills client ignores it.

## Testing

**Key fixtures** (conftest.py):
- `temp_db_path` - Isolated temp database (use this!)
- `qa_corpus` - HuggingFace ServiceNow/repliqa dataset
- `temp_yaml_config` - Temp config file, pointed at via `HAIKU_RAG_CONFIG_PATH`
- `allow_model_requests` - Enables pydantic-ai model calls (disabled by default)

**Markers:**
- `@pytest.mark.asyncio` - Async tests
- `@pytest.mark.integration` - External service tests. CI excludes them (`-m "not integration"`); they RUN locally by default. Start services with `docker compose -f tests/docker/docker-compose.yml up -d` (postgres, docling-serve, seaweedfs); each test skips if its service isn't reachable.
- `@pytest.mark.vcr()` - HTTP call recording/replay

Tests run under xdist (`-n auto` in addopts); pass `-n0` to disable when debugging races or container lifecycle. asyncio fixtures default to **session** loop scope (`asyncio_default_fixture_loop_scope = "session"`).

**Coverage is enforced at 100%** (`fail_under = 100` in `[tool.coverage.report]`). New code needs a test or a `# pragma: no cover - <short reason>` on one line (never a wrapped multi-line comment). CI prints term-missing, so a gate failure names the line.

**VCR Recording:**
Tests use pytest-recording (VCR.py) to record and replay HTTP calls. Cassettes are stored in `tests/cassettes/` and committed to the repo, allowing CI to run without external services.

```bash
# Run with recorded cassettes (default)
pytest

# Record new cassettes (exact test, serial, real services available)
pytest tests/test_embedder.py::test_ollama_embedder -n0 --record-mode=rewrite

# Run against live services (disable VCR)
pytest --disable-recording

# Record with a real API key (each SDK reads its own variable, e.g. CO_API_KEY)
CO_API_KEY=... pytest tests/test_reranker.py::test_cohere_reranker -n0 --record-mode=rewrite
```

To add VCR to a new test, add `@pytest.mark.vcr()`, then record its cassette with `pytest <path> -n0 --record-mode=once` against the real service. The default `record_mode` is `none` (none is set in pyproject), so without `--record-mode` an unrecorded call errors with a connection-style failure instead of recording. Recording reaches the real service, so it needs network.

## Code Conventions

- Python 3.12+ native typing (no `from __future__ import annotations`)
- Absolute imports only (no relative imports)
- Async/await for all I/O operations
- LanceDB is opened via `lancedb.connect_async`; all table reads/writes (`open_table`, `query`, `add`, `delete`, `merge_insert`, `list_tables`, `drop_table`) are awaited. `connect_async` requires absolute paths — `connect_lancedb` makes a local location absolute before connecting. `Store(location, config)` takes a path or a URI; `Store.db_path` is `None` behind a URI.
- No module docstrings at top of .py files
- Type hints everywhere, Pydantic for validation
- Minimal comments (only explain WHY when not obvious)

## Comments and Docstrings

**This section overrides "match the surrounding style".** Much of the existing
prose in this repo is longer than these rules allow; do not take it as the
standard to match.

- A docstring is one line by default. More than three lines needs a contract a
  caller cannot see from the signature — an ordering requirement, a value that
  looks wrong and is not, an invariant the caller must hold.
- A comment states what is true, never what used to be true, what would break
  without it, or what alternative was rejected. "would otherwise", "instead of",
  "rather than", "so that X does not" are the tells.
- Rationale, measurements and alternatives go in the commit message and the PR
  description. Not in the code.
- Test docstrings follow the same rules. Name the behaviour, not the bug.

Before proposing a commit, read the added comment lines on their own:
`git --no-pager diff -U0 --no-color | grep -E '^\+' | grep -E '#|"""'`

## Critical Rules

- ALWAYS use `--db /tmp/test.lancedb` when testing CLI commands
- Run `ruff check` and `ty check` after Python changes
- Run `biome check` in `app/frontend/` after frontend changes
- Update docs/ and README.md when changing functionality
- Online docs: https://ggozad.github.io/haiku.rag/

## Common Gotchas

- Tags are per-table lance refs; a haiku.rag tag = the same name on all five tables from one `current_table_versions()` snapshot. lance restricts ref names to alphanumeric/`.`/`-`/`_` (RuntimeError otherwise). `AsyncTable.tags` is a property minting a fresh `AsyncTags` per access — fault-injection tests must patch `AsyncTags` methods at class level with call counting.
- `optimize(cleanup_older_than=...)` hard-errors when a tagged version falls in the cleanup window, and the lancedb Python API does not expose `error_if_tagged_old_versions`. `Store.vacuum` grows the effective retention so the cutoff stays older than the oldest tag (`TAG_RETENTION_MARGIN` guards the boundary) and suppresses `OSError` only — every `RuntimeError` re-raises.
- Multi-table writes go through `async with store.write_transaction():` — it takes `_write_lock`, snapshots `current_table_versions()`, and on any exception restores every table in `RESTORE_TABLE_ORDER` under a shield before re-raising (an absorbed `CancelledError` is re-delivered). Client code never restores tables itself.
- Lock ordering is `_rebuild_lock → _write_lock` (tag ops, restore) and `_vacuum_lock → _write_lock` (vacuum); rebuild holds `_rebuild_lock` for its whole run and tag operations fail fast against it. All of this is in-process only — cross-process writers need a quiescent boundary for tag create/restore.
- lance 8 `merge_insert` rejects partial-schema sources upfront when an insert branch requires non-nullable columns the source lacks — declare update-only merges (drop `when_not_matched_insert_all`) when matches are guaranteed (bit the v0.45.0 migration backfill).
- HF Hub outages stall unmarked-network tests at setup (chunking-tokenizer fetch retries on 5xx); `HF_HUB_OFFLINE=1 pytest ...` runs from the disk cache and looks like a hang otherwise.

- Each `create_document*` / `import_document` writes a NEW version of the `documents`, `chunks`, AND `document_items` tables — one version per doc. Bulk-ingesting in a loop fills disk with table versions. Use `import_documents([DocumentImport(...)])` (writes each table once and embeds all missing chunks in one pass for the whole batch) or `DocumentRepository.create([...])` / `DocumentItemRepository.create_all(...)` at the repo layer.
- Opening a `Store` never writes to it: no settings sync, no version write on open. Stored embedding settings change only via `init`/`rebuild`; the version is a forward-only schema marker written only by `init` and `migrate` (never on open, never downgraded).
- Embedding drift on open: `vector_dim` mismatch always raises `ConfigMismatchError`; provider/name drift (same dim) warns in read-only mode and raises in writable mode — reconcile with `haiku-rag rebuild --set-embedder` (rewrites stored identity, no re-embed). Read verbs open read-only; maintenance (`rebuild`/`vacuum`/`migrate`/`create_index`/`info`/`history`) and CLI document deletion pass `skip_validation=True` to open despite drift. The MCP server does NOT: it opens one validating read-only client for its lifetime, eagerly in the lifespan, so an unopenable database or a `vector_dim` mismatch fails startup instead of serving reads that all return empty. It still starts on same-dim identity drift, which only warns. Deleting under drift is a CLI operation.
- Index sets live in `index_specs()` / `ensure_indexes()` (schema.py); every path that creates a table routes through them. `ensure_indexes` matches on (column, index *type*) and never drops or converts an index it did not declare, so an operator's extra index on a declared column survives. The chunks FTS index is the exception to create-at-table-creation: `ensure_indexes` skips FTS while the table is empty (an index born over zero rows never catches up on `add`), the chunk write paths (`create`, `replace_for_document`, `delete_by_document_id`, embed-only rebuild per flush) call it after writing, and it rebuilds an FTS index whose `num_indexed_rows` is 0 (or whose stats are None) over a populated table. Consequences: a test needing the zero-coverage state must write rows via `chunks_table.add` directly — any repository write heals it; the first chunk write into a fresh table writes one extra chunks table version (the index build); tag restore brings back the tagged index state, deliberately.
- `create_index(replace=True)` REBUILDS an identical index: new index uuid, new table version, and the old index files stay until the next vacuum. It also matches by index *name*, and the default name is `{column}_idx`, so a custom-named index on the same column is invisible to it and gets a second index built beside it, both maintained on every write. Always check `list_indices()` first. On lancedb 0.38+ an unnamed `create_index(replace=True)` no longer replaces a different-typed index on the same column — it builds a suffixed sibling (`label_idx_2`); pass `name=` explicitly (`ensure_indexes` does).
- docling and docling-serve are pinned as a PAIR, exactly: `docling==2.124.0` + `docling-core==2.93.0` client-side, `quay.io/docling-project/docling-serve:v1.32.0` in `tests/docker/` and `examples/docker/` (three references). The coupling is not the wire format — `DoclingDocument` has stayed schema `1.10.0` since 2.102 — it is `tests/test_chunker.py::test_local_and_serve_chunkers_produce_same_output`, which asserts chunk-for-chunk equality of content, `doc_item_refs`, `labels`, `headings` and `page_numbers` between the two converters, and is vcr-marked so it runs in CI. Each docling-serve release publishes the library set it bundles; take the client version from there and move both together. Re-record the serve cassettes when the image moves: that test plus the six `TestDoclingServeConverterIntegration` cassettes, which are vcr-marked and NOT integration-marked despite the name, so they otherwise keep replaying whatever server was running when they were recorded. `:latest` in a compose file is a trap — Docker never re-pulls it, and ours sat at 1.25.0 from June while upstream was at 1.32.0.
- `processing.conversion_options.pdf_backend` defaults to `docling_parse`, NOT docling's own default `threaded_docling_parse`. On some documents the threaded backend never returns: every Python thread parks in `standard_pdf_pipeline.get_batch` on `self._not_empty.wait()` with no timeout, while a native parser thread spins where faulthandler cannot see it (three dumps 20s apart, identical frames, 31 threads at 110% CPU). Linux x86_64 and linux/arm64 hang; macOS/arm64 does not, so reproduce it in the `docling-serve:v1.32.0` image, which is linux/arm64 and carries the pinned stack — you are root in the container, so `faulthandler.register(SIGUSR1, all_threads=True)` plus `kill -USR1` works without ptrace. docling caches its `DocumentConverter` per process, so one wedge stalls later conversions *on any backend*; recovery needs a new process, never a retry. Both converters are sent the value, so the default is a free choice and the parity test passes at any of them. Measured over ten arXiv papers, `docling_parse` finds the most tables and cells for ~9% more wall time; `pypdfium2` loses 7% of words and 28% of table cells (its higher item count is fragmentation).
- Table differences between docling 2.102.2 and 2.124.0 were the BACKEND, not `docling-ibm-models` 4.0.1. Hold `pdf_backend` constant before attributing a difference to a version. Held constant, the version change is near-null: 3,571 tables against 3,572 over 1000 arXiv papers, and on the 9-page test paper markdown words 7,360 → 7,361 and `#/tables/3` 55 → 55 cells. The only real 2.124.0 change there is item merging, 141 items → 130 carrying the same text.
- `processing.conversion_timeout` (600s) bounds one conversion, and every part of the semantics is deliberate — do not "simplify" any of it. The deadline starts when the conversion *holds* `_CONVERTER_LOCK`, not when it was requested (`worker_count` defaults to 4, so charging the wait divides the budget by the contenders and dead-letters a document that only queued). The lock is owned by the conversion thread, never by the coroutine: cancellation cannot reach a thread, so coroutine ownership leaks the lock on a cancel during the wait and releases it under a running conversion on a cancel after. The deadline runs in a task the caller `asyncio.shield`s, or a cancelled caller takes the watcher with it and a stall is never recorded. Conversions run on daemon threads through `contextvars.copy_context().run` — `asyncio.run` joins the default executor on teardown and interpreter exit joins its non-daemon workers, so `asyncio.to_thread` lets one stall hold the process. A queued conversion polls `acquire(timeout=0.5)` and re-checks `_WEDGED` between attempts and after acquiring, since a thread already blocked on the lock would otherwise never learn the converter is stranded, and a late-returning stall would hand it a converter the process has declared dead.
- Ingester recovery for a stalled conversion: `PermanentError` carries `conversion_stalled` (tombstone this document) and `fatal_to_process` (this process must exit) as separate flags, because an HTML/Markdown stall strands no shared converter. The worker records the job dead BEFORE exiting — `reap_stale` refunds the attempt of a claim whose owner vanished, so exiting first returns the poison document to the queue with its attempts intact, forever — and the exit runs from a `finally`, so it happens when `mark_dead` returns False and when it raises.
- Queue schema 3: `jobs.conversion_stalled` plus `uq_jobs_blocking_op`, a partial unique index over `(source_id, uri, op, coalesce(revision, ''))` covering live and stalled jobs. Enforced by the index, not a read: no read is atomic against a worker committing `mark_dead` beside it, and SQLite's deferred transaction takes no write lock for a `SELECT` (the queue engine runs `pool_size=5`, not 1). `coalesce` because a unique index treats NULLs as distinct in both dialects, which would exempt exactly the revision-less rows the tombstone exists for. `prune_terminal` skips these rows; a new revision, a DELETE, `retry` and `prune_dead` free the slot. The v3 migration emits the index from the same `Index` object `create_all` uses, so fresh and migrated schemas cannot drift.
- `TestConversionTimeout` needs a FRESH `_CONVERTER_LOCK` per test (`monkeypatch.setattr(docling_local, "_CONVERTER_LOCK", threading.Lock())`), never a forced `release()`: a stalling stub thread still owns the lock, so releasing it admits another test's conversion while that thread is inside and the stub's own `with` raises on exit. `_run` reads the module global at call time, so a leftover thread holds the old object. Stall stubs stay bounded by a `threading.Event` so threads do not accumulate across a session.
- `processing.split_pages` no longer reproduces single-pass conversion. docling now orders and joins text across a whole document, and a slice sees only its own pages, so a paragraph spanning a boundary stays two items and a caption near one can order differently. Measured on the 9-page doclaynet fixture: ~1 extra text item and ~1 differing chunk per boundary (slice_size 1 → 6 extra items / 8 boundaries; 2 → 5 of 77 chunks differ; 3 → 2 of 77), with the word multiset identical either way at every slice size tried (1, 2, 3), so content is only regrouped and reordered, never lost, duplicated or altered. Whether a boundary costs anything depends on what crosses it — `slice_size=8` on 9 pages is still byte-identical while `slice_size=5` is not — so equality is guaranteed only with no boundary at all, which is what the test uses to pin `concatenate`'s re-indexing and page_delta. Single-pass is the better output; `split_pages` trades that for a memory ceiling.
- rapidocr 3.9.2 ships LaTeX in a docstring (`\d` in `multiheadAttention.py`), and beartype's import hook compiles it, so pytest's `filterwarnings = ["error", ...]` turned the `SyntaxWarning` into a fatal `SyntaxError` in every docling-local PDF test. `pyproject.toml` carries a message-scoped `"ignore:invalid escape sequence:SyntaxWarning"` for it.
- docling item LABELS are platform-dependent and cannot be made to agree: a label survives only if the layout classifier clears a **hard-coded 0.5 confidence cutoff** with no flag to move it, so a borderline element flips between machines. Measured on `tests/data/doclaynet.pdf` page 1, the copyright line: local footnote confidence 0.51238, docling-serve 0.49788. The server drops the label and substitutes a synthetic text item at confidence 1.0 for the unassigned parser cell, which is why its document carries two more body texts than local's while every extracted item matches. Ruled out as causes: docling/docling-core/docling-parse/docling-ibm-models versions, all 22 comparable `ConvertDocumentsOptions`, layout model weights (`refs/main` 8f39ad3c on both), transformers/tokenizers/safetensors versions, accelerator device (MPS and CPU), `parser_threads` and torch threads. So a local/serve parity test must compare items, item text, page numbers, chunk count, chunk text and word multisets — and exclude labels and `self_ref` (which indexes the synthetic items too).
- PDF glyphs whose index or name maps to no printing character leak into chunk text, and nothing between the converter and the stored chunk removes them: measured on docling 2.124.0 / docling-parse 7.17.0, a BEL (`U+0007`) where a symbol-font list marker was, a tab where a space was in list markers and headings, and `/tildelow` tokenised as `-` plus `tildelow`. Older docling leaks these too, just fewer, so it is not a regression. Deliberately NOT fixed (issue #613, closed 2026-09-08 pending evidence of real harm): retrieval is unaffected (four of six differential documents were word-for-word identical), JSON escapes `\u0007`, and section-bounded expansion keys off the integer `heading_level`, not heading strings. What it does reach is `_contextualize_content` (`store/repositories/chunk.py:69`), which prepends headings into the FTS-indexed text, and the citation locator — where a BEL is the terminal bell. If it ever needs fixing: `Cc` is a closed Unicode category so one predicate covers every control character permanently, but a tab is legitimate whitespace (a normalization question, not a strip) and a glyph name is indistinguishable from real text, so the three do NOT share one fix. Don't blanket-strip `Cf` alongside `Cc` — soft hyphen interacts with dehyphenation. The seam is NOT `converters/text_utils.py::prepare_text_content`, which sits in the text-extensions branch and never sees a PDF; the candidates are the two chunkers' `content=` sites (`chunkers/docling_local.py:153`, `chunkers/docling_serve.py:194`), which leave the artifact in `DocumentItemRecord.text` and the stored blob, or a post-conversion pass over the `DoclingDocument`, which needs a funnel since conversion is invoked from five call sites plus `import_document`.
- Inline markup puts a paragraph's runs in an `InlineGroup`, and `iterate_items()` skips groups, so every consumer saw loose runs and an owner with empty `text`. `converters/base.py::flatten_inline_groups` normalizes this at the three points a `DoclingDocument` is built (both `docling_local` sync paths, `docling_serve._parse_zip_to_docling`) plus `_rebuild_rechunk`, which flattens the stored blob and writes it back. Only html, md, msword, opendocument, boxnote, webvtt and jats emit inline groups — PDF, PPTX, XLSX and image inputs produce none. The backends parent a paragraph to the *heading above it*, so the item a group belongs to is the one whose `text` is empty, never merely the parent. `import_document` is deliberately NOT normalized: its chunks are the caller's and their `doc_item_refs` index into the document as passed, which flattening renumbers.
- Mutating a `DoclingDocument` is O(document) per call: `delete_items` renumbers every ref, so deleting per group cost 90s of a 95s `CHANGELOG.md` conversion where one batched call costs 6.6s. `DocSerializer` also caches excluded refs keyed by `self_ref` (`serializer/common.py:253`), so a serializer must not be reused across mutations — serialize everything first, then mutate. Never give a merged item several `ProvenanceItem`s: `serializer/doclang.py:700` splits any item with `len(prov) > 1` into one fragment per record, slicing `item.orig` by `charspan`, so N records emit the text N times. A re-rendering has no offsets to rebuild charspans from; leave such a group unflattened instead. haiku.rag reads only `page_no` from prov (`chunkers/docling_local.py:134`, `store/models/document_item.py:156`) and never `charspan`.
- docling 2.120+ routes HTML image loading through `docling/backend/utils/image_resource_loader.py`, and `HTMLDocumentBackend._load_image_data` is now a thin wrapper the emit path does NOT use — `_emit_image` → `_create_image_ref` → `ImageResourceLoader.create_image_ref` → the loader's own `load_image_data`. Monkeypatch `ImageResourceLoader.load_image_data` (signature `(self, src_loc, base_path)`), not the backend method. The loader also gates fetching behind `enable_local_fetch` / `enable_remote_fetch`, both defaulting False; our HTML and Markdown backend options set `enable_remote_fetch` from `processing.conversion_options.fetch_remote_images`.
- lancedb pin bumps: read the PyPI wheel list per platform first, and check upload times — 0.38.0 published macos-arm64 and linux-aarch64 on 2026-08-31 and manylinux-x86_64 and win_amd64 only on 2026-09-01, so a bump attempted inside that window installs on an Apple Silicon Mac and dies on x86_64 CI. Neither version ships a macos-x86_64 wheel. A missing FTS index and one covering zero rows behave identically at 0.37.1, the current pin: every match is returned, recall exact, but unsorted by `_score` and in insertion order, so `limit` slices arbitrarily and hybrid fuses that row order through RRF as a ranking, displacing the dense arm's hits — worse than dense-only. Single-term probes on small tables look healthy. `ensure_indexes`' zero-coverage FTS rebuild is load-bearing; lance 11 (0.38.0) is said to fix the scan path, untested here. FTS BM25 scores changed for stop-word-heavy queries between 0.34 and 0.37.1, so retrieval benchmark numbers are pin-conditional.
- A BTree over 5M UUID strings builds in ~2s at lance's default memory pool. `LANCE_MEM_POOL_SIZE` is not needed for scalar index builds at that scale.
- `optimize()` on a table that carries an index writes a table version where an unindexed table wrote none. Since `documents` is now indexed, an `auto_vacuum` pass touches it — a test measuring table-version deltas must set `storage.auto_vacuum = False` or the background vacuum lands inside the window it measures.
- The lancedb `Session` is process-wide and shared across connections (`connect_lancedb`), and holds the index and metadata caches. On object storage the first vector query loads the index into it (~135 MB for a 500k-chunk 2560-dim corpus), so that cost is paid once per process rather than per connection. `lancedb.read_consistency_interval_seconds` defaults to 30: a connection always sees its own writes, another process's only after the interval.
- `FastMCP.lifespan()` combines *provider* lifespans only and never runs the `lifespan=` passed to the constructor. `_lifespan_manager()` is what the transports enter, so a test exercising a constructor lifespan must use it. The server's lifespan can be re-entered, so it resets the cached client in a `finally` — without that the next cycle hands out a closed one.
- A tool result's text blocks do not reach the model when `structuredContent` is set: Claude Code and the Agent SDK forward the JSON plus image and resource blocks only, on the assumption that the text duplicates the structured data, while Desktop and claude.ai forward both and pay twice. No setting controls it. So a tool whose text is a *rendering* rather than a copy must send one channel — which is why the MCP search tools carry text and images and no structured content, and why their tests read the rendered text.
- Claude Desktop stores the server-level `instructions` and never reads them, and no client but Claude Code honours a skill's `allowed-tools`. Every tool description therefore has to stand alone: guidance that only lives in the instructions or the skill does not reach those clients. `execute_code`'s docstring carries the whole sandbox contract for that reason. Claude Code also moves any MCP call to the background after ~120s (`MCP_TOOL_TIMEOUT`, undocumented).
- fastmcp 4 / MCP SDK 2: protocol types are snake_case (`read_only_hint`, `open_world_hint`, `mime_type`, `input_schema`, `server_info`) and the camelCase bridge warns and goes in fastmcp 5, hence the `<5.0.0` cap. `Client` defaults to the sessionless protocol where `initialize_result` is `None` — read `client.instructions` / `client.server_info`, which both modes populate. `ToolResult` imports from `fastmcp.tools` (`fastmcp.tools.tool` is the decorator, not a module, and ty rejects it). Tag visibility is `server.disable(tags=...)` after construction, not a constructor filter. Server-initiated sampling is gone, so the server can never borrow the client's model. Without an explicit `version=` the server reports *fastmcp's* version as its own.
- Monty caps host callbacks per checkout (`max_suspensions`, 1000 by default since 0.0.22) and it cannot be disabled; `_session_limits` sets `_MAX_HOST_CALLS = 10_000_000` so the time budgets govern instead. A capability run keeps one session across every execution, so a corpus-wide pass over a few hundred documents hits the default. With the cap out of reach, `analysis.code_timeout` is the only thing bounding a read-heavy program, which is why the deadline is checked before every host call and not just in `_run_on_loop`.
- Monty runs VFS callbacks on a thread coverage does not trace, so a reader's own lines need a direct-read test (`tests/sandbox/test_sandbox_toc.py`) beside the behavioural test that reads through a program.
- `toc.json`'s `item_range` is an index range into the position-ordered item list, i.e. the line numbering of `items.jsonl` — not database item positions. A gap in positions made the old numbers slice in the next heading. `get_document_section` slices by index the same way, so the two stay in step.
- `document_items` is fetched across documents only through the grouped accessors: `resolve_refs_grouped`, `get_items_in_ranges`, `get_pictures_grouped`, `get_caption_picture_refs_grouped`. The single-document ones they replaced are gone. Each builds a per-document predicate, `(document_id = 'a' AND col IN (…)) OR (…)`, never `col IN (union)` — see the collision note below. Search enrichment, context expansion and the reranker's blob fetch are all flat in document count, and tests assert the exact counts, so a regression to per-document fetching fails rather than merely slowing down.
- `get_pictures_grouped` takes `with_text=False` by default. The text is on the same rows, so it costs nothing in queries but does in bytes, and the multimodal reranker scores pixels and discards it. Only the enrichment path opts in.
- `expand_with_items` does not fetch: it takes the ref positions and window items to expand from, and `expand_context` does both fetches once for every document. Assemble results in `document_groups` order — the score sort that follows is stable, so arrival order is the tiebreak and separate passes for expandable and passthrough documents reorder equal scores.
- `document_items.self_ref` values collide across documents (`#/pictures/0` exists in every one), so any fetch batched across documents must key on (document_id, self_ref) and use a per-document-precise predicate. Keying on `self_ref IN (union)` both over-fetches blobs and can hand one document another's picture.
- `Store.vacuum()` uses 24h retention; to actually purge recent old fragments call `table.optimize(cleanup_older_than=timedelta(seconds=0))` directly per table.
- `HfApi.upload_large_folder` has no `path_in_repo` — `evaluations upload` stages the DB under a tempdir parent with `os.link` hardlinks before uploading; do not regress to `upload_folder`'s direct `path_in_repo` without keeping resumability.
- `Document.get_docling_document()` decompresses only the structure blob; calling `set_docling(...)` after a structure-only load writes `docling_pages=None` and destroys page rasters. Update `docling_document` directly via `compress_docling_split`.
- No document read loads the docling blobs by default: `get_by_id`/`get_by_uri` project `id`+`content` (merged with `document_meta`), `list_all` needs `include_content=True` for the text. Peak RSS tracks the blob bytes ~1:1, so a whole-row read is GBs on image-heavy corpora. Load blobs deliberately with `get_docling_data`/`get_pages_data`, or `get_by_id(..., include_blobs=True)` — required before any path that writes the row back through `DocumentRepository.update`, which would otherwise persist `None` over the stored blobs.
- Eval tests must run from inside `evaluations/`: `cd evaluations && uv run pytest tests/...`. Running from repo root hits an import-collision with `tests/` because both packages have files like `test_config.py`.
- `textual_image` has two paths: `widget.Image` for Textual TUI apps, `renderable.Image` for Rich consoles (e.g. `format_citations_rich`). Both accept PIL Image objects.
- `uv run ty check <files>` only checks the files you pass. Pre-commit's ty hook is broader and may surface errors in untargeted files. Run `uv run pre-commit run --files <paths>` before declaring lint clean.
- Bulk import rewrites: use Python one-liners (`import re; re.sub(...)`), not `sed -i`. macOS BSD sed lacks `\|` alternation in regex and silently no-ops.
- `Citation` and `resolve_citations` live in `haiku.rag.store.models.citation` (moved here from `agents.research.models`).
- Rich-renderer tests: render to string via `Console(record=True, width=200).print(r); console.export_text()` and assert substrings, instead of asserting on `panel.title.plain` etc. Decouples tests from Rich internals.
- Ingester state lives in TWO databases: LanceDB (documents) and the queue (queue + sync_state). The queue defaults to SQLite `ingester.db` in `storage.data_dir`, but `ingester.queue.dburi` can point it at Postgres instead. Deleting one without the other leaves stale sync_state — discover() reports UNCHANGED for every URI and no jobs queue. For the SQLite default, `rm ingester.db*` (catches `-shm`/`-wal`) before restart.
- Commenting out the children of a non-Optional Pydantic config block parses to `null` and Pydantic v2 rejects it (e.g. `providers:` followed by only `#` lines). Drop the key entirely or keep one child uncommented.
- Both `docker/Dockerfile` and `docker/Dockerfile.slim` install `--extra ingester` only — `[ingester]` does NOT pull `[docling]`. Published images cannot run `processing.converter: docling-local`. Build locally with `--extra docling` added to the `uv sync` lines and tag as the public name to override. `[ingester]` DOES pull `[s3]`, so obstore is always importable from ingester-only modules.
- The partial unique index `uq_jobs_live(source_id, uri)` allows one live job per (source, uri) regardless of op — prevents a DELETE running after a sibling UPSERT and wiping a fresh document. Side-effect: rapid (Change.deleted, Change.added) FS events from `git checkout` / atomic-rename saves squash to delete-only. `FSPoller._handle_watch_change` mitigates with `path.exists()` recheck on Change.deleted; preserve this when refactoring.
- Default FS `poll_interval_s` is 300s. To force a sweep without waiting: `curl -X POST http://localhost:8765/sources/{id}/refresh` (no auth when `api.auth_token` is unset).
- `obstore` exceptions subclass only `obstore.exceptions.BaseError(Exception)` — not `OSError`, not `httpx` — so they miss every branch of `workers/pipeline._classify` except the unknown-error default. `_classify` maps `PermissionDeniedError`, `UnauthenticatedError`, `UnknownConfigurationKeyError` and `InvalidPathError` to `PermanentError`; everything else under `BaseError` (`GenericError`, `JoinError`) stays transient. `NotFoundError` is deprecated upstream — obstore emits a builtin `FileNotFoundError` for a missing object, which was already permanent.
- An exception propagating out of an FS periodic sweep dies silently. `FSPoller` runs `_sweep_loop` as a `create_task` that `run()` never awaits, `PollerManager.live_pollers` counts only the outer `run()` task parked on `_stop.wait()`, and shutdown's `gather(..., return_exceptions=True)` retrieves the exception so asyncio never warns. `/health` stays green. Only `PeriodicPoller` and the FS *initial* sweep surface a propagated sweep error.
- Poller discovery failures are handled in three places with the same `record_failure` + `logger.exception` body: `pollers/base.py` `_sweep_once` and `_dry_run_once`, plus `pollers/fs.py` `_watch_loop`. Change one, change all three.
- Ingester queue is SQLAlchemy Core async (`ingester/queue/db.py` tables/`MetaData`, `repository.py` repos hold an `AsyncEngine`). `ingester.queue.dburi` selects the backend: SQLite default (`pool_size=1`) or Postgres (`postgresql+asyncpg://`, claims via `FOR UPDATE SKIP LOCKED`). `claim_next` MUST stay one `UPDATE … WHERE id=(SELECT … LIMIT 1) RETURNING` — a split select-then-update isn't atomic across SQLite connections (two processes double-claim).
- Postgres queue tests (`tests/ingester/test_queue_postgres.py`): asyncpg binds a connection to its creating loop and pytest-asyncio uses a per-test loop, so build the engine inside the test with `NullPool`. Isolate parallel xdist workers with a per-test Postgres schema (`search_path`), not shared tables — xdist `--dist load` ignores `xdist_group`.
- Build the SQLite queue URL with `URL.create("sqlite+aiosqlite", database=str(path))`, never an f-string `sqlite+aiosqlite:///{path}` — reparsing a path containing `?`/`#` (valid POSIX filename chars) truncates it into the URL query/fragment and opens the wrong file.
- `make_engine` builds the Postgres engine with `pool_pre_ping=True` so a long-running ingester survives a DB restart / dropped idle connection. SQLite (single pooled connection) is left unpinged.
- Coverage uses `concurrency = ["greenlet", "thread"]` (pyproject `[tool.coverage.run]`): the SQLite PRAGMA connect listener runs via SQLAlchemy's greenlet + aiosqlite worker thread, so without it `queue/db.py` reads as uncovered. CI runs `-m "not integration"`, so Postgres-only construction branches are covered by build-without-connect unit tests (`make_engine(dburi=...)`, `_insert(jobs, "postgresql")`), not the integration tests.
- The ingester dashboard (`ingester/api/static/index.html`) is a single self-contained vanilla-JS page (no build step). The biome pre-commit hook is scoped to `app/frontend/`, so this file is NOT auto-linted — run `biome lint <file>` manually. biome lints `.html` embedded JS but does NOT format it; sanity-check syntax with `node --check` on the extracted `<script>`.
- `create_document_from_source` has three exit paths with different metadata merges: full ingest does `{**user_metadata, **source_metadata}` (source-derived keys win), but the revision short-circuit calls `_refresh_doc_metadata(source_metadata=None)` where the merge is `{**doc.metadata, **user_metadata}` — so anything passed as `metadata=` (e.g. ingester metadata providers) can overwrite stored `md5`/`source_revision`/`content_type` unless those reserved keys are stripped first.
- Ingester plugins use `importlib.metadata` entry points. Metadata providers register under `haiku.rag.metadata_providers`; only referenced providers are loaded.
- `SearchResult.document_meta` and `Citation.document_meta` carry the parent document's metadata for UI consumers (e.g. a UI reading `source_url`); excluded from `format_for_agent`. The capability cite tools' direct-lookup fallback (chunk ids without a prior search) backfills it from the document lookup.
- `SearchResult.chunk_meta` / `Citation.chunk_meta` are the verbatim `Chunk.metadata`, deliberately NOT stripped of the keys `ChunkMetadata` types, so a third-party chunker's fields survive schema evolution. Consequences: on a context-expanded result it is the *anchor* chunk's, so its `headings`/`page_numbers`/`doc_item_refs` can disagree with the merged typed fields, which stay authoritative; and `format_for_agent` excludes it unless `include_chunk_meta` asks for the custom keys, which the MCP search tools do and the capabilities do not. Display code strips the typed keys (`detail_view._format_extra_metadata` filters `ChunkMetadata.model_fields`).
- `_cite` resolves a chunk id that misses exactly to the nearest id the run retrieved, above `CHUNK_ID_MATCH_CUTOFF` (0.75), before that database fallback: models transcribing 36-character UUIDs drop and duplicate characters and whole hyphen groups. Unrelated UUID4s score around 0.5, and candidates are limited to evidence the model saw. The repair runs before `resolve_citations` so citation order, and therefore `cited_map`, is preserved.
- Repeated cite reminders in `capabilities/instructions/*.md` are load-bearing. Stating the mandate once instead of three times cost 21pp of cite rate on gemma4-26b (z≈4.7). Treat prompt-length reduction there as a behavioural change needing its own eval, never a tidy-up.
- `get_model` sets `openai_chat_supports_multiple_system_messages: False` on OpenAI-compatible endpoints (ollama, openai with base_url) so pydantic-ai merges leading system messages. Instructions from multiple sources (agent preamble, capability instructions, request-limit notices) map to one system message each, and strict vLLM chat templates (e.g. Qwen's) reject more than one with a 400 "System message must be at the beginning."
- Ingester worker tests (`tests/ingester/test_pipeline.py`, `test_workers.py`) mock the client with `AsyncMock(spec=HaikuRAG)`; function-style `side_effect` doubles for `create_document_from_source` use explicit signatures (`def _route(uri, *, sources=None, source_id=None, ...)`). Changing the real call signature (adding a kwarg) makes those doubles raise `TypeError`, which `_classify` turns into a transient failure — update every such double when the signature changes.
- ty treats `dict` as invariant in its value type: a `dict[str, Subtype]` is not assignable to a param typed `dict[str, Base]` (or `dict[str, SomeProtocol]`). Type read-only params that accept such dicts as `Mapping[...]` (covariant), not `dict[...]`.
- The ingester queue uses versioned schema migrations (`ingester/queue/migrations.py`, `SCHEMA_VERSION`). `create_all` only creates missing tables/indexes — it NEVER adds a column to an existing table. Adding a column to a queue model needs BOTH the model change AND an explicit `ALTER TABLE` (+ backfill) in `apply_migrations` under a `current < N` block, then bump `SCHEMA_VERSION`. Existing DBs migrate in place on open.
- Ingester reaping is lease-based, not duration-based. A worker renews `last_heartbeat_at` on its in-flight jobs every `heartbeat_interval_s` via the pool's heartbeat task; `reap_stale` resets a claim only when `COALESCE(last_heartbeat_at, claimed_at)` is older than `lease_ttl_s` — so legitimate job duration is unbounded while crash detection stays short. The reaper only ever targets dead owners; it does NOT cancel a live coroutine. `WorkerConfig` is `extra="forbid"` and enforces `heartbeat_interval_s <= lease_ttl_s / 3`; the former `claim_timeout_s` key is gone and now fails config validation.
- Ingester worker ids are `{pid}-{uuid}-{n}` (globally unique so `claimed_by` guards hold across processes sharing a Postgres queue). `WorkerPool` boot-reap is scoped to `lease_ttl_s`, not 0, so a peer process's live claims survive our startup. `_untrack_inflight` drops only the pool's OWN entry for a job: if the reaper reset a claim and a sibling re-claimed it, evicting the entry would stop renewing the sibling's lease — preserve this on refactor.
- Adding a field/sub-model to `AppConfig` (config/models.py) is zero-plumbing: the loader validates it via pydantic, and `init-config`/`settings` both serialize `AppConfig().model_dump()`, so it auto-appears in generated YAML and `settings` with no loader or CLI changes.
- Throwaway lancedb analysis scripts: run `uv run python` from the repo root (lancedb isn't importable after `cd`-ing out of it). The sync `LanceTable` has no `.query()`, and `.to_lance()` needs `pylance` (not installed) — use `tbl.to_arrow()` then select columns from the Arrow table.
- `doctor` renders each `CheckResult.details` line through Rich as `[dim]{detail}[/dim]` (app.py), so avoid literal `[...]` in detail strings — Rich parses them as markup tags. Use other delimiters (e.g. `#1`).
- `doctor`'s near-duplicate detection (`doctor.py`) reduces each document to one summed (unnormalized) chunk-vector centroid during the single vector scan — no second copy of the vector matrix — then normalizes and clusters by union-find over pairwise cosine ≥ `similarity_threshold`. Documents with fewer than `min_chunks` embedded chunks are skipped. It stores only each document's best cosine to a twin, not all O(D²) pairs.
- Coverage: run `uv run pytest -m "not integration" --cov=haiku` (matches CI `test.yml`). Deep-dotted `--cov=haiku.rag.<module>` crashes beartype (claw circular-import while loading conftest); file-path `--cov=<path>.py` collects no data (source is the `haiku` package). Scope the report by grepping the term-missing output, not by narrowing `--cov`.
- Test context expansion / `visualize_chunk` without the embedder (no VCR needed): `import_document(docling_doc, [Chunk(..., embedding=[0.1] * vector_dim)])` — a precomputed `embedding` skips `embed_chunks`. Build the `DoclingDocument` with `add_page`/`add_text` + `ProvenanceItem` bboxes for page-image/bounding-box tests.
- Textual `auto` grid rows ignore margins: `margin-bottom` on a widget in an `auto` grid row collapses the row and clips the widget's children out of view (bit the chat `FlexibleInput`). Use bottom padding for spacing instead. Assert layout in tests with `parent.region.contains_region(child.region)`, not just region heights.
- User-attached images (`ask`/`analyze` `images=`, chat Ctrl+I) need the capability instructions' "Questions with attached images" section — without it, vision models emit the not-enough-information refusal without searching, because the instructions frame everything as knowledge-base content.
- A line that executes once per process (module-level lazy init, e.g. `get_config()`'s `_config is None` branch) is covered only if some xdist worker reaches it before an alternative — shard-dependent, so it varies with core count and can read 100% locally and 99.99% on a 2-core runner. Assert such branches directly with the module global reset via `monkeypatch`.
- `client.update_document` re-chunks and re-embeds, so it needs `@pytest.mark.vcr()` and a cassette. To rewrite stored content without touching the embedder, use `DocumentRepository.update` (writes the row only).
- Embed-only rebuild recreates the chunks table, so patching `store.chunks_table.add` on the instance is silently discarded — patch `lancedb.AsyncTable.add` at class level and filter on `self.name` (same shape as the `AsyncTags` gotcha). Its phase 2 writes via `chunks_table.add`, NOT `_flush_rebuild_batch`; and a FULL rebuild refreshes source-backed documents in place, keeping the document id (it re-fetches with `force=True`, bypassing the revision and MD5 short-circuits, and falls back to stored content if the fetch fails).
- pydantic-ai's request guard sits ABOVE VCR's HTTP interception, so from 2.19.0 (#6774, which extended `ALLOW_MODEL_REQUESTS` to embeddings) a `@pytest.mark.vcr()` test replaying a recorded embedder call is rejected before VCR ever sees the request. `tests/conftest.py`'s autouse `allow_replayed_model_requests` enables model requests for any vcr-marked test; the explicit `allow_model_requests` fixture stays for live (`--disable-recording`) use. An unrecorded call still fails, on `record_mode=none`, so the protection is unchanged. Symptom if this is ever removed: dozens of `RuntimeError: Model requests are not allowed` across tests/sandbox, tests/multi_db, tests/store and tests/ingester, plus a 0%-CPU wedge at the end of the run as a failing test leaves a session un-cleaned.
- `VLLMProvider.model_profile()` infers the model family by NAME PREFIX, so an alias decides what a served model is taken to be: `nvidia/Gemma-4-26B-A4B-NVFP4` matches `gemma-4` and gets `supports_thinking=True`, while `gemma4-26b` (the same model on the same vLLM port) matches nothing and gets `False`. Qwen3.8 and Qwen3-Coder are deliberate upstream exclusions. the `vllm` branch of `get_model` sets no `openai_reasoning_effort`, unlike the `ollama` and `openai` branches, because the effort vocabulary is per-model: `Inferact/Qwen3.8-27B-NVFP4` **rejects** `high` with `400 Unexpected reasoning effort high`, taking `xhigh`/`medium`/`low`/`none`. pydantic-ai derives the level from the profile instead (`_resolve_openai_thinking_effort`): `true` → `medium`, `false` → `none`, and `thinking` is stripped entirely where `supports_thinking` is absent. Consequences, all measured against vLLM: `enable_thinking` WORKS on `nvidia/Gemma-4-26B-A4B-NVFP4` and is INERT on `gemma4-26b` — the same weights on the same port, so the alias decides the knob. It is inert on Qwen3.8 and Muse Glimmer too, and on Qwen3.8 that is a change: sending `reasoning_effort: none` did disable its thinking. `extra_body: {reasoning_effort: …}` is the precise knob: it lands as a top-level request field, OVERRIDES the profile-derived level (gemma-4 `true` + `extra_body` `none` → 0 thinking chars), reaches each model's own vocabulary (Qwen3.8 `xhigh` → 154 chars, `none` → 0 from a 195 baseline), and 400s on a value the server rejects rather than going quiet. A template with its own switch takes `chat_template_kwargs` instead — Muse Glimmer's `reasoning_strength` (low 158 vs high 228 chars), and any server started with `--reasoning-parser`, which consumes the template switch first. `provider: vllm` means two different implementations: pydantic-ai's chat provider on a `ModelConfig`, and haiku.rag's own multimodal client under `embeddings.model` / `reranking.model`.
- logfire 5 moved `LogfireQueryClient` out of `logfire.experimental.query_client` to `logfire.query_client` (same constructor and `query_json_rows` signature). It arrives transitively with pydantic-ai 2.40 via `pydantic-ai-slim[logfire]`, together with `anthropic` 1.x (httpx2) and `openai` 3.x. Only `logfire.span`/`logfire.warn`/`configure` are used in the product, so the break was confined to `evaluations/scripts/build_t2_submission.py`.
- `astral-sh/setup-uv` publishes floating major tags only through v7; v8/v9 exist as exact tags only, so `@v9` fails to resolve and the job dies in seconds. Pin exact (`@v9.0.0`). v4 also predates GitHub's current cache API and gets `400 Failed to restore`, silently redownloading every wheel.
- A test driving an in-process `fastmcp.Client` needs `@pytest.mark.filterwarnings("ignore:Found propagated trace context:RuntimeWarning")`. Any earlier test in the same xdist worker that runs the CLI calls the real `telemetry.configure()`, which installs logfire's `WarnOnExtractTraceContextPropagator` process-wide; the client then propagates trace context into the server and the propagator warns `RuntimeWarning` on its first extraction per process. `filterwarnings = ["error", ...]` turns that into an exception inside the tool, surfacing as `fastmcp.exceptions.ToolError: Found propagated trace context`. Shard-dependent, so it passes in isolation and on high-core machines: reproduce with `uv run pytest tests/test_cli.py <the test> -n0`.

- Capability state (`RAGState`, `AnalysisState`) shares the flat `EvidenceState` base and is dumped and re-validated at every carry point: `client/agents.py`, `chat/app.py`, `evaluations/capability_runner.py`, and over the wire in `app/backend/main.py`, where the AG-UI client hands the snapshot back on the next turn. Nesting a field under a sub-object or renaming one is a breaking wire change; additions are safe. `begin_invocation()` drops the previous question's working evidence (`citations`, `searches`) and is called only when a new question starts.
- `providers.docling_serve.timeout` bounds each HTTP call (submit, poll, result), not the whole conversion — the status poll is a `while True` loop with no deadline, so a job that never finishes hangs whatever the value.
- Mutating a pydantic model's `model_config` after class creation has no effect until `Model.model_rebuild(force=True)`; a strictness probe that skips the rebuild silently validates under the old config.
- `haiku.rag.sources` is importable from core without the `[s3]`/`[ingester]` extras because `obstore` is imported inside `S3Source` methods, not at module level. Hoisting that import breaks a slim install.
- One-shot directory ingest enumerates through `sources.fs.walk_files`, which never follows directory symlinks and keeps a symlinked file only when its target resolves inside the named root. There is no size cap on that path (the ingester's `max_file_size` applies to `FSSource` only).
- `run_db_checks` (doctor.py) is an orchestration list over `_check_*` functions that take the already-computed locals. The vector matrix stays a local so the `del vectors` memory ceiling holds — putting it on a shared snapshot object regresses peak RSS.
- Agent specs construct capabilities through `from_spec`, never `cls()`: `id` must come from `create_capability()`, because `_base.py` filters tools by `tool.capability_id != self.id` and pydantic-ai's duplicate-id rejection is what enforces one compaction/policy capability per run (a bare `cls()` leaves `id=None` and two are accepted silently). pydantic-ai does NOT coerce or validate spec arguments — `load_from_registry` passes raw parsed YAML, so a `db_path` arrives as `str` and a `config` as `dict`. Third-party capabilities are never auto-discovered: the caller passes `custom_capability_types=[...]` (no entry points, no registry hook), so never claim otherwise in docs. A spec always needs a model, in the spec or as a kwarg. Overriding `from_spec` also moves schema generation from `cls.__init__` to the `from_spec` signature; a zero-argument override drops the `spec_params_*` def entirely, which is what keeps compaction's per-run caches out of the schema.
- `is_local_uri` and `uri_to_path` (`uri.py`) own the two decisions "is this local" and "what path is this" — four call sites each had their own copy. `urlparse("C:/docs/a.pdf")` reports the drive letter as scheme `c`, and `urlparse().path` keeps the leading slash before a Windows drive, so `file:///C:/docs/a.pdf` becomes `\C:\docs\a.pdf`; `url2pathname` is the stdlib fix, per platform. A file URI's host is reattached after conversion, not passed to `url2pathname`, which as of 3.14 rejects a non-local authority off Windows. Don't re-derive any of this locally.
- `tests/test_uri.py` runs in its own CI matrix (ubuntu/macos/windows x py3.13/3.14) **without the project installed**: `uv run --no-project ... --noconftest -o addopts=`. Keep it stdlib-only — a project import, or anything needing the repo conftest, breaks the Windows legs that cover drive-letter conversion.
- Repeating a search query within a question accumulates: `state.searches[query]` is merged through `merge_results` (`_tools.py`), keyed on `chunk_id`, so a narrower re-search cannot drop what a wider one already showed the model. `SearchResult`s built by hand in tests without a `chunk_id` are indistinguishable and collapse to the first. `search_corpus` returns `"No results found."` rather than an empty string.
- Searches in one model response deduplicate their returns: a result whose rendered evidence a sibling of the same `run_step` already showed keeps its rank slot but collapses to `Also matched, shown above: [id]`, and a picture attaches once per response on `(source, document_id, self_ref)`. Equivalence is `evidence_signature` (`_tools.py`): the `format_for_agent` rendering at neutral rank/total plus picture keys, bucketed under `qualified_id` — another database's copy or a different expansion of the same anchor formats in full. `state.searches` keeps every result whole, so cites resolve from elided context. All search state (`merge_results`, `_note_evidence`, shown keys) commits only after formatting AND image construction succeed; a failed sibling contributes nothing but still consumes its budget. Dedup never crosses run_steps (compaction rewrites prior-turn returns, so an "above" reference could dangle). In-code sandbox `search()` bypasses pricing, dedup, and spans entirely.
- `app.py` and `cli.py` are covered honestly through Typer's `CliRunner` (`tests/test_app.py`, `tests/test_cli.py`); `cli.py` keeps exactly one pragma, on the `__main__` guard. Don't reintroduce class- or function-level pragmas there.

- The **configuration places databases; arguments select.** `lancedb.uri` and
  `HAIKU_RAG_DB` do not exist: a config carrying `uri` fails validation with the
  `databases` spelling to use. Nothing derives a per-database config, so the name
  always travels with the scope (`create_mcp_server` and the capabilities take or
  resolve one; `ChatApp` sets `capability.scope`).
- Never manufacture a default path to pass along: `HaikuRAG(None, config)` resolves
  the configuration, and a manufactured `data_dir / "haiku.rag.lancedb"` beside a
  configured placement raises `AmbiguousDatabaseError`. Only the CLI's `--db`
  overrides configuration, through `DatabaseScope.at`.
- Downstream hosts (the ingester included) build clients from config alone.
- `_fuse` maps a reranked chunk back to its database by `id(chunk)`, since chunk
  ids repeat between copies. Every shipped reranker indexes back into the list it
  was given; one that returns copies raises a named `ValueError`.
- `clients_covering(sources)` **opens** the databases; `_require_known_sources`
  only checks names. Validating at an operation boundary must use the latter, or
  an unscoped question opens every configured database before the model runs.
- `Store.stored_settings` and `Store.stored_embedding` are read at open and never
  follow a later write. Refresh them together through `_remember_settings`.
- Textual: `pilot.pause()` does not reliably flush a handler that awaits I/O on a
  slow runner. Await the handler directly (`await modal.on_checkbox_changed(...)`)
  as the other tests in that file do.
- `UnknownDatabaseError` subclasses `KeyError` and overrides `__str__`, so
  `pytest.raises(KeyError)` still catches it — which means a broad `KeyError`
  assertion cannot tell the contract from a bare one. Assert the specific type.

## Planning and Commits

When planning work, break the plan into **self-consistent, commitable chunks**. Each chunk should:
- Be a logical unit that can stand on its own
- Leave the codebase in a working state
- Be small enough for meaningful review

## Common Patterns

**Test with VCR recording:**
```python
@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_my_feature(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document("Test content", uri="test://doc")
        results = await client.search("test")
        assert len(results) > 0
```

**Creating a test database fixture:**
```python
@pytest.fixture
async def client(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        yield rag
```

**Using the native RAG capability:**
```python
from pydantic_ai import Agent
from haiku.rag.capabilities.rag import create_capability

agent = Agent(
    "openai:gpt-4o",
    capabilities=[create_capability(db_path=db_path, config=config)],
)

result = await agent.run("What documents do we have?")
```

**Custom embedder:**
```python
from pydantic_ai.embeddings import Embedder
from haiku.rag.embeddings import EmbedderWrapper

# Wrap any pydantic-ai Embedder
embedder = EmbedderWrapper(Embedder("openai:text-embedding-3-small"), vector_dim=1536)
```

## Evaluations

Separate workspace package (`evaluations/`) for benchmarking.

```bash
cd evaluations
uv sync
evaluations run <dataset>                    # default --target rag-capability
evaluations run <dataset> --target analysis-capability --capability-model ollama:qwen3.8
evaluations run <dataset> --judge-model ollama:gemma4  # independent judge
evaluations download <dataset|all>           # Pre-built eval DBs from HuggingFace
evaluations upload <dataset|all>             # Upload eval DBs
```

Datasets: `hotpotqa`, `orb_text`, `orb_multimodal`, `orb_multimodal_nemotron`, `t2_finqa`, `t2_tatdqa`, `mtrag_clapnq`, `mtrag_clapnq_rewrite`, `mtrag_clapnq_live`, `mtrag_clapnq_live_uncompacted` (the four mtrag keys share one DB).

**Multi-turn (MTRAG)**: `mtrag_clapnq` runs gold-prefix QA (`ConversationInput` cases replay the reference prefix as message history) plus lastturn retrieval; `_rewrite` retrieves with the human rewrites; `_live` and `_live_uncompacted` set `spec.live` (one case per conversation, `--limit` counts conversations) and differ only in `spec.compaction`, which registers `EvidenceCompactionCapability` in the runner — the only eval coverage compaction has, since every other dataset is single-turn where it is inert. Live runs carry `all_messages()` and ONE capability-state dict across turns (0.74.0 compaction raises on history with a fresh state dict) and record question-length per-turn arrays (`turn_cited_uris`, `turn_n_search_calls`, `turn_n_rejected_searches`, `turn_n_failed_tools`, `turn_n_requests`, `turn_citation_status`), counted per turn from `new_messages()` so compaction rewriting earlier history cannot skew them. Gold-prefix and live pass rates answer different judge questions and are NOT comparable; the supported comparison is compacted vs uncompacted, paired by turn. `_live_summary`'s macro rate excludes conversations with zero judged turns — a judge outage is an operational exclusion, not a failed conversation.

**Targets:** `rag-capability` (default), `analysis-capability`. Both run end-to-end through native Pydantic AI agents (see `evaluations/capability_runner.py`). `--capability-model` defaults to `config.qa.model` or the configured analysis model.

**Citation retrieval metric**: `CitationMAPEvaluator` scores the URIs the capability registers via its citation tool against gold `expected_uris`, alongside the LLMJudge. Score key: `cited_map`.

**The judge is pinned and frozen.** Every reference config under `evaluations/configs/` whose dataset is judged carries the same block (`temperature: 0.6`, `max_tokens: 16384`, `extra_body` with `top_p` 0.95 / `top_k` 20 / `min_p` 0 / `chat_template_kwargs.enable_thinking: true`), guarded by `tests/test_reference_configs.py`. Do not change it without an eval: greedy decoding lost 5-14% of verdicts to repetition spirals on Qwen, and `max_tokens: 32768` measured worse than 16384 (130 lost verdicts vs 36) because the spiral is not budget-bound. `DEFAULT_JUDGE_MODEL` carries only the subset ollama honours (`temperature`, `max_tokens`, `top_p`) — ollama silently ignores `top_k`, `min_p` and `chat_template_kwargs`. `t2_finqa` / `t2_tatdqa` set `qa_evaluator`, which replaces the evaluator list, so no judge is ever constructed for them and a judge block there is dead config.

**Per-case diagnostics**: runs record `cited_chunk_ids`, `searched_uris`, `n_searches`, `n_search_calls`, `n_rejected_searches`, `n_failed_tools`, `n_executions` and `n_requests` as eval attributes. Counted from the message history, since `state.searches` is keyed by query and `for_run` gives the run a `replace()` copy whose counters the host never sees. Reading them:
- `n_rejected_searches` is search-budget exhaustion; `n_failed_tools` is any failed call of that capability's tools, and a failed `analysis_execute_code` is either an exhausted execution budget or an error in model-written Python
- `n_searches` counts distinct search *keys*, and analysis files every in-code `search()` under one `_sandbox` key, so sandbox searches are not counted anywhere
- `n_requests` is the run's request count, matching a capability's own budget only while it stays loaded
- `citation_status` derives `grounded` / `ungrounded` / `missing` from the evidence record via `ledger.citation_status` (an explicit `cite([])` is `ungrounded`, distinct from declaring nothing)

`evaluations run --filter/-f CLAUSE` restricts every benchmark search (retrieval, QA, and live conversations) and is recorded as `document_filter` in experiment metadata — a filtered run must never be compared against an unfiltered one.

**Never steer on raw cite rate** — it is confounded by task success. Measure cite rate among *correct* answers, split by whether the case used `execute_code`.

**Eval prompts**: datasets do not carry custom system prompts. Capability targets use packaged instructions plus `config.prompts.domain_preamble`. `DatasetSpec` has no `system_prompt` field.

**Eval-side rules**: don't assert specific phrases in packaged instructions; test behavior instead. `build_experiment_metadata` is additive. Run targeted tests in `evaluations/` with `uv run pytest`.

**Reasoning knobs on vLLM**: a vLLM server started with `--reasoning-parser` consumes `chat_template_kwargs.enable_thinking` itself — it never reaches the chat template. Muse-Glimmer QA blocks must set `chat_template_kwargs.reasoning_strength: high` instead (the mtrag reference config does); a template defaulting it to low silently cuts search calls ~37% with no error anywhere. Verify a kwarg by RENDERING (`/tokenize` with `return_token_strs`) or by measured behavior, never by HTTP acceptance.

## haiku.rag.app (Conversational RAG Application)

The `app/` directory contains a conversational RAG application with pydantic-ai's native AG-UI support and CopilotKit frontend.

### Quick Start

```bash
cd app
docker compose -f docker-compose.dev.yml up -d --build
# Frontend: http://localhost:3000
# Backend: http://localhost:8001
```

### Structure

```
app/
├── backend/
│   └── main.py             # Starlette app with pydantic-ai AGUIAdapter
└── frontend/
    ├── app/                # Next.js app
    │   └── api/            # API routes (copilotkit, info, visualize, documents)
    ├── components/         # React components
    │   ├── Chat.tsx        # Main chat interface
    │   ├── CitationBlock.tsx   # Citation rendering with visual grounding
    │   ├── SessionManager.tsx  # Session management
    │   ├── DbInfo.tsx          # Database info panel
    │   └── DocumentFilter.tsx  # Document filtering for sessions
    └── biome.json          # Biome config for linting/formatting
```

The backend adapts a native `RAGCapability` agent with Pydantic AI's `AGUIAdapter`. It emits one final state snapshot; native tool events need no bridge.

### Context Expansion (`context.py`)

Section-bounded expansion: sections within `max_context_chars` are returned whole. Too-large sections expand outward bounded by section edges. Too-small sections (< 20% of budget) grow across boundaries. Only truly overlapping ranges merge — adjacent sections stay independent. The expanded result is hard-capped at `max_context_chars`. `visualize_chunk` expands context before resolving bounding boxes to cover all pages.

### Analysis Sandbox (`sandbox/sandbox.py`)

pydantic-monty interpreter in a subprocess worker checked out of an `AsyncMonty` pool, with a virtual filesystem at `/documents/{id}/`. Every file is a read-only `CallbackFile`: `metadata.json` returns an already-built string, `content.txt` is lazy per-document, and `items.jsonl`/`chunks.jsonl`/`toc.json` use a lazy bulk cache where the first read triggers one query for all items. One session serves every `execute()` call, so variables persist; `_run_on_loop` bridges sync reads to async DB queries.

`analysis.code_timeout` is a deadline, not a per-read timeout: `_check_deadline` runs before every host call (`_timed` wraps all five readers, the in-code `search()` and `list_documents()` call it first, `_run_on_loop` keeps its own check), so past the deadline no further host call starts and one already running finishes, with compute stopped by Monty's watchdog. `_run_on_loop` alone is not enough to bound it — `metadata.json` is served from memory and the JSONL files from the per-document cache after their first read, so neither touches the bridge.

`chunks.jsonl` carries `{chunk_id, metadata}` per chunk, joining to `items.jsonl` through `chunk_ids`; in-code `search()` rows carry `chunk_meta` and `picture_refs`, and `list_documents()` rows carry `metadata`. A host error keeps its message for every caller — the capability's agent reads it to repair its own code, and the MCP boundary is trusted, so there is one path and no masking.

### Capability Tools

**RAGCapability** exposes `rag_search` and `rag_cite`.

**AnalysisCapability** exposes `analysis_search`, `analysis_execute_code`, and `analysis_cite`. The sandbox mounts a document VFS at `/documents/{id}/`.

**EvidenceCompactionCapability** exposes no tools. It discovers the evidence capabilities through `RunContext.capabilities` and rewrites `request_context.messages` in `wrap_model_request` — never the stored history, which is what keeps question identities and epochs (both message counts) meaningful.

**CitationPolicyCapability** exposes no tools. It reads the same discovered records and acts in `after_model_request` / `after_run`. The cite tools accept an empty `chunk_ids`, recorded as a declaration with no refs that derives `ungrounded` — distinct from an answer that declared nothing (`missing`).

| Tool | Purpose |
|------|---------|
| `rag_search` / `analysis_search` | Hybrid search with expanded context |
| `analysis_execute_code` | Run Python in the analysis sandbox |
| `rag_cite` / `analysis_cite` | Register citations in host state |

### Development

- Backend hot reloads automatically
- Frontend: `docker compose -f docker-compose.dev.yml up -d --build frontend`
- Run `biome check` in `app/frontend/` after making frontend changes (use `biome check --write` to auto-fix)
- Logfire integration for debugging LLM calls
