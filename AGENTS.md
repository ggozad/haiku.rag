# haiku.rag Project Guide

Agentic RAG system built on LanceDB with hybrid search, multiple embedding providers, reranking, and a native Pydantic AI RAG capability with sandboxed code execution.

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
- `haiku.rag` - `haiku.rag-slim` with the docling, voyageai, cohere, zeroentropy, tui, cross-encoder and jina extras
- `haiku.rag-slim` - Core with optional extras

**Layout:**
```
haiku_rag_slim/haiku/rag/   # Source code
├── store/                  # LanceDB persistence
│   ├── engine.py           # Store class: connection, locks, migrations, vacuum, tags
│   ├── schema.py           # Table records (DocumentRecord, ChunkRecord, SettingsRecord), Arrow schemas, index_specs/ensure_indexes
│   ├── compression.py      # docling blob compression
│   ├── info.py             # gather_database_info, get_database_stats, DatabaseInfo
│   ├── models/             # Domain models: chunk (Chunk, SearchResult), document, document_item, citation
│   ├── repositories/       # CRUD: DocumentRepository, ChunkRepository, DocumentItemRepository, SettingsRepository
│   ├── upgrades/           # Version migrations (v0_20_0 … v0_75_0)
│   └── exceptions.py       # ReadOnlyError, MigrationRequiredError, AmbiguousDatabaseError, UnknownDatabaseError,
│                           # AmbiguousCitationError, SourceUnavailableError, ConfigMismatchError
├── embeddings/             # VoyageAI, Cohere, vLLM, OpenRouter (ollama/openai via pydantic-ai)
├── reranking/              # cross-encoder, Cohere, Zero Entropy, Jina, Jina-local, vLLM, OpenRouter
├── sandbox/                # pydantic-monty sandbox used by RAGCapability and MCP execute_code
│   ├── sandbox.py          # Sandbox, SandboxResult, recovery_hint
│   └── dependencies.py     # AnalysisContext (per-invocation filter)
├── providers/              # docling_serve.py (docling-serve client), picture_description.py
├── capabilities/           # Native Pydantic AI capabilities
│   ├── _tools.py           # Shared capability primitives
│   ├── ledger.py           # CapabilityEvidenceRecord, citation_status
│   ├── evidence.py         # discover_evidence(), DiscoveredEvidence
│   ├── compaction.py       # EvidenceCompactionCapability, build_capsule, compact_history
│   ├── policy.py           # CitationPolicyCapability, CitationPolicyState
│   ├── rag.py              # RAGCapability, RAGState, create_capability
│   └── instructions/       # Model instructions (rag.md, rag_multiple_collections.md)
├── tools/                  # Reusable pydantic-ai FunctionToolsets
│   ├── context.py          # RAGDeps protocol
│   ├── filters.py          # SQL filter builders
│   ├── search.py           # create_search_toolset()
│   └── document.py         # create_document_toolset()
├── chunkers/               # docling-local, docling-serve
├── converters/             # docling-local, docling-serve, pdf_split.py, text_utils.py, exceptions.py
├── config/                 # models.py (all config classes), loader.py
├── chat/                   # Chat TUI (app.py, widgets/)
├── inspector/              # Inspector TUI (app.py, widgets/)
├── client/                 # HaikuRAG high-level API
│   ├── __init__.py         # Lazy facade: HaikuRAG, RebuildMode, DatabaseScope, DocumentImport, all_found
│   ├── client.py           # HaikuRAG class, RebuildMode
│   ├── scope.py            # DatabaseScope, DatabaseRef (which databases an operation covers)
│   ├── session.py          # SingleDatabaseSession, FederatedSession
│   ├── documents.py        # create/import/update_document(s), create_document_from_source, DocumentImport
│   ├── processing.py       # convert, chunk
│   ├── titles.py           # generate_title
│   ├── search.py           # search, expand_context, visualize_chunk
│   ├── agents.py           # ask
│   ├── rebuild.py          # rebuild_database
│   ├── downloads.py        # download_models
│   └── exceptions.py
├── ingester/               # haiku-ingester service (own CLI + [ingester] extra)
│   ├── cli.py, app.py      # haiku-ingester entry point + application layer
│   ├── batch.py            # run-batch, dry-run manifests
│   ├── reconcile.py        # Startup reconciliation of LanceDB and sync_state
│   ├── metadata.py         # Metadata provider loading
│   ├── queue/              # db.py, migrations.py, repository.py (JobRepo/SyncStateRepo), models.py
│   ├── workers/            # pool.py (WorkerPool), pipeline.py (run_job), retry.py
│   ├── pollers/            # base, periodic, fs, manager, factory
│   ├── api/                # FastAPI control plane (server.py, auth.py, schemas.py)
│   │   ├── routes/         # health, sources, jobs, dlq, stats, config, database, dashboard, providers
│   │   └── static/index.html # Self-contained vanilla-JS dashboard served at /
│   └── exceptions.py       # PermanentError, TransientError
├── sources/                # base, fs, http, s3, webdav adapters + registry, plugins, filter, walk_files
│                           # (used by the ingester AND one-shot client ingestion)
├── app.py                  # HaikuRAGApp CLI application layer
├── cli.py                  # Typer CLI entry point
├── context.py              # Section-bounded context expansion, build_toc
├── mcp.py                  # FastMCP server (create_mcp_server)
├── doctor.py               # Database health checks, provider probes, near-duplicate detection
├── circuit_breaker.py      # CircuitBreaker (ingester pollers and workers, docling-serve provider)
├── s3.py                   # Object-store helpers
├── telemetry.py            # Logfire configuration
├── logging.py              # Logging configuration
├── uri.py                  # is_local_uri, uri_to_path
└── utils.py                # get_model, format_citations_rich, raise_missing_extra, and more
tests/                      # Mirrors source structure
├── conftest.py             # Fixtures (see Testing), capture_logs, for_path, writing
├── cassettes/              # VCR cassettes for the central suites (see Testing)
├── json_body_serializer.py # Custom VCR serializer for JSON bodies
├── docker/                 # docker-compose.yml for integration services
├── client/ chunkers/ converters/ embeddings/ interfaces/ providers/ reranking/ store/
├── capabilities/ chat/ ingester/ multi_db/ sandbox/ sources/ tools/
└── data/                   # Test data files
evaluations/                # Benchmarking workspace
├── configs/                # Reference configs per dataset
├── scripts/
├── evaluations/
│   ├── benchmark.py        # Typer CLI: run, pair, download, upload
│   ├── population.py       # populate_db
│   ├── retrieval.py        # run_retrieval_benchmark
│   ├── qa.py               # run_qa_benchmark, run_live_qa_benchmark
│   ├── results.py          # Per-case result files
│   ├── pairing.py          # evaluations pair
│   ├── artifacts.py        # download_dataset_db, upload_dataset_db, HF_REPO_ID
│   ├── experiment.py       # build_experiment_metadata
│   ├── config.py           # DatasetSpec, DocumentPayload, RetrievalSample
│   ├── capability_runner.py # native capability evaluation runner
│   ├── datasets/           # frames, hotpotqa, mtrag, open_rag_bench, t2_ragbench
│   └── evaluators/         # judge, map, citation, retrieval, number_match, refusal, conversation, transcript
└── tests/                  # benchmark and capability-runner tests
docs/                       # Documentation (zensical, not mkdocs; nav lives in zensical.toml)
docker/                     # Dockerfile (full), Dockerfile.slim (published)
examples/                   # Working examples (docker/, custom agents)
plugins/haiku-rag/          # Claude Code / Codex plugin (see MCP Server)
scripts/                    # bump_version.py, build-docker-images.sh, run-integration-tests.sh
app/                        # Conversational RAG application (see below)
```

## Architecture

**Key Classes:**

| Class | Location | Purpose |
|-------|----------|---------|
| `HaikuRAG` | client/client.py | High-level API (main entry point) |
| `HaikuRAGApp` | app.py | CLI application layer |
| `Store` | store/engine.py | Async LanceDB connection, tables, upgrades |
| `DatabaseScope` | client/scope.py | The databases an operation covers, resolved once |
| `SingleDatabaseSession` | client/session.py | One database: store, repositories, lifecycle. Reads and writes |
| `FederatedSession` | client/session.py | Several, composed lazily. Reads only |
| `DocumentRepository` | store/repositories/document.py | Document CRUD |
| `ChunkRepository` | store/repositories/chunk.py | Chunk CRUD + search |
| `DocumentItemRepository` | store/repositories/document_item.py | Document items, pictures |
| `SettingsRepository` | store/repositories/settings.py | Config persistence |

**Factory Functions:**
- `get_embedder(config)` → `EmbedderWrapper` (embeddings/__init__.py)
- `get_reranker(config)` → `RerankerBase | None` (reranking/__init__.py)
- `get_converter(config)` → `DocumentConverter` (converters/__init__.py)
- `get_chunker(config)` → `DocumentChunker` (chunkers/__init__.py)
- `create_capability(db_path?, config?, *, defer_loading=False, rag=None, request_limit=30, sources=None, vision=None)` → `RAGCapability` (capabilities/rag.py). Tools `search`, `execute_code`, `cite`, state under `rag`.
  - `request_limit` counts model requests per agent run. At the limit the capability's tools are removed except `cite`, which survives `CITATION_GRACE_REQUESTS` (2) further requests that call one of the capability's tools. Requests spent on other tools do not count.
  - A spent search or code budget does not withdraw the tool: it keeps failing. Withdrawing a tool the model still calls costs the agent's unknown-tool retries and aborts the run.
  - `vision` gates image attachment on search results and must reflect the model the hosting agent runs. Default `qa.model.vision`.
  - `rag` lends an open client: it lands in `borrowed_rag`, which `_ensure_rag` prefers and `_close` never closes.
  - `from_spec` delegates here so agent specs can declare the capability. `db_path` accepts a `str`.
- `create_capability()` → `EvidenceCompactionCapability` (capabilities/compaction.py). Optional, registering it is the only switch. Replaces earlier questions' evidence on the *request* with a capsule of what was cited (pictures included, fetched through the owning capability). Other earlier evidence returns as a receipt. No config, no budget: it reduces a request without bounding it.
- `create_capability()` → `CitationPolicyCapability` (capabilities/policy.py). Optional. Requires every answer to declare its grounding, asks once per question, records failures in `CitationPolicyState.violations`, never asks the model to change its answer. Enforced when the question retrieved evidence or the conversation already cited something.
- `create_search_toolset(config, ...)` → `FunctionToolset[RAGDeps]` (tools/search.py)
- `create_document_toolset(config, ...)` → `FunctionToolset[RAGDeps]` (tools/document.py)

**Base Classes:**
- `EmbedderWrapper` (embeddings/__init__.py) — wraps pydantic-ai `Embedder`
- `RerankerBase` (reranking/base.py)
- `DocumentConverter` (converters/base.py)
- `DocumentChunker` (chunkers/base.py)

**Domain vs Persistence:**
- Domain: `Document`, `Chunk`, `SearchResult`, `Citation` (Pydantic BaseModel in store/models/)
- Persistence: `DocumentRecord`, `ChunkRecord`, `SettingsRecord` (LanceModel in store/schema.py)

## HaikuRAG Client API

Key methods on `HaikuRAG` (client/client.py):

```python
async with HaikuRAG(db_path, config, create=True) as rag:
    # Document operations
    doc = await rag.create_document(content, uri, title, metadata)
    doc = await rag.create_document_from_source(path_or_url, title=..., metadata=...)
    doc = await rag.import_document(docling_doc, chunks, uri, title, metadata)
    docs = await rag.import_documents([DocumentImport(docling_doc, chunks, uri, title, metadata), ...])  # one table version per table
    doc = await rag.get_document_by_id(id)
    doc = await rag.get_document_by_uri(uri)
    docs = await rag.list_documents(limit, offset, filter)
    count = await rag.count_documents(filter)
    await rag.update_document(document_id, content=..., metadata=..., chunks=..., title=..., docling_document=..., uri=...)
    await rag.delete_document(id)

    # Processing (no database access)
    docling_doc = await rag.convert(Path("file.pdf"))  # a str that is not a URL is converted as text
    chunks = await rag.chunk(docling_doc)
    title = await rag.generate_title(doc)

    # Search & QA
    results = await rag.search(query, limit, filter)    # not expanded
    expanded = await rag.expand_context(results)
    answer, citations = await rag.ask(question, filter=None, images=None)  # images: Sequence[bytes], needs vision: true

    # Lookups
    doc = await rag.resolve_document(id_or_title)
    chunk = await rag.get_chunk_by_id(chunk_id)
    png = await rag.get_picture_bytes(document_id, self_ref)

    # Maintenance
    async for doc_id in rag.rebuild_database(mode=RebuildMode.FULL): ...  # also RECHUNK, EMBED_ONLY, TITLE_ONLY, DESCRIPTIONS, SET_EMBEDDER
    await rag.vacuum()

    # Tags (Store-level; TagInfo has tables/missing_tables/complete)
    await rag.store.create_tag("release-1")
    tags = await rag.store.list_tags()
    safety_tag = await rag.store.restore_tag("release-1")
    await rag.store.delete_tag("release-1")

    # Visualization
    images = await rag.visualize_chunk(chunk, source=result.source)  # source required over a set

    # Coverage (see "Multiple Databases")
    rag.covers_multiple      # more than one database
    rag.source_names         # configured names covered, in order
    rag.source               # the one name, or None while covering a set
    rag.location             # configured URI or path, None while covering a set
    owner = await rag.reader_for("papers")          # the client reading that database
    papers, wiki = await rag.clients_for(["papers", "wiki"])
    covering = await rag.clients_covering(sources)  # honours a per-query selection
```

`download_models(config)` (client/downloads.py) is a module function, not a method.

## Multiple Databases

`lancedb.databases` maps a name to a location. The name is the only identity that
leaves the configuration: it travels in `SearchResult.source`, `Citation.source`,
`Document.source` and errors, where a location must not.

- **The configuration places databases, arguments select.** `DatabaseScope.resolve(config, database_name=, database_path=)`
  is the single place selection happens, and it never mutates config. Every
  database has a name: the `lancedb.databases` key, or the path's stem. A name
  selects one configured entry. A path places a database only where the
  configuration places none, and beside `lancedb.databases` raises
  `AmbiguousDatabaseError`. Neither selector covers the configured set, a set of
  one included. Nothing configured is the one entry `haiku.rag` at
  `storage.data_dir / "haiku.rag.lancedb"`, selectable by name.
- **`--db` is the one human override:** `DatabaseScope.at(path)` consults no configuration.
  `DatabaseRef.given` marks a caller-given path whose errors may name it. A
  configured or default database's errors name the database and never its
  location (`SourceUnavailableError`, with the remedy for a missing one).
- **No derived configs.** `lancedb.uri` and `HAIKU_RAG_DB` do not exist, and a config carrying `uri` fails validation naming `databases`. The name always travels with the scope: `create_mcp_server` and the capabilities take or resolve one, and `ChatApp` sets `capability.scope`.
- **Never manufacture a default path.** `HaikuRAG(None, config)` resolves the configuration. A manufactured `data_dir / "haiku.rag.lancedb"` beside a configured placement raises `AmbiguousDatabaseError`. Downstream hosts, the ingester included, build clients from config alone.
- **Two sessions, two search paths.** `SingleDatabaseSession` reads and writes.
  `FederatedSession` composes single sessions lazily and reads only. A selection of
  one runs the ordinary single-database search: fusion would replace the
  database's hybrid scores with ranks, and embedding up front would embed for a
  filter the repository can see matches nothing.
- **Fusion.** With a reranker, it scores the union directly. Without one, the
  union is ordered by cosine similarity to the query vector, which is comparable
  across databases because the selection shares an embedder. Full-text-only
  searches order by retrieval score. Ties resolve by within-database rank, then
  configured order. Fused results carry the ordering score, so context expansion
  preserves fused order. `Chunk.embedding` is populated only when cosine fusion
  will read it (`with_vectors` mirrors `_fuse`'s cosine condition; an image query
  with a reranker configured still takes cosine). Over-fetch (`limit * 10`) only
  with a reranker. Image queries are vector-only and skip the reranker.
- **`_fuse` maps a reranked chunk back by `id(chunk)`**, since chunk ids repeat between copies. Every shipped reranker indexes into the list it was given. One that returns copies raises a named `ValueError`.
- **One reranker per set.** A client covering a database for another borrows the
  lender's (`_lender`, `_own_reranker`). Only the owner closes it.
- **Collisions.** Chunk ids repeat between copies of a database. In-memory
  identity uses `qualified_id(source, id)`. Serialized structures cannot
  represent it, so a cited id retrieved from, or previously cited from, more than
  one selected database raises `AmbiguousCitationError`. A copy the search never
  returned grounded nothing, so one retrieved result resolves normally.
- **`clients_covering(sources)` opens the databases; `_require_known_sources` only checks names.** Validating at an operation boundary must use the latter, or an unscoped question opens every configured database before the model runs.
- **`UnknownDatabaseError` subclasses `KeyError`** and overrides `__str__`. `pytest.raises(KeyError)` cannot tell the contract from a bare one, so assert the specific type.
- **Model context** names the database as a `Collection:` line, only when the
  search spans more than one. Storage vocabulary stays "database", the model
  boundary says "collection".

## Configuration

**Config classes** (config/models.py):
- `AppConfig` - Root config
- `StorageConfig` - data_dir, auto_vacuum, vacuum_retention_seconds, compaction_target_bytes
- `EmbeddingsConfig` / `EmbeddingModelConfig` - provider, name, vector_dim, multimodal (gates image embedding), batch_size
- `RerankingConfig` - optional reranker model, multimodal (vllm and openrouter)
- `ModelConfig` - provider, name, base_url, api_key, thinking, temperature, max_tokens, vision, extra_body
- `QAConfig` - model, max_searches (default 5), max_executions (default 15). The budgets are per question. `max_searches` counts search *units*: searches emitted in one model response (same `RunContext.run_step`) share a unit, up to `FREE_SIBLINGS_PER_ROUND` (3) per unit, so a run searches at most `max_searches × 3` times and a sequential searcher pays one unit per search. A budget-rejected round fails its remaining siblings
- `SandboxConfig` - code_timeout, max_output_chars: limits of one `execute_code` call, read by the capability and the MCP tool. In-code `search()` does not count against `qa.max_searches`, so code execution outlives a spent search budget
- `PictureDescriptionConfig` - picture description model settings
- `ProcessingConfig` - chunk_size, converter, chunker, chunker_type, conversion_options, pictures, split_pages, conversion_timeout, auto_title, title_model
- `SearchConfig` - limit, max_context_chars, vector_index_metric, vector_refine_factor, vector_nprobes
- `DoctorConfig` - duplicates (DuplicateDetectionConfig: similarity_threshold 0.97, min_chunks 3)
- `ProvidersConfig` - ollama, docling_serve (OllamaConfig, DoclingServeConfig)
- `PromptsConfig` - domain_preamble, picture_description
- `IngesterConfig` - sources, queue (QueueConfig), workers (WorkerConfig), api (APIConfig)
- `EvaluationsConfig` - judge (ModelConfig | None)
- `LanceDBConfig` - databases (name → local path or URI), api_key, region, storage_options, read_consistency_interval_seconds, index_cache_size_bytes, metadata_cache_size_bytes

Every model inherits `ConfigModel` (`extra="forbid"`), so an unknown or misspelled key raises at load. converter, chunker and chunker_type are `Literal`s and numeric fields carry bounds. Provider fields stay unrestricted `str`: `get_model` passes an unknown provider to pydantic-ai by name, logging a warning when model settings are set, since none apply. Read config through `get_config()`: there is no module-level `Config` singleton, and capturing the config in a default argument freezes it before `set_config()` runs.

A `qa.model` block's unset fields take `ModelConfig` defaults (`vision=False`, `thinking=None`), not `QAConfig`'s default factory (`vision=True`, `thinking=True`, `temperature=0.3`). The same holds for `title_model` and `picture_description.model`.

Adding a field or sub-model to `AppConfig` needs no plumbing: the loader validates it, and `init-config` and `settings` serialize `AppConfig().model_dump()`.

**Search order:** `--config` / `HAIKU_RAG_CONFIG_PATH` → `./haiku.rag.yaml` → platform user directory

**Global CLI options:** `--db-name NAME` selects one entry from
`lancedb.databases` and is global, so it precedes the command. `--db PATH` is
per-subcommand and follows it. They are mutually exclusive. `search`, `ask`,
`chat` and `mcp` cover the configured set, everything else works on
one database. `settings`, `init-config` and `download-models` resolve no scope.

## CLI Commands

```
# Document management
add           Add document from text
add-src       Add from file/URL/directory/S3
get           Get document by ID
delete/rm     Delete document by ID
list          List documents (with optional filter)

# Search & QA
search        Hybrid search (vector + full-text), not context-expanded
ask           QA via the RAG capability (always shows citations; --image PATH repeatable)

# Maintenance
init          Initialize new database
rebuild       Re-convert, re-chunk, re-embed (--rechunk, --embed-only, --title-only, --descriptions, --set-embedder)
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

# Tools
visualize     Show visual grounding for a chunk
inspect       Launch TUI to inspect database
chat          Launch TUI for conversational RAG
download-models  Download Docling, HuggingFace and Ollama models
```

`search`, `ask`, `list`, `get`, `visualize`, `chat` and `inspect` always open read-only. `cli()` turns the domain errors and `FileNotFoundError` into `Error: …` and exit 1.

**haiku-ingester** (separate entry point, `[ingester]` extra):
```
serve       Run pollers + workers, and the HTTP API unless --no-api; blocks until SIGINT/SIGTERM
run-batch   One discover sweep over every source, drain the queue, exit; non-zero on dead-letter or incomplete sweep
queue       init | migrate the queue DB
```

**Tags:** `haiku-rag tag create/list/delete NAME` name database states across all five tables. `tag restore NAME` brings the live database back to a tagged state: it creates a `before-restore-*` safety tag first, requires stopped writers, and never migrates. Vacuum retains the oldest tagged version and everything newer.

## MCP Server Tools

Exposed via FastMCP (mcp.py). `create_mcp_server(db_path?, config?)` resolves a
scope. `_covering(scope, config)` takes one already resolved. The server opens
one read-only client for its lifetime and covers the configured database set.
There are no write tools: ingest with `haiku-rag add`/`add-src`/`delete` or `haiku-ingester`.

- `search_documents(query, limit?, include_images?, filter?, sources?)` → `ToolResult`
- `search_documents_by_image(image_base64, limit?, include_images?, filter?, sources?)` → `ToolResult` (registered only when the embedder `supports_images`)
- `get_document(document_id, source?)` → Document
- `get_document_outline(document_id, source?)` → list[OutlineNode]
- `get_document_section(document_id, section_id, source?)` → DocumentSection
- `list_documents(limit?, offset?, filter?, sources?)` → list[DocumentInfo]
- `execute_code(code, filter?, sources?)` → the program's stdout

**Search results are text, not JSON.** Both search tools expand through
`HaikuRAG.expand_context` and return the `format_for_agent` rendering (rank,
`Document ID`, `Collection` over several databases, matched chunk metadata,
passage) plus one `ImageContent` per distinct picture. They carry no structured
content, see the Claude Code gotcha under MCP. Every other tool returns a
pydantic model, whose JSON is the payload.

**`execute_code`** runs one program per call in the same Monty sandbox the
RAG capability uses, over the documents `filter` and `sources` select, and
returns what it printed. One sandbox per call: a session outliving the call
would hit Monty's cumulative duration budget and never see documents ingested
after its first mount. The server's value to a client that is already a model is
the sandbox, so there is no `ask_question` or `analyze` tool and
`haiku.rag.utils` has no `format_citations` (`format_citations_rich` and the
`_citation_*` helpers stay, for the CLI).

**One error contract.** `mask_error_details=False`, passed explicitly since the
setting is also read from the environment: every failure reaches the client as
an MCP error carrying its message, host errors inside a program included. A bad
filter fails in the read itself with the query engine's message and an unknown
collection with `UnknownDatabaseError`'s, so there is no pre-check query and no
`ToolError` translation. Explicit domain errors stay (`No document with id`,
`No section`, `Invalid base64 image`).

**Plugin bundle.** `plugins/haiku-rag/` holds one `.mcp.json` (`haiku-rag mcp
--stdio`) and one Agent Skill, `skills/haiku-rag/SKILL.md`, under two manifests:
`.claude-plugin/plugin.json` and `.codex-plugin/plugin.json`. Marketplaces are
`.claude-plugin/marketplace.json` (Claude Code) and `.agents/plugins/marketplace.json`
(Codex). Both manifests carry the package version: `scripts/bump_version.py`
rewrites both, and `tests/test_bump_version.py` pins them to
`haiku_rag_slim/pyproject.toml`. A test asserts the skill's `allowed-tools`,
prefix stripped, equal the tools the server registers. The shared `.mcp.json`
uses Claude's `mcpServers` key, which Codex's loader accepts although its
documentation does not list it: don't "fix" it to `mcp_servers`. `allowed-tools`
is Claude Code's pre-approval and other Agent Skills clients ignore it.

## Testing

**Key fixtures** (tests/conftest.py):
- `temp_db_path` - Isolated temp database (use this!)
- `qa_corpus` - Q&A pairs from `tests/data/qa_corpus.json`
- `temp_yaml_config` - Temp config file, pointed at via `HAIKU_RAG_CONFIG_PATH`
- `allow_model_requests` - Enables pydantic-ai model calls (disabled by default)
- `allow_expected_model_requests` - autouse: enables model calls for `vcr` and `integration` tests
- `set_mock_api_keys` - autouse: mock provider keys for playback
- `postgres_dburi`, `docling_serve_url` - integration services, skip when unreachable
- `capture_logs(logger, level)` - context manager collecting log records (see gotcha)

**Markers:**
- `@pytest.mark.integration` - External services (postgres, docling-serve, seaweedfs). CI excludes them. Start them with `docker compose -f tests/docker/docker-compose.yml up -d`. Each test skips when its service is unreachable.
- `@pytest.mark.slow` - Deterministic end-to-end tests. CI runs them separately with `pytest -m "slow and not integration"`.
- `@pytest.mark.vcr()` - HTTP call recording/replay

`pytest` runs the whole core suite, slow and integration included. CI's fast lane is `pytest -m "not slow and not integration" --cov`. `test-windows` runs it on Windows without `--cov`, as a check that is not required. `cd evaluations && uv run pytest --cov` runs the evaluations suite and its coverage gate. Tests run under xdist (`-n auto` in addopts); pass `-n0` when debugging races or container lifecycle. Async tests are detected through pytest-asyncio's auto mode, and asyncio fixtures default to **session** loop scope.

**Core coverage is enforced at 100%** (`fail_under = 100`, `source = ["haiku_rag_slim"]`); the evaluations workspace enforces 85%. New code needs a test or a `# pragma: no cover - <short reason>` on one line. CI prints term-missing, so a gate failure names the line. Run coverage with `--cov` and scope the report by grepping it: a deep-dotted `--cov=haiku.rag.<module>` crashes beartype, and a file-path `--cov=<path>.py` collects nothing.

**VCR Recording:**
Tests use pytest-recording (VCR.py). Cassettes are committed: the suites in `_CENTRAL_CASSETTE_SUITES` (client, chunkers, converters, embeddings, interfaces, providers, reranking, store) record under `tests/cassettes/<module>/`, every other suite under `<suite>/cassettes/<module>/`.

```bash
# Run with recorded cassettes (default)
pytest

# Record new cassettes (exact test, serial, real services available)
pytest tests/embeddings/test_embedder.py::test_ollama_embedder -n0 --record-mode=rewrite

# Run against live services (disable VCR)
pytest --disable-recording

# Record with a real API key (each SDK reads its own variable, e.g. CO_API_KEY)
CO_API_KEY=... pytest tests/reranking/test_reranker.py::test_cohere_reranker -n0 --record-mode=rewrite
```

To add VCR to a new test, add `@pytest.mark.vcr()` and record with `pytest <path> -n0 --record-mode=once` against the real service. Without `--record-mode`, pytest-recording plays back in `none` mode, so an unrecorded call errors with a connection-style failure. Recording needs network.

Docling-serve polling and retry delays are skipped only during cassette playback. Recording and `--disable-recording` runs keep them.

## Code Conventions

- Python 3.12+ native typing (no `from __future__ import annotations`)
- Absolute imports (the `store` package `__init__` files are the only relative ones)
- Async/await for all I/O operations
- LanceDB is opened via `lancedb.connect_async`; all table reads/writes are awaited. `connect_async` requires absolute paths, so `connect_lancedb` makes a local location absolute. `Store(location, config)` takes a path or a URI; `Store.db_path` is `None` behind a URI.
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

Each entry is the trap and what to do. The evidence behind them is in the commit history.

### Store and LanceDB

- **One write transaction.** Multi-table writes go through `async with store.write_transaction():`. It takes `_write_lock`, snapshots `current_table_versions()`, and on any exception restores every table in `RESTORE_TABLE_ORDER` under a shield before re-raising (an absorbed `CancelledError` is re-delivered). Client code never restores tables itself.
- **Lock order** is `_rebuild_lock → _write_lock` (tag ops, restore) and `_vacuum_lock → _write_lock` (vacuum). Rebuild holds `_rebuild_lock` for its whole run, and tag operations fail fast against it. All in-process: cross-process writers need a quiescent boundary for tag create and restore.
- **One version per write.** Each `create_document*` / `import_document` writes a version of `documents`, `chunks` and `document_items`. Bulk ingestion in a loop fills disk with versions. Use `import_documents([DocumentImport(...)])`, or `DocumentRepository.create([...])` / `DocumentItemRepository.create_all(...)`.
- **Opening a Store never writes to it.** Stored embedding settings change only via `init`/`rebuild`, and the version is a forward-only marker written by `init` and `migrate`. `Store.stored_settings` and `Store.stored_embedding` are read at open and never follow a later write. Refresh them together through `_remember_settings`.
- **Embedding drift on open.** A `vector_dim` mismatch always raises `ConfigMismatchError`. Provider/name drift at the same dimension warns read-only and raises writable. Reconcile with `haiku-rag rebuild --set-embedder`. Maintenance commands, `list`, `get`, `visualize` and CLI deletion pass `skip_validation=True`. The MCP server does not: it opens one validating read-only client in its lifespan, so a `vector_dim` mismatch fails startup.
- **lance 8 `merge_insert`** rejects partial-schema sources upfront when an insert branch needs non-nullable columns the source lacks. Declare update-only merges (drop `when_not_matched_insert_all`) when matches are guaranteed.
- **Index sets** live in `index_specs()` / `ensure_indexes()` (schema.py), and every table-creating path routes through them. `ensure_indexes` matches on (column, index type) and never drops an index it did not declare. It skips FTS on an empty table (an index born over zero rows never catches up on `add`) and rebuilds an FTS index covering zero rows. The chunk write paths call it after writing. So a test needing the zero-coverage state writes via `chunks_table.add` directly, and the first chunk write into a fresh table writes one extra version.
- **A zero-coverage or missing FTS index** returns every match unsorted, in insertion order, so `limit` slices arbitrarily and hybrid fuses that order as a ranking. `ensure_indexes`' rebuild is load-bearing. Single-term probes on small tables look healthy.
- **`create_index(replace=True)` rebuilds an identical index** (new uuid, new version, old files kept until vacuum), and matches by index *name* (`{column}_idx` by default), so a custom-named index on the same column gets a sibling. On lancedb 0.38+ an unnamed replace builds a suffixed sibling beside a different-typed index. Check `list_indices()` first and pass `name=`.
- **`optimize()` on an indexed table writes a version**, so an `auto_vacuum` pass touches `documents`. A test measuring version deltas sets `storage.auto_vacuum = False`.
- **The lancedb `Session`** is process-wide and holds the index and metadata caches, so on object storage the first vector query's index load is paid once per process. `read_consistency_interval_seconds` defaults to 30: a connection sees its own writes at once, another process's after the interval.
- **Vacuum.** `Store.vacuum()` retains `storage.vacuum_retention_seconds` (default 24h). Tables with `large_binary` payloads (`documents`, `document_items`, matched by `has_payload_columns`) go through lance directly via `_run_lance_maintenance` (`compact_files(target_rows_per_fragment=…)`, `optimize_indices`, `cleanup_old_versions`, wrapped in `_wait_protected`). The other tables use `table.optimize()`. `AsyncTable.optimize` exposes no compaction options (lancedb#2325). To purge recent fragments, call the path with zero retention.
- **`compaction_target_bytes`** sizes fragments compaction *writes*. It cannot shrink a larger one, which lance rewrites whole once before splitting it. Peak memory is max(target, largest fragment), and large ingest batches make large fragments.
- **Tags** are per-table lance refs: a haiku.rag tag is the same name on all five tables from one `current_table_versions()` snapshot. Names are alphanumeric/`.`/`-`/`_` only. `AsyncTable.tags` mints a fresh `AsyncTags` per access, so fault-injection tests patch `AsyncTags` methods at class level with call counting.
- **Tags and vacuum.** `optimize(cleanup_older_than=...)` hard-errors when a tagged version falls in the cleanup window, and lancedb does not expose `error_if_tagged_old_versions`. `Store.vacuum` grows the retention to keep the cutoff older than the oldest tag (`TAG_RETENTION_MARGIN`) and suppresses `OSError` only. Do not pass `error_if_tagged_old_versions=False` on the lance path: with a tag present it prunes manifests and frees no bytes.
- **lancedb and pylance move in lockstep.** lancedb bundles lance as a Rust crate, so nothing ties `pylance==X` to a lancedb version. Pair them by release date, then verify by round trip: lancedb writes, pylance compacts, lancedb reads and writes again. Before a bump, check the PyPI wheels per platform and their upload times. After a bump, run `pytest tests/store/test_vacuum_bounded.py -m integration -n0`: CI never runs the vacuum memory tests. FTS BM25 scores change between lancedb versions, so retrieval benchmarks are pin-conditional.
- **Blobs.** No document read loads the docling blobs by default: `get_by_id`/`get_by_uri` project `id`+`content`, and `list_all` needs `include_content=True` for text. Load blobs with `get_docling_data`/`get_pages_data`, or `get_by_id(..., include_blobs=True)`, which any path writing the row back through `DocumentRepository.update` needs, or it persists `None` over the blobs.
- **`Document.get_docling_document()`** decompresses only the structure blob. `set_docling(...)` after a structure-only load writes `docling_pages=None`. Update `docling_document` directly via `compress_docling_split`.
- **`document_items` across documents** goes through the grouped accessors only (`resolve_refs_grouped`, `get_items_in_ranges`, `get_pictures_grouped`, `get_caption_picture_refs_grouped`). Each builds a per-document predicate, `(document_id = 'a' AND col IN (…)) OR (…)`: `self_ref` values collide across documents, and `self_ref IN (union)` over-fetches and can hand one document another's picture. Tests assert the query counts.
- **`get_pictures_grouped`** defaults `with_text=False`. Only the enrichment path opts in.
- **`expand_with_items`** does not fetch: it takes positions and window items, and `expand_context` fetches once for every document. Assemble results in `document_groups` order, since the stable score sort uses arrival order as the tiebreak.
- **Embed-only rebuild recreates the chunks table**, so patching `store.chunks_table.add` on the instance is discarded. Patch `lancedb.AsyncTable.add` at class level and filter on `self.name`. Its phase 2 writes via `chunks_table.add`. A FULL rebuild refreshes source-backed documents in place, keeping the id (re-fetch with `force=True`, falling back to stored content).
- **`client.update_document` re-chunks and re-embeds**, so it needs `@pytest.mark.vcr()`. To rewrite stored content without the embedder, use `DocumentRepository.update`.
- **Context expansion and `visualize_chunk` without the embedder:** `import_document(docling_doc, [Chunk(..., embedding=[0.1] * vector_dim)])`. Build the `DoclingDocument` with `add_page`/`add_text` and `ProvenanceItem` bboxes for page-image tests.
- **Throwaway lancedb scripts:** `uv run python` from the repo root. The sync `LanceTable` has no `.query()`. `.to_lance()` works.

### Docling and conversion

- **docling and docling-serve are pinned as a pair**: `docling==2.124.0` + `docling-core==2.93.0`, and `quay.io/docling-project/docling-serve:v1.32.0` in `tests/docker/` and `examples/docker/`. The coupling is `tests/chunkers/test_chunker.py::test_local_and_serve_chunkers_produce_same_output` (slow, vcr), which the `test-slow` CI job enforces. Take the client version from the serve release's bundled library set and move both together. Re-record the serve cassettes when the image moves: that test plus the slow vcr `TestDoclingServeConverterIntegration` tests. Never use `:latest` in a compose file: Docker never re-pulls it.
- **`pdf_backend` defaults to `docling_parse`**, not docling's `threaded_docling_parse`, which never returns on some documents on Linux (all threads parked in `standard_pdf_pipeline.get_batch`). docling caches its `DocumentConverter` per process, so one wedge stalls later conversions on any backend: recovery needs a new process. Reproduce in the `docling-serve:v1.32.0` image (linux/arm64, root, `faulthandler.register(SIGUSR1, all_threads=True)` plus `kill -USR1`). Before attributing a conversion difference to a docling version, hold `pdf_backend` constant.
- **`processing.conversion_timeout`** (600s) bounds one conversion. Every part is deliberate: the deadline starts when the conversion *holds* `_CONVERTER_LOCK`; the lock is owned by the conversion thread, never the coroutine; the deadline runs in a task the caller `asyncio.shield`s; conversions run on daemon threads through `contextvars.copy_context().run`, not `asyncio.to_thread`; a queued conversion polls `acquire(timeout=0.5)` and re-checks `_WEDGED` before and after acquiring. Do not simplify any of it.
- **`TestConversionTimeout`** needs a fresh `_CONVERTER_LOCK` per test (`monkeypatch.setattr(docling_local, "_CONVERTER_LOCK", threading.Lock())`), never a forced `release()`. Stall stubs stay bounded by a `threading.Event`.
- **`processing.split_pages`** does not reproduce single-pass conversion: a paragraph or caption crossing a slice boundary regroups or reorders, while the word multiset stays identical. Only no boundary guarantees equality, which is what the `concatenate` test uses.
- **Labels are platform-dependent.** A label survives only past docling's hard-coded 0.5 layout confidence, so a borderline element flips between machines, and docling-serve substitutes a synthetic text item. A local/serve parity test compares items, item text, page numbers, chunk count, chunk text and word multisets, never labels or `self_ref`.
- **PDF control characters** (a BEL for a symbol-font list marker, tabs in headings, `/tildelow` glyph names) reach chunk text. Deliberately not fixed (issue #613). If it needs fixing: `Cc` covers control characters, a tab is a normalization question, and a glyph name is indistinguishable from text. Don't strip `Cf` (soft hyphens). The seams are the chunkers' `content=` sites or a post-conversion pass, not `prepare_text_content`.
- **Inline groups.** `converters/base.py::flatten_inline_groups` normalizes `InlineGroup`s at the three points a `DoclingDocument` is built (both `docling_local` sync paths, `docling_serve._parse_zip_to_docling`) and nowhere else. `import_document(s)` store the caller's document as given, since their chunks' `doc_item_refs` index into it, and `extract_item_text` recovers an empty owner's text there. `rebuild --rechunk` re-chunks the stored blob unnormalized. Only html, md, msword, opendocument, boxnote, webvtt and jats emit inline groups. The owner is the item with empty `text`, never merely the parent. A group with non-text runs, several provenance records or an inbound ref is left alone. `TestInlineGroups` pins the contract. Upstream: docling-core #768, #769, #770.
- **Mutating a `DoclingDocument` is O(document) per call**: batch `delete_items`. `DocSerializer` caches excluded refs by `self_ref`, so serialize everything before mutating. A ref into a cascade-deleted descendant is renumbered wrongly, and `GraphCell.item_ref` is renumbered by nothing. Never give a merged item several `ProvenanceItem`s: the doclang serializer emits the text once per record.
- **`MarkdownParams` escapes by default** (`escape_underscores`, `escape_html`). A serializer whose output becomes stored text sets both False (`flatten_inline_groups` and `extract_items` do, and `extract_items` sets `image_placeholder=""`).
- **HTML images** load through `docling/backend/utils/image_resource_loader.py`. Monkeypatch `ImageResourceLoader.load_image_data` (`(self, src_loc, base_path)`), not the backend method. Fetching is gated by `enable_remote_fetch`, which our options set from `fetch_remote_images`.
- **HTML furniture.** docling's HTML backend files everything before the first heading as `ContentLayer.FURNITURE`, which every reader of the body layer drops silently. `processing.conversion_options.infer_furniture` (default False) turns it off on `docling-local`. docling-serve has no such option.
- **rapidocr 3.9.2** ships LaTeX in a docstring that beartype's import hook compiles. `pyproject.toml` carries `"ignore:invalid escape sequence:SyntaxWarning"`.
- **`providers.docling_serve.timeout`** bounds each HTTP call, not the conversion: the status poll has no deadline.

### Ingester

- **Two databases.** Ingester state is LanceDB (documents) and the queue (jobs, sync_state). `ingester/reconcile.py` reconciles them at every `serve`/`run-batch` start from one `document_meta` listing: restore sync_state rows for owned documents, clear revisions through `SyncStateRepo.invalidate` for URIs the index lost, attribute unowned ingested documents through `HaikuRAG.set_document_source`. The evidence of a successful write is `last_ingested_at` (`list_ingested_uris`), never a revision: the permanent-failure path writes a revision with `ingested=False`. `upsert` cannot clear a revision (`_upsert_stmt` coalesces it), hence `invalidate`. A URI two sources ingested stays unattributed. Each repair is one transaction.
- **Queue path** defaults to `get_default_data_dir() / "ingester.db"`, not `storage.data_dir`. `ingester.queue.dburi` points it at Postgres.
- **Ingester test doubles** use `AsyncMock(spec=HaikuRAG)`. They stub `_ingest_observed`, not `create_document_from_source`, and must give `list_documents` the documents they stored: an empty MagicMock reads as "the index lost everything" and invalidates every revision. Function-style doubles use explicit signatures, so a new kwarg on the real call raises `TypeError`, which `_classify` turns into a transient failure. Update every double when the signature changes.
- **`_ingest_observed(observed_revision=)`** is the ingester's entry point, passing `job.revision`. It is not on the public `create_document_from_source`. With it the revision short-circuit skips its `head()`. It decides only whether to skip: a mismatch falls through to the fetch, and `force` never reaches it. A source changing after discovery waits for the next sweep.
- **`create_document_from_source` metadata merges** differ by path: full ingest does `{**user_metadata, **source_metadata}`, the revision short-circuit's `_refresh_doc_metadata` does `{**doc.metadata, **user_metadata}` before `source_metadata`. Strip reserved keys (`md5`, `source_revision`, `content_type`, `source_id`) from caller metadata. `source_id` is the ingesting source's id, or the stored one when the caller names none (`_keep_source_id`), so an ad-hoc re-ingest cannot detach a document. A different source id takes ownership with a WARNING. PDF attachment children get `user_metadata={}` and no `source_id`.
- **Stalled conversions.** `PermanentError` carries `conversion_stalled` (tombstone this document) and `fatal_to_process` (exit) separately, since an HTML/Markdown stall strands no converter. The worker records the job dead before exiting, and exits from a `finally`: `reap_stale` refunds the attempt of an owner that vanished, so exiting first requeues the poison document forever.
- **Queue schema 3** adds `jobs.conversion_stalled` and `uq_jobs_blocking_op`, a partial unique index over `(source_id, uri, op, coalesce(revision, ''))` for live and stalled jobs. Enforced by the index because no read is atomic against a concurrent `mark_dead`. `coalesce` because unique indexes treat NULLs as distinct. `prune_terminal` skips these rows. A new revision, a DELETE, `retry` and `prune_dead` free the slot. The migration emits the index from the `Index` object `create_all` uses.
- **`uq_jobs_live(source_id, uri)`** allows one live job per (source, uri), so a DELETE cannot run after a sibling UPSERT. Rapid (deleted, added) events squash to delete-only, which `FSPoller._handle_watch_change` mitigates with a `path.exists()` recheck. Keep it.
- **Queue migrations are versioned** (`ingester/queue/migrations.py`, `SCHEMA_VERSION`). `create_all` never adds a column to an existing table: a new column needs the model change, an explicit `ALTER TABLE` (+ backfill) under `current < N`, and a `SCHEMA_VERSION` bump.
- **Queue engine.** SQLAlchemy Core async. SQLite is `pool_size=5, max_overflow=5` with PRAGMA listeners. Postgres (`postgresql+asyncpg://`) gets `pool_pre_ping=True` and claims with `FOR UPDATE SKIP LOCKED`. `claim_next` must stay one `UPDATE … WHERE id=(SELECT … LIMIT 1) RETURNING`. Build the SQLite URL with `URL.create("sqlite+aiosqlite", database=str(path))`, never an f-string: `?` and `#` in a path become query and fragment.
- **Postgres queue tests** build the engine inside the test with `NullPool` (asyncpg binds a connection to its loop) and isolate xdist workers with a per-test schema, since `--dist load` ignores `xdist_group`.
- **Coverage** uses `concurrency = ["greenlet", "thread"]` for the aiosqlite worker thread. Postgres-only construction branches are covered by build-without-connect tests (`make_engine(QueueConfig(dburi=...))`, `_insert(jobs, "postgresql")`).
- **Reaping is lease-based.** Workers renew `last_heartbeat_at` every `heartbeat_interval_s`. `reap_stale` resets a claim when `COALESCE(last_heartbeat_at, claimed_at)` is older than `lease_ttl_s`, and never cancels a live coroutine. `WorkerConfig` enforces `heartbeat_interval_s <= lease_ttl_s / 3`. `claim_timeout_s` is gone.
- **Worker ids** are `{pid}-{uuid}-{n}`. Boot-reap is scoped to `lease_ttl_s`, so a peer's live claims survive. `_untrack_inflight` drops only the pool's own entry for a job, or it stops renewing a sibling's re-claim.
- **`POST /sources/{id}/refresh`** runs `_sweep_once`, which skips (`refreshed: false`) while the source has pending jobs or an open breaker. Default FS `poll_interval_s` is 300.
- **obstore exceptions** subclass only `obstore.exceptions.BaseError`. `workers/pipeline._classify` maps `PermissionDeniedError`, `UnauthenticatedError`, `UnknownConfigurationKeyError` and `InvalidPathError` to `PermanentError`. A missing object is a builtin `FileNotFoundError`, already permanent.
- **An exception out of an FS periodic sweep dies silently**: `_sweep_loop` is a task `run()` never awaits, `live_pollers` counts the outer task, and shutdown's `gather(..., return_exceptions=True)` swallows it. `/health` stays green.
- **Poller discovery failures** are handled in three places with the same `record_failure` + `logger.exception` body: `pollers/base.py` `_sweep_once` and `_dry_run_once`, and `pollers/fs.py` `_watch_loop`. Change one, change all three.
- **Plugins** use `importlib.metadata` entry points: `haiku.rag.metadata_providers` and `haiku.rag.sources`. Only referenced ones load.
- **Docker images.** Both Dockerfiles install `--extra ingester`, which does not pull `[docling]`. The full image gets docling from the `haiku.rag` package itself. The published slim image cannot run `converter: docling-local`. `[ingester]` pulls `[s3]`.
- **Dashboard** (`ingester/api/static/index.html`) is outside the biome hook's scope. Run `biome lint <file>` by hand, and `node --check` on the extracted script.

### Capabilities and sandbox

- **`RAGState` is flat** and is dumped and re-validated at every carry point: `client/agents.py`, `chat/app.py`, `evaluations/capability_runner.py`, and `app/backend/main.py`, where the AG-UI client returns the snapshot next turn. Nesting or renaming a field is a breaking wire change. Additions are safe. `begin_invocation()` drops the previous question's `citations`, `searches` and `executions`.
- **Agent specs construct capabilities through `from_spec`**, never `cls()`: `id` comes from `create_capability()`, `rag.py` filters tools by `tool.capability_id != self.id`, and pydantic-ai's duplicate-id rejection enforces one compaction and one policy capability per run. pydantic-ai passes raw parsed YAML (`db_path` a `str`, `config` a `dict`). Third-party capabilities are never auto-discovered: the caller passes `custom_capability_types=[...]`. A spec needs a model, in the spec or as a kwarg. A zero-argument `from_spec` override keeps per-run caches out of the schema.
- **`_cite` repairs a near-miss chunk id** to the nearest id the run retrieved, above `CHUNK_ID_MATCH_CUTOFF` (0.75), before the database fallback, and before `resolve_citations`, so `cited_map` order holds.
- **Repeated cite reminders in `capabilities/instructions/*.md` are load-bearing.** Treat any prompt-length reduction there as a behaviour change needing its own eval.
- **Repeated search queries accumulate**: `state.searches[query]` merges through `merge_results` keyed on `chunk_id`. Hand-built `SearchResult`s without a `chunk_id` collapse to the first. `search_corpus` returns `"No results found."`, never an empty string.
- **Sibling dedup.** Searches in one model response collapse evidence a sibling already showed to `Also matched, shown above: [id]`, and attach each picture once per response on `(source, document_id, self_ref)`. Equivalence is `evidence_signature` (`_tools.py`). `state.searches` keeps results whole. Search state commits only after formatting and image construction succeed. Dedup never crosses run_steps. In-code `search()` bypasses pricing, dedup and spans.
- **User-attached images** need the instructions' "Questions with attached images" section, or vision models refuse without searching.
- **`SearchResult.document_meta` / `Citation.document_meta`** carry the document's metadata for UIs, excluded from `format_for_agent`. `chunk_meta` is the verbatim `Chunk.metadata`, typed keys included: on an expanded result it is the anchor chunk's, and `format_for_agent` shows it only with `include_chunk_meta` (the MCP search tools set it).
- **Sandbox** (`sandbox/sandbox.py`): pydantic-monty in a subprocess worker from an `AsyncMonty` pool, with read-only `CallbackFile`s under `/documents/{id}/`. `metadata.json` is prebuilt, `content.txt` lazy per document, `items.jsonl`/`chunks.jsonl`/`toc.json` a lazy bulk cache. One session serves every `execute()` call of a capability run. `_run_on_loop` bridges sync reads to async queries. A host error keeps its message for every caller.
- **`sandbox.code_timeout` is a deadline**: `_check_deadline` runs before every host call (`_timed` wraps the five readers, in-code `search()` and `list_documents()` call it first, `_run_on_loop` keeps its own), because cached reads never touch the bridge. Monty's watchdog stops compute.
- **Monty caps host callbacks** per checkout (`max_suspensions`, 1000 by default), with no way to disable it. `_session_limits` sets `_MAX_HOST_CALLS = 10_000_000`, so the time budgets govern.
- **Monty runs VFS callbacks on a thread coverage does not trace**: a reader's lines need a direct-read test (`tests/sandbox/test_sandbox_toc.py`).
- **Compaction and policy hooks.** `EvidenceCompactionCapability` discovers the evidence capability through `RunContext.capabilities` and rewrites `request_context.messages` in `wrap_model_request`, never the stored history, which keeps question identities and epochs (message counts) meaningful. `CitationPolicyCapability` reads the same records in `after_model_request` / `after_run`. `cite([])` records a declaration with no refs (`ungrounded`), distinct from declaring nothing (`missing`).
- **Context expansion** (`context.py`) is section-bounded: a section within `max_context_chars` returns whole, a larger one expands outward inside its edges, one under 20% of the budget grows across boundaries (picture and table matches excepted). Only overlapping ranges merge, and the result is hard-capped at `max_context_chars`. `visualize_chunk` expands before resolving bounding boxes. `HaikuRAG.search` does not expand: the capability, MCP and in-code search call `expand_context`.
- **`toc.json`'s `item_range`** indexes the position-ordered item list (the line numbering of `items.jsonl`), not database positions. `get_document_section` slices the same way.

### MCP

- **`FastMCP.lifespan()`** combines provider lifespans only and never runs the constructor's `lifespan=`. Tests use `_lifespan_manager()`. The server's lifespan resets the cached client in a `finally`, since it can be re-entered.
- **Claude Code and the Agent SDK drop a tool result's text when `structuredContent` is set**, forwarding the JSON and images only. Desktop and claude.ai forward both. So a tool whose text is a rendering sends one channel, which is why the search tools carry no structured content and their tests read the text.
- **Claude Desktop ignores server `instructions`**, and only Claude Code honours `allowed-tools`, so every tool description stands alone. `execute_code`'s docstring carries the whole sandbox contract. Claude Code moves an MCP call to the background after about 120s.
- **fastmcp 4 / MCP SDK 2**: protocol types are snake_case, and the camelCase bridge goes in fastmcp 5 (hence `<5.0.0`). `Client` defaults to sessionless, where `initialize_result` is `None`: read `client.instructions` / `client.server_info`. `ToolResult` imports from `fastmcp.tools`. Tag visibility is `server.disable(tags=...)`. Server-initiated sampling is gone. Pass `version=` or the server reports fastmcp's version.
- **An in-process `fastmcp.Client` test** needs `@pytest.mark.filterwarnings("ignore:Found propagated trace context:RuntimeWarning")`: an earlier CLI test in the worker installs logfire's propagator. Shard-dependent. Reproduce with `uv run pytest tests/interfaces/test_cli.py <the test> -n0`.

### Models and providers

- **`get_model` branches** for ollama, vllm, openai, openrouter, anthropic, google, groq, bedrock and mistral, applying `temperature`, `max_tokens`, `thinking` and `extra_body`. Any other provider is returned as `"provider:name"` and pydantic-ai applies none of them, so it logs a warning.
- **`get_model` sets `openai_chat_supports_multiple_system_messages: False`** on OpenAI-compatible endpoints, since strict vLLM chat templates reject more than one system message.
- **Thinking on self-hosted endpoints.** `VLLMProvider.model_profile()` infers the family by name prefix, and pydantic-ai strips unified `thinking` where the profile lacks `supports_thinking`. So `ollama`, `vllm` and `openai` with a `base_url` send `openai_reasoning_effort` directly (`reasoning_effort_settings`): `False` → `none`, `True` → `medium`, a level as written. The server's chat template decides what it means. Verify a kwarg by wire capture and generation, never a rendered prompt: vLLM's `/tokenize` skips the `reasoning_effort` translation. `provider: vllm` is pydantic-ai's chat provider on a `ModelConfig` and haiku.rag's multimodal client under `embeddings.model` / `reranking.model`.
- **pydantic-ai's request guard sits above VCR**, rejecting a replayed embedder call before VCR sees it. The autouse `allow_expected_model_requests` enables model requests for `vcr` and `integration` tests. Symptom if removed: many `RuntimeError: Model requests are not allowed` plus a hang at the end of the run.
- **`LogfireQueryClient`** is `logfire.query_client` in logfire 5.

### Testing and tooling

- **Windows.** Text mode follows the host locale (cp1252), so read and write repo files with `encoding="utf-8"`. Build an expected rendered path with `pathlib`, not a literal `/`. Write a file under test through `tmp_path`: Windows refuses to reopen an open `NamedTemporaryFile`. Creating a symlink needs Administrator or Developer Mode; without it the `requires_symlinks` tests (`tests/platform.py`) skip and the symlink branch of `sources/fs.py` goes uncovered. Guard a POSIX-only call with `sys.platform`, which ty narrows on, not `hasattr` or `getattr`.
- **`haiku.rag`'s logger does not propagate**, so `caplog` misses its records once any test in the worker has called `get_logger()`. Use `capture_logs(logger, level)` from `tests/conftest.py`.
- **HF Hub outages** stall unmarked-network tests at setup. `HF_HUB_OFFLINE=1 pytest ...` runs from the disk cache.
- **`uv run ty check <files>`** checks only those files. The pre-commit ty hook is broader: run `uv run pre-commit run --files <paths>`.
- **ty treats `dict` as invariant**: type read-only parameters that accept subtype dicts as `Mapping[...]`.
- **Mutating a pydantic model's `model_config`** needs `Model.model_rebuild(force=True)` to take effect.
- **Bulk rewrites:** Python one-liners, not `sed -i`. BSD sed lacks `\|` alternation and silently no-ops.
- **`git grep`** floods on cassettes: pass `-- ':!tests/cassettes'`.
- **`Citation` and `resolve_citations`** live in `haiku.rag.store.models.citation`.
- **Rich-renderer tests** render with `Console(record=True, width=200)` and assert on `export_text()`.
- **`textual_image`**: `widget.Image` for Textual apps, `renderable.Image` for Rich consoles. Both take PIL images.
- **Textual `auto` grid rows ignore margins**: use bottom padding. Assert layout with `parent.region.contains_region(child.region)`.
- **Textual `pilot.pause()`** does not reliably flush a handler awaiting I/O on a slow runner: await the handler directly.
- **A line that runs once per process** (module-level lazy init) is covered only on some shards. Assert it directly with the module global reset via `monkeypatch`.
- **Commenting out every child of a non-Optional config block** parses to `null`, which Pydantic rejects. Drop the key or keep one child.
- **`doctor`** renders details through Rich markup, so avoid literal `[...]`. Near-duplicate detection keeps one summed centroid per document from the single vector scan, and `run_db_checks` keeps the vector matrix a local so `del vectors` bounds memory.
- **`astral-sh/setup-uv`** has floating major tags only through v7. Pin exact tags (workflows use `@v10.0.1`).
- **`haiku.rag.sources`** imports without the `[s3]`/`[ingester]` extras because `obstore` is imported inside `S3Source` methods. Keep it there.
- **One-shot directory ingest** uses `sources.fs.walk_files`, which never follows directory symlinks and keeps a symlinked file only when its target resolves inside the root. No size cap on that path.
- **`is_local_uri` and `uri_to_path`** (`uri.py`) own "is this local" and "what path is this", Windows drive letters and file-URI hosts included. Don't re-derive either.
- **`tests/test_uri.py`** runs in its own CI matrix without the project installed (`uv run --no-project ... --noconftest -o addopts=`). Keep it stdlib-only.
- **`app.py` and `cli.py`** are covered through Typer's `CliRunner`. `cli.py` keeps one pragma, on the `__main__` guard.

## Planning and Commits

When planning work, break the plan into **self-consistent, commitable chunks**. Each chunk should:
- Be a logical unit that can stand on its own
- Leave the codebase in a working state
- Be small enough for meaningful review

## Common Patterns

**Test with VCR recording:**
```python
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

Separate workspace package (`evaluations/`) for benchmarking. Run its tests from inside `evaluations/` (`cd evaluations && uv run pytest ...`): from the repo root, its `tests/` collides with the core `tests/`.

```bash
cd evaluations
uv sync
evaluations run <dataset>
evaluations pair <treated>.jsonl <baseline>.jsonl
evaluations download <dataset|all>           # Pre-built eval DBs from HuggingFace
evaluations upload <dataset|all>             # Upload eval DBs
```

Datasets: `frames`, `hotpotqa`, `orb_text`, `orb_multimodal`, `orb_multimodal_nemotron`, `t2_finqa`, `t2_tatdqa`, `mtrag_clapnq`, `mtrag_clapnq_rewrite`, `mtrag_clapnq_live`, `mtrag_clapnq_live_uncompacted` (the four mtrag keys share one DB).

**Capability runs** go end to end through a native Pydantic AI agent (`evaluations/capability_runner.py`). The capability model is `config.qa.model` and the judge `config.evaluations.judge`, with no command-line override for either.

**Citation retrieval metric**: `CitationMAPEvaluator` scores the URIs the capability registers via `cite` against gold `expected_uris`. Score key: `cited_map`.

**The judge is pinned and frozen.** Every reference config under `evaluations/configs/` whose dataset is judged carries the same block (`temperature: 0.6`, `max_tokens: 16384`, `extra_body` with `top_p` 0.95 / `top_k` 20 / `min_p` 0 / `chat_template_kwargs.reasoning_effort: low`), guarded by `evaluations/tests/test_reference_configs.py`. Do not change it without an eval: greedy decoding lost verdicts to repetition spirals on Qwen, and a larger `max_tokens` did not help. `DEFAULT_JUDGE_MODEL` carries only what ollama honours (`temperature`, `max_tokens`, `top_p`). `t2_finqa` / `t2_tatdqa` set `qa_evaluator`, which replaces the evaluator list, so no judge is constructed for them.

**Multi-turn (MTRAG)**: `mtrag_clapnq` runs gold-prefix QA (`ConversationInput` cases replay the reference prefix as message history) plus lastturn retrieval. `_rewrite` retrieves with the human rewrites. `_live` and `_live_uncompacted` set `spec.live` (one case per conversation, `--limit` counts conversations) and differ only in `spec.compaction`, which registers `EvidenceCompactionCapability`: the only eval coverage compaction has. Live runs carry `all_messages()` and one capability-state dict across turns, and record per-turn arrays (`turn_cited_uris`, `turn_n_search_calls`, `turn_n_sandbox_search_calls`, `turn_n_rejected_searches`, `turn_n_failed_tools`, `turn_n_executions`, `turn_n_requests`, `turn_citation_status`) counted from `new_messages()`. Gold-prefix and live pass rates are not comparable; compare compacted with uncompacted, paired by turn. `_live_summary`'s macro rate excludes conversations with zero judged turns.

**Run identity**: experiment metadata carries `git_sha` and `git_dirty` (untracked files ignored), `config_hash` (SHA-256 of the resolved `AppConfig`; `populate_db` disables `storage.auto_vacuum` on a copy), and the corpus fingerprint `db_path`, `db_documents`, `db_chunks`, `db_embedder_provider`, `db_embedder_model`, `db_embedder_dim`, `db_version`, `db_written_at` (newest table version time). The fingerprint is all None when `lancedb.databases` places the set, and only `db_path` is set for a missing path. `evaluations run` prints the revision and hash at start.

**Per-case result files**: a single-question QA run writes `<data dir>/evaluations/results/<name>.<trace id>.jsonl` (`evaluations/results.py`, `--results DIR`), one row per case: the `CaseOutcome` fields (`case_name`, `key`, `passed`, `cited`, `cited_map`, `aborted`), trace id, answer, judge reason, attributes, task duration. A pydantic-evals `CaseLifecycle` appends rows to `<name>.<run id>.partial.jsonl` as cases finish, and `write_results` removes it. The run id (`new_run_id`) keeps concurrent runs of one name apart and names the file when there is no trace (`notrace-<run id>`). The final file is opened exclusively. `key` is the case-metadata field named by `DatasetSpec.pair_key`, recorded as `pair_key`. Live runs write none. `evaluations pair TREATED BASELINE` (`evaluations/pairing.py`) joins two files on `key` and prints per-arm accuracy, floor, cite rate, mean `cited_map`, aborts and unjudged, exact McNemar on verdicts, the sign test on `cited_map`, and the smallest significant `|b - c|`.

**Telemetry**: `evaluations run` configures Logfire itself and refuses to start when it resolved no token (`logfire.DEFAULT_LOGFIRE_INSTANCE.config.token`, covering `LOGFIRE_TOKEN` and a credentials file; Logfire does not read the token from its file configuration), unless `--no-telemetry`.

**Per-case diagnostics**: runs record `cited_chunk_ids`, `searched_uris`, `n_searches`, `n_search_calls`, `n_rejected_searches`, `n_failed_tools`, `n_executions`, `n_sandbox_search_calls` and `n_requests` as attributes, counted from the message history (`for_run` gives the run a `replace()` copy whose counters the host never sees):
- `n_rejected_searches` is search-budget exhaustion. `n_failed_tools` is any failed call of the capability's tools. A failed `execute_code` is an exhausted execution budget or an error in model-written Python
- `n_searches` counts distinct search *keys*, with every in-code `search()` filed under one `_sandbox` key. `n_sandbox_search_calls` counts those calls (`CodeExecutionEntry.search_calls`), each a `sandbox.search` span
- `n_requests` is the run's request count
- `citation_status` derives `grounded` / `ungrounded` / `missing` via `ledger.citation_status` (an explicit `cite([])` is `ungrounded`)

`evaluations run --filter/-f CLAUSE` restricts every benchmark search and is recorded as `document_filter`. Never compare a filtered run with an unfiltered one.

**Never steer on raw cite rate**: it is confounded by task success. Measure cite rate among *correct* answers, split by whether the case used `execute_code`.

**Eval prompts**: datasets carry no custom system prompts. `DatasetSpec` has no `system_prompt` field. Capability targets use the packaged instructions plus `config.prompts.domain_preamble`.

**Eval-side rules**: don't assert phrases in packaged instructions, test behaviour. `build_experiment_metadata` is additive, except that the `qa_*` model mirrors were removed: `capability_*` records the model that ran, always present on a QA run and absent on a retrieval run. `qa_max_searches`, `qa_max_executions`, `sandbox_code_timeout` and `sandbox_max_output_chars` stay.

**Reasoning knobs on vLLM**: a vLLM server started with `--reasoning-parser` consumes `chat_template_kwargs.enable_thinking` itself. Muse-Glimmer QA blocks set `chat_template_kwargs.reasoning_strength: high` (the mtrag reference config does). A template defaulting it to low silently cuts search calls, with no error. Verify a kwarg by rendering (`/tokenize` with `return_token_strs`) or by measured behaviour, never by HTTP acceptance.

**`HfApi.upload_large_folder`** has no `path_in_repo`: `evaluations upload` stages the database under a tempdir with `os.link` hardlinks. Keep that rather than `upload_folder`, which is not resumable.

## haiku.rag.app (Conversational RAG Application)

The `app/` directory contains a conversational RAG application with pydantic-ai's native AG-UI support and a CopilotKit frontend. User documentation is `docs/apps.md`.

```bash
cd app
docker compose -f docker-compose.dev.yml up -d --build
# Frontend: http://localhost:3000
# Backend: http://localhost:8001
```

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
    │   ├── SessionManager.tsx  # Browser-stored sessions
    │   ├── DbInfo.tsx          # Database info panel
    │   └── DocumentFilter.tsx  # Document filtering for sessions
    └── biome.json          # Biome config for linting/formatting
```

The backend adapts a native `RAGCapability` agent with Pydantic AI's `AGUIAdapter` and emits one final state snapshot. Both compose files bind frontend and backend to 127.0.0.1: there is no authentication, and the frontend proxies every backend route.

- Backend hot reloads automatically
- Frontend: `docker compose -f docker-compose.dev.yml up -d --build frontend`
- Run `biome check` in `app/frontend/` after frontend changes (`biome check --write` to fix)
