# Web application

A browser application over haiku.rag, in the `app/` directory of the repository. It has four pages:

- **Chat**: conversational RAG over every configured database, with citations, pictures, visual grounding, image questions and stored sessions.
- **Inspector**: a database's documents, their items, outline, chunks, pages and pictures, what the agent receives for a chunk, search, and database info.
- **Doctor**: `haiku-rag doctor` as a page, and a map of the corpus with its near-duplicate documents.
- **Ingestion**: the ingester's queue, sources, providers and jobs, with cancel, retry and refresh. The header's ingestion chip opens it and shows the ingester's state: unreachable, degraded, dead letters, ingesting, or idle.

Three services run it: a Starlette backend that reads LanceDB, `haiku-ingester serve` as the only writer, and a Next.js frontend. The terminal equivalents are [Chat](chat.md), the [ingester](ingester.md) dashboard, `haiku-rag inspect` and `haiku-rag doctor`.

!!! warning "No authentication"
    The application has no authentication. The compose files bind every service to `127.0.0.1`, and the frontend proxies the backend and the ingester, so don't expose any of them to an untrusted network.

## Quick start

```bash
cd app
cp .env.example .env
cp haiku.rag.yaml.example haiku.rag.yaml
docker compose up -d --build
# with hot reload instead: docker compose -f docker-compose.dev.yml up -d --build
```

The frontend is at `http://localhost:3000`, the backend at `http://localhost:8001` and the ingester at `http://localhost:8765`. The ingester creates the database on its first start and ingests the documents in `DOCS_VOLUME`.

## Configuration

`.env` in the `app/` directory holds the secrets and the host paths:

```bash
# API keys for the providers haiku.rag.yaml uses (none for Ollama)
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key

# Host path of the LanceDB database, mounted at /data
DB_VOLUME=./data/haiku.rag.lancedb

# Folder of documents the ingester ingests, mounted at /docs
DOCS_VOLUME=./docs

# Bearer token of the ingester's API, which the frontend sends for you
INGESTER_TOKEN=change-me

# Optional: Logfire for observability
LOGFIRE_TOKEN=your-logfire-token
```

`haiku.rag.yaml`, copied from `haiku.rag.yaml.example`, places the database at `/data`, configures the models, and gives the ingester its queue, token and source. The backend and the ingester both read it, so `INGESTER_TOKEN` must be set for either to start:

```yaml
lancedb:
  databases:
    haiku.rag: /data

qa:
  model:
    provider: ollama
    name: gpt-oss

embeddings:
  model:
    provider: ollama
    name: qwen3-embedding:4b
    vector_dim: 2560

ingester:
  queue:
    path: /queue/ingester.db
  api:
    auth_token: ${INGESTER_TOKEN}
  sources:
    - type: fs
      id: docs
      root: /docs
```

### Several databases

`lancedb.databases` may name several databases. The chat searches all of them, the document filter narrows a conversation to chosen documents and databases, and the inspector and the doctor work on one database at a time. The filter applies its document ids in every chosen database, so with two copies of one database, choosing a document in one copy also admits it from the other. A search across the set fails while any member cannot be opened: the header lists each database with its state, and an unavailable one names the reason.

The ingester writes one database. Under a configuration naming several, pass the one it writes with `--db` in the `ingester` service's command:

```yaml
command: ["haiku-ingester", "--config", "/app/haiku.rag.yaml", "serve", "--host", "0.0.0.0", "--db", "/data/papers.lancedb"]
```

`--db` takes a path, so an ingester writing to object storage under a configuration naming several databases needs its own `haiku.rag.yaml` naming that one.

## Chat

Answers stream with their tool calls. After each answer come the sources it cited, with their pictures and a view of the cited passage on its pages, the program it ran, and its citation status. Questions can carry images, picked, pasted or dropped, when `qa.model.vision` is set.

Sessions live on the backend, in a SQLite file at `HAIKU_RAG_APP_SESSIONS`, which the compose files place on the `sessions` volume. The backend commits a turn when its run completes, and a commit once started finishes even if the browser leaves. A run stopped, failed or interrupted before that stores nothing and returns the question to the input, since the conversation's question identities count its messages. The backend runs as one worker: it holds a session for the run in flight, and refuses a second run, a rename or a deletion of that session meanwhile.

## Ingestion

The header chip reads the ingester's health every ten seconds while the page is visible. The page reads the ingester's control plane through the frontend every three seconds while it is visible: health, queue counts and throughput, sources, docling-serve endpoints, and active, failed and recently succeeded jobs. Every job is browsable by status, source and URI. Cancelling a job removes it from the queue, and a conversion already running finishes and may still write its document.

## Inspector

Documents are listed by a title-or-URI search and a SQL filter on their attributes. A document shows its items in reading order, coloured by the chunk holding them, its outline with chunk counts, its chunks, its pages with their chunks drawn, and its pictures. A chunk shows its content and provenance, the context the agent receives for it, its place on the page, and the chunks similar to it. Search runs across the databases by text, or by image when the embedder is multimodal, and shows the results as the agent sees them. Info reports what `haiku-rag info` does, with the database's tags and version history, and it is the one view of a database the app cannot open.

Every inspector view has an address that reopens it:

| Address | View |
|---------|------|
| `/inspect/{db}` | The database's documents |
| `/inspect/{db}/{document}?tab=&offset=&page=&item=` | A document, its tab, the chunk list's offset, the drawn page and the highlighted item |
| `/inspect/{db}/{document}/chunks/{chunk}?view=` | A chunk, over its document |
| `/search?q=&db=&type=&limit=&filter=` | A search, one `db` per database searched |

`list_q`, `list_filter` and `list_offset` carry the document list's search, filter and page on every address of a database. An image cannot ride in an address: an image search keeps its image and results until the browser reloads. The doctor's address is `/doctor/{db}`.

## Doctor

The doctor runs on one configured database in the background, with or without provider probes, and reports every check. A database read at a tag is refused: the doctor reads the live state.

The corpus map places each document by its chunks' centroid through t-SNE, so nearby documents are similar but distances between clusters mean little, colours it by near-duplicate family, source or content type, and shades every document by its centroid cosine to a selected one, listing the nearest. The families download as the YAML `haiku-rag doctor --duplicates-out` writes.

A doctor run and a map each read every chunk vector, at least chunks × dimension × 4 bytes of memory, which the page shows before either starts. One job runs at a time across databases, on a thread of its own. Results live in the backend's memory until the next job of the same kind on that database, or a restart.

## API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/stream` | POST | AG-UI chat stream over a stored session |
| `/api/sessions` | GET, POST | List and create sessions |
| `/api/sessions/{id}` | GET, PATCH, DELETE | Read, rename or scope, delete a session |
| `/api/sessions/{id}/export` | GET | A session as Markdown |
| `/api/databases` | GET | Each database with its availability and size |
| `/api/documents` | GET | Documents of every database, or of `source` |
| `/api/pictures` | GET | A stored picture, by `source`, `document_id` and `ref` |
| `/api/visualize/{chunk_ids}` | GET | Page images with chunks drawn, by `source` |
| `/api/inspect/{source}/…` | GET | Documents, items, outline, chunks, pages, pictures, chunk context and similar chunks, info |
| `/api/search` | POST | Search by text or image across databases |
| `/api/doctor/{source}` | GET | The latest doctor run |
| `/api/doctor/{source}/run` | POST | Start a doctor run |
| `/api/doctor/{source}/projection` | GET, POST | The latest corpus map, or start one |
| `/api/doctor/{source}/projection/similarity` | GET | Every document's centroid cosine to `id`, in map order, for the map `generation` names (409 once it is replaced) |
| `/api/doctor/{source}/duplicates.yaml` | GET | The map's near-duplicate families |
| `/health` | GET | Health, model and databases |

## Development

The backend is the `haiku_rag_app` package, a member of the repository's uv workspace. Its tests run from its directory with a 100% coverage gate:

```bash
(cd app/backend && uv run pytest --cov)
(cd app/frontend && pnpm test && pnpm run typecheck && pnpm run check)
```

The development compose file mounts the backend package and reloads it on change. A change to the core package needs the image rebuilt and the container recreated: `docker compose -f app/docker-compose.dev.yml up -d --build backend`.

If `LOGFIRE_TOKEN` is set, model calls are traced to Logfire.
