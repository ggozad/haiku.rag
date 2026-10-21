# haiku.rag app

A browser application over haiku.rag: chat over every configured database, the ingester's queue and jobs, an inspector for documents and chunks, and the doctor with a map of the corpus. The [web app docs](https://ggozad.github.io/haiku.rag/apps/) describe setup and use. This file describes how the app is built.

> **Note:** The application has no authentication. The Compose files publish every service's port on `127.0.0.1` only, and the frontend proxies the backend and the ingester. Keep them unreachable from untrusted networks.

```bash
cd app
cp .env.example .env                      # API keys, host paths, the ingester token
cp haiku.rag.yaml.example haiku.rag.yaml  # models, database at /data, ingester source /docs
docker compose up -d --build              # or docker-compose.dev.yml for hot reload
```

The frontend is at `http://localhost:3000`, the backend at `http://localhost:8001` and the ingester at `http://localhost:8765`.

## Services

- **backend**: the `haiku_rag_app` package, a Starlette app. It opens one read-only client over every configured database on first use, and retries a database that failed to open.
- **ingester**: `haiku-ingester serve`, the only writer of LanceDB. The example configuration ingests the documents mounted at `/docs` into `haiku.rag`. With several databases, it writes each one listed under `ingester.databases`, and the ingestion page names each source's database. One process serves them all: if one database fails to open, the ingester starts for none of them, and the backend keeps serving the others.
- **frontend**: Next.js. It proxies the backend at `/api/backend/*`. At `/api/ingester/*` it proxies the ingester, adds `INGESTER_TOKEN` as a bearer token, and forwards only these writes: refresh a source, retry a job or a dead letter, and cancel a job.

The backend and the ingester both read `haiku.rag.yaml`. The example configuration references `${INGESTER_TOKEN}`, so both require it.

## Sessions

The backend stores sessions in a SQLite file. `HAIKU_RAG_APP_SESSIONS` sets the file's path. The Compose files set it to `/sessions/sessions.db`, on the `sessions` volume. The backend builds each run from the stored session and takes only the new user message from the client. It commits a turn when its run completes, and finishes a started commit even if the browser disconnects. When a run stops or fails before the commit, the backend stores nothing of it, and the frontend returns the question to the input once the stored session confirms the turn is absent. The backend uses one worker. During an active run, it rejects another run of the same session, and a rename or deletion of it.

## Doctor and map jobs

The backend reads every chunk vector for each doctor or map job. The vectors occupy at least chunks × dimension × 4 bytes. One job runs at a time across databases, on a thread of its own. The backend keeps job results in memory until another job of the same kind starts for that database, or the backend restarts. For similarity lookups it also keeps each map's document centroids, within one 256 MiB budget across databases. To fit a newer map's centroids, it removes the oldest kept ones until they fit. When one map's centroids exceed the budget alone, the backend keeps none of them. That map stays visible, and its neighbour lookups are unavailable. The doctor refuses a database read at a tag, since it checks the live state.

## Addresses

Every inspector view has an address that reopens it:

| Address | View |
|---------|------|
| `/inspect/{db}` | The database's documents |
| `/inspect/{db}?view=map&doc=` | The database's map, with a document selected |
| `/inspect/{db}/{document}?tab=&offset=&page=&item=` | A document, its tab, the chunk list's offset, the drawn page and the highlighted item |
| `/inspect/{db}/{document}/chunks/{chunk}?view=` | A chunk and its view |
| `/search?q=&db=&type=&limit=&filter=` | A search, one `db` per database searched |
| `/doctor/{db}` | The doctor for a database |
| `/?session={id}` | A chat session |

`list_q`, `list_filter` and `list_offset` carry the document list's search, filter and page on every address of a database. An image search keeps its image and results until the browser reloads.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/stream` | POST | AG-UI chat stream over a stored session |
| `/api/sessions` | GET, POST | List and create sessions |
| `/api/sessions/{id}` | GET, PATCH, DELETE | Read, rename or scope, delete a session |
| `/api/sessions/{id}/export` | GET | A session as Markdown |
| `/api/databases` | GET | Each database with its availability and size |
| `/api/documents` | GET | A page of each database's documents, or of `source`'s, by `limit` and `offset`, matching `q` in a title or URI, with each database's total |
| `/api/pictures` | GET | A stored picture, by `source`, `document_id` and `ref` |
| `/api/visualize/{chunk_ids}` | GET | Page images with chunks drawn, by `source` |
| `/api/inspect/{source}/…` | GET | Documents, items, outline, chunks, pages, pictures, chunk context and similar chunks, info |
| `/api/search` | POST | Search by text or image across databases |
| `/api/doctor/{source}` | GET | The latest doctor run |
| `/api/doctor/{source}/run` | POST | Start a doctor run |
| `/api/doctor/{source}/projection` | GET, POST | The latest corpus map, or start one |
| `/api/doctor/{source}/projection/similarity` | GET | The cosine similarity between each mapped document's centroid and the centroid of document `id`, in map order. `generation` identifies the map. Returns 404 when the map's centroids are not kept, and 409 when `generation` differs from the kept map's |
| `/api/doctor/{source}/duplicates.yaml` | GET | The map's near-duplicate families |
| `/health` | GET | Health, model and databases |

## Development

The backend is a member of the repository's uv workspace. Its tests run from its directory with a 100% coverage gate:

```bash
(cd app/backend && uv run pytest --cov)
(cd app/frontend && pnpm test && pnpm run typecheck && pnpm run check)
```

The development compose file mounts the backend package and reloads it on change. After changing the core package, rebuild the backend image and recreate its container, from the repository root: `docker compose -f app/docker-compose.dev.yml up -d --build backend`.

If `LOGFIRE_TOKEN` is set, model calls are traced to Logfire.
