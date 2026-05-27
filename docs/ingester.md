# Ingester

The ingester is a long-running service that watches sources for
changes and feeds documents into haiku.rag's LanceDB. It runs as a
separate process (`haiku-ingester serve`), owns its own SQLite job
queue, and exposes a small HTTP control plane for operations.

Use the ingester when:

- you have a corpus you want to keep in sync continuously
- documents arrive over time from filesystem, S3, or HTTP sources
- you want retry + dead-letter behavior, not "fire and forget"

For one-off ingestion, the `haiku-rag add-src` CLI is enough — see
[CLI → Add Documents](cli.md).

## Install

The ingester ships behind an optional extra:

```bash
pip install 'haiku.rag-slim[ingester]'
# or, for the full package:
pip install 'haiku.rag[ingester]'
```

That pulls `fastapi`, `uvicorn`, `aiosqlite`, and the `[s3]` extra.
The production binary is `haiku-ingester`.

## Configure sources

Add an `ingester:` block to your `haiku.rag.yaml`. The minimum is a
single source:

```yaml
ingester:
  sources:
    - type: fs
      id: local-docs
      root: /Users/you/docs
      delete_orphans: true
```

### Filesystem

```yaml
ingester:
  sources:
    - type: fs
      id: local-docs                          # optional; auto-derives from root
      root: /Users/you/docs
      poll_interval_s: 300
      delete_orphans: true
      ignore_patterns: ["**/.git/**", "**/node_modules/**"]
      include_patterns: ["*.md", "*.pdf"]    # optional whitelist
```

Uses `watchfiles` for push events plus a periodic sweep that catches
anything the OS dropped between starts. Patterns follow
[gitignore syntax](https://git-scm.com/docs/gitignore#_pattern_format).

### S3 / object storage

```yaml
ingester:
  sources:
    - type: s3
      id: corp-docs
      uri: s3://my-bucket/incoming/
      poll_interval_s: 300
      delete_orphans: true
      ignore_patterns: ["draft*"]
      include_patterns: ["*.pdf", "*.md"]
      storage_options:
        endpoint: http://seaweed:8333         # omit for AWS default chain
        aws_access_key_id: ${AWS_KEY}
        aws_secret_access_key: ${AWS_SECRET}
        region: us-east-1
        allow_http: "true"
```

ETags are the cheap-skip key. Each sweep lists the prefix, compares
the listed ETag against the document's stored `metadata["source_revision"]`,
and only fetches keys whose ETag has changed. If the bytes turn out to
match the stored MD5 (multipart re-upload landing a new ETag on the
same content), only the revision is refreshed — no re-chunk.

`storage_options` follows the same convention as `lancedb.storage_options` —
the dict is passed straight to obstore (the Rust `object_store` library
LanceDB uses internally), so credentials configured for the LanceDB
backend can be copy-pasted here.

### HTTP

```yaml
ingester:
  sources:
    - type: http
      id: arxiv
      urls:
        - https://arxiv.org/pdf/2301.12345.pdf
      headers:
        Authorization: Bearer ${SOME_TOKEN}
      poll_interval_s: 86400
```

HTTP is pull-based with HEAD-driven change detection. A `410 Gone`
response from a configured URL triggers a delete event; other failure
statuses fall through to UPSERT-with-no-revision so the worker can
GET and decide.

### WebDAV

```yaml
ingester:
  sources:
    - type: webdav
      id: nextcloud
      base_url: https://nextcloud.example.com/remote.php/dav/files/alice/Documents/
      username: alice
      password: ${NEXTCLOUD_APP_PASSWORD}
      ignore_patterns: ["**/Trash/**"]
      poll_interval_s: 600
```

Each sweep issues one `PROPFIND` with `Depth: infinity` against
`base_url` and parses the multistatus response. Files (non-collection
resources) are emitted as UPSERT / UNCHANGED based on the `getetag`
property (falling back to `getlastmodified` if the server omits it);
URIs that were in the previous snapshot but no longer appear under the
collection are emitted as DELETE.

Fetches are plain HTTP GETs — any WebDAV server already supports them.

Bearer-token auth can replace HTTP Basic via the standard `headers` map:

```yaml
    - type: webdav
      id: kdrive
      base_url: https://kdrive.infomaniak.com/app/drive/123/
      headers:
        Authorization: Bearer ${KDRIVE_TOKEN}
```

## Workers and retry

```yaml
ingester:
  workers:
    worker_count: 4
    max_concurrent: 4
    poll_idle_interval_s: 1.0
    claim_timeout_s: 1800
    reaper_interval_s: 60
    shutdown_grace_s: 60            # SIGTERM drains in-flight up to this long
    retry:
      max_attempts: 5
      base_delay_s: 2.0
      max_delay_s: 300.0
      jitter: 0.25                  # ±25%
```

The worker pool runs `worker_count` async workers behind a shared
`max_concurrent` semaphore. Jobs that hit a `TransientError` are
rescheduled with exponential backoff plus jitter, up to `max_attempts`,
then land in the dead-letter queue. `PermanentError` (unsupported
extension, 4xx HTTP except 408/429, etc.) skips retry entirely.

A reaper task resets jobs whose `claimed_at` is older than
`claim_timeout_s` so a crashed worker doesn't strand its job.

**Backpressure.** Each poller skips its periodic sweep when its source
already has queued or claimed jobs in the queue. The unique-index dedup
would coalesce a re-sweep anyway; the skip saves the listing round-trip
(`PROPFIND` / `S3 LIST` / FS walk). FS push events from `watchfiles`
still flow during a skipped sweep, so new files aren't lost.

**Graceful shutdown.** On `SIGINT` / `SIGTERM`, pollers stop immediately
and workers are given `shutdown_grace_s` to finish in-flight jobs. Jobs
still running after the grace window are cancelled — they stay
`claimed` in the queue and are reset by the reaper on the next start
once `claim_timeout_s` elapses.

**Tuning.**

- `claim_timeout_s` must exceed the longest legitimate job duration; a
  shorter value lets the reaper resurrect in-flight jobs.
- `worker_count <= max_concurrent`; extras stall in the semaphore.
- `max_concurrent` should match downstream capacity. docling-serve
  processes one task per instance, so `max_concurrent` above the number
  of `providers.docling_serve.base_url` entries over-subscribes the
  fleet — extra submissions queue inside docling-serve and inflate
  `claimed_at` duration toward `claim_timeout_s`.
- `poll_idle_interval_s`: lower = faster pickup, more SQLite churn.
- `reaper_interval_s`: worst-case post-crash reclaim is
  `claim_timeout_s + reaper_interval_s`.

**Per-source override.** A source can opt out of the global retry
policy:

```yaml
ingester:
  sources:
    - type: http
      id: flaky-api
      urls: [...]
      retry:
        max_attempts: 10
        base_delay_s: 10
```

## Circuit breaker

After N consecutive `discover()` failures, a source's circuit breaker
opens and polling pauses for a cooldown. Other sources keep running.

```yaml
ingester:
  sources:
    - type: http
      id: rate-limited
      urls: [...]
      circuit_breaker:
        failure_threshold: 5
        cooldown_s: 600
```

## Run it

```bash
haiku-ingester serve                          # workers + pollers + API
haiku-ingester serve --no-api                 # workers + pollers only
haiku-ingester serve --db /path.lancedb       # explicit DB
haiku-ingester serve --host 0.0.0.0           # bind API on all interfaces
haiku-ingester serve --port 9000              # override API port
```

`--host` and `--port` are CLI overrides for `ingester.api.host` and
`ingester.api.port` in `haiku.rag.yaml`. Both default to the YAML value
(which itself defaults to `127.0.0.1:8765` — loopback only).

The service blocks until SIGINT or SIGTERM. Shutdown drains the API
server, then pollers, then in-flight workers.

### Single-writer constraint

LanceDB supports exactly one writer + N readers per database URI. Run
exactly one `haiku-ingester serve` against a given LanceDB. Multiple
MCP servers or read-only consumers against the same DB are fine.

## HTTP control plane

By default the ingester exposes a FastAPI control plane on
`127.0.0.1:8765`. Set `ingester.api.auth_token` to require a Bearer
token; without one the API stays open and the service logs a warning.

!!! warning "Non-loopback binds need a token"
    Loopback (`127.0.0.1`) is local-only and safe to leave open. If
    you bind to any other interface (`0.0.0.0`, a LAN IP, behind a
    reverse proxy) **set `auth_token`** — the control plane can
    cancel jobs, retry from the DLQ, and trigger source refreshes.
    The startup warning is your only signal that you forgot.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | browser dashboard (HTML; unauthenticated, the JS attaches the bearer on its own JSON fetches) |
| `GET` | `/health` | liveness + queue counts + live worker/poller counts; `status` is `"ok"` or `"degraded"` |
| `GET` | `/sources` | configured pollers + last-poll time + breaker state + last skip reason |
| `POST` | `/sources/{id}/refresh` | force an out-of-band sweep |
| `GET` | `/jobs` | filtered list (`status`, `source_id`, `uri`, `limit`, `offset`) |
| `GET` | `/jobs/{id}` | one job |
| `POST` | `/jobs/{id}/retry` | reset attempts to 0, status to queued |
| `DELETE` | `/jobs/{id}` | cancel a queued/claimed job |
| `GET` | `/dlq` | dead jobs |
| `POST` | `/dlq/{id}/retry` | resurrect from DLQ |
| `GET` | `/stats` | rolling throughput (5m / 30m / 1h succeeded), worker occupancy, oldest queued age, per-source DLQ + backlog |

OpenAPI docs at `http://localhost:8765/docs`. The dashboard at `/` polls
the JSON endpoints above every few seconds and surfaces the same data
visually — queue depth chips, per-source health with a `queue busy` badge
when sweeps are skipped, throughput counters, active jobs with a Cancel
button, recent failures with a Retry button, and the last-completed
feed.

```yaml
ingester:
  api:
    enabled: true
    host: 127.0.0.1
    port: 8765
    auth_token: secret                        # null → unauthenticated
```

## Operating

### Smoke-test a single URI

`run-once` bypasses the queue and runs a single Job through the
pipeline. Useful for sanity-checking a source before starting the
service.

```bash
haiku-ingester run-once /path/to/test.pdf
haiku-ingester run-once https://example.com/spec.pdf
haiku-ingester run-once s3://my-bucket/key.pdf
```

Exit codes: `0` success, `1` transient error, `2` permanent error.

### The queue

The ingester's SQLite queue lives at
`~/Library/Application Support/haiku.rag/ingester.db` on macOS
(platform user data dir; configurable via `ingester.queue.path`). It's
created automatically by `serve`.

For ops setup you can pre-create it:

```bash
haiku-ingester queue init             # create the DB and schema
haiku-ingester queue migrate          # apply pending schema changes
```

### Logs

The service logs via Python `logging` to stderr through a Rich handler.
A typical run looks like:

```
INFO     Ingester running: 4 worker(s), 1 source(s)
INFO     API listening on 127.0.0.1:8765
INFO     Swept local-docs: 142 upsert, 0 delete, 8 unchanged
INFO     Processing upsert file:///.../a.md (job 5d9a...)
INFO     Job 5d9a... succeeded in 0.34s: file:///.../a.md
```

When `LOGFIRE_TOKEN` is set, spans are also shipped to Logfire.

### Operating against the API

```bash
TOKEN=$INGESTER_TOKEN   # omit -H entirely if no token configured

curl http://localhost:8765/health
curl -H "Authorization: Bearer $TOKEN" http://localhost:8765/sources
curl -H "Authorization: Bearer $TOKEN" 'http://localhost:8765/jobs?status=dead'

# Force a poll now
curl -H "Authorization: Bearer $TOKEN" -X POST \
    http://localhost:8765/sources/local-docs/refresh

# Resurrect a dead job
curl -H "Authorization: Bearer $TOKEN" -X POST \
    http://localhost:8765/jobs/<id>/retry
```
