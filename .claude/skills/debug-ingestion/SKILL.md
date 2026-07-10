---
name: debug-ingestion
description: Debug haiku.rag ingestion in Logfire. Use when asked to look at Logfire for ingestion, find failed or dead ingestion jobs, investigate retries or circuit-breaker events, trace a document through convert/chunk/embed/store, find which docling-serve instance served a request, spot slow conversions, or tell concurrent ingesters apart. Drives the Logfire MCP against the `haiku-ingester` service.
---

# Debug ingestion in Logfire

The ingester ships spans to Logfire under `service_name = 'haiku-ingester'` (or a
custom `OTEL_SERVICE_NAME` if set per process). This skill finds failing jobs,
traces a document through the pipeline, and pinpoints the docling-serve instance
that served a request. Read-only.

## How to query

1. Confirm the current schema with `mcp__logfire__schema_reference` (spans and
   logs share the `records` table).
2. Run SQL with `mcp__logfire__arbitrary_query` (`query` + `age` in minutes, max
   30 days). The same SQL works pasted into Logfire's Explore UI.
3. Read span attributes as JSON: `attributes->>'key'`, cast when needed
   (`(attributes->>'attempt')::int`).
4. Hand back a clickable trace with `mcp__logfire__logfire_link(trace_id)`.
5. For recent exceptions tied to a file, `mcp__logfire__find_exceptions_in_file`
   accepts `client/documents.py`, `ingester/workers/pool.py`, or
   `ingester/pollers/base.py`.

Adjust `service_name` if the operator set `OTEL_SERVICE_NAME` (e.g. per tenant).
Interactive `haiku-rag` ingests emit the same `document.*` spans under the CLI's
service name (or `unknown_service` for older runs), not `haiku-ingester`.

## Vocabulary

Span tree (all scope `haiku.rag`), each level nests under the one above and
shares its `trace_id`:

- Poller: `ingester.poller.sweep` | `ingester.poller.dry_run` |
  `ingester.poller.watch_event`.
  - `attributes->>'source_id'`; watch adds `change`, `uri`.
  - sweep sets `skipped`, `skip_reason` (`pending_work` / `circuit_open`),
    `upsert`, `delete`, `unchanged`, `consecutive_failures`. A failed sweep
    records the exception on the span (`exception_type` / `exception_message`).
- Job: `ingester.job` — `attributes->>'source_id'`, `->>'uri'`, `->>'op'`
  (`UPSERT`/`DELETE`), `(->>'attempt')::int`. `is_exception=true` marks a job
  that raised.
- Document pipeline: `document.fetch` (`bytes`, `content_hash`),
  `document.convert`, `document.chunk` (`chunks_created`), `document.embed`,
  `document.store` (`op` `create`/`update`, `document_id`).
- docling-serve: `docling_serve.request` — `attributes->>'name'` (operation),
  `->>'url'` (instance), `(->>'attempt')::int`. A retry emits a new span with a
  different `url`, so failover shows as sibling spans.

Failures in Logfire are span-level: sweep and job exceptions sit on the span
(`is_exception`, `exception_type`, `exception_message`, `level >= 17`). A job that
raised `PermanentError` was dead-lettered on that attempt; `attempt >= 2` on an
`ingester.job` span is a retry. The worker circuit breaker opening emits a
dedicated event `span_name = 'ingester.worker breaker opened'` (attributes
`source_id`, `threshold`, `cooldown_s`), fired once per closed->open transition.
The ingester's per-job dead/reschedule narration stays on stderr and the queue
(`GET /jobs?status=dead` on the control-plane API), not Logfire.

## Canned queries

Job outcomes by source:

```sql
SELECT attributes->>'source_id' AS source_id,
       attributes->>'op' AS op,
       is_exception,
       count(*) AS n
FROM records
WHERE service_name='haiku-ingester' AND span_name='ingester.job'
GROUP BY 1,2,3
ORDER BY n DESC;
```

Failed jobs with the exception and trace to drill in:

```sql
SELECT attributes->>'uri' AS uri,
       (attributes->>'attempt')::int AS attempt,
       exception_type, exception_message, trace_id
FROM records
WHERE service_name='haiku-ingester' AND span_name='ingester.job'
  AND is_exception=true
ORDER BY start_timestamp DESC
LIMIT 50;
```

Retried and dead-lettered jobs (a `PermanentError` was dead-lettered on that
attempt; `TransientError` at a high `attempt` was retried):

```sql
SELECT attributes->>'uri' AS uri,
       (attributes->>'attempt')::int AS attempt,
       attributes->>'op' AS op,
       exception_type, trace_id
FROM records
WHERE service_name='haiku-ingester' AND span_name='ingester.job'
  AND (is_exception=true OR (attributes->>'attempt')::int > 1)
ORDER BY start_timestamp DESC
LIMIT 50;
```

Trace one document end-to-end (take `trace_id` from a job above):

```sql
SELECT span_name, duration, is_exception,
       attributes->>'url' AS docling_url,
       attributes->>'op' AS op
FROM records
WHERE trace_id='<TRACE_ID>'
ORDER BY start_timestamp;
```

Worker circuit-breaker trips (a source paused after consecutive transient
failures):

```sql
SELECT start_timestamp,
       attributes->>'source_id' AS source_id,
       (attributes->>'cooldown_s')::float AS cooldown_s
FROM records
WHERE service_name='haiku-ingester'
  AND span_name='ingester.worker breaker opened'
ORDER BY start_timestamp DESC
LIMIT 50;
```

docling-serve instance health and failover:

```sql
SELECT attributes->>'url' AS instance,
       (attributes->>'attempt')::int AS attempt,
       is_exception,
       count(*) AS n
FROM records
WHERE service_name='haiku-ingester' AND span_name='docling_serve.request'
GROUP BY 1,2,3
ORDER BY n DESC;
```

Slowest pipeline stages:

```sql
SELECT span_name,
       attributes->>'uri' AS uri,
       duration, trace_id
FROM records
WHERE service_name='haiku-ingester'
  AND span_name IN ('document.convert','docling_serve.request','document.embed','document.chunk')
ORDER BY duration DESC
LIMIT 20;
```

Per-source sweep summary:

```sql
SELECT attributes->>'source_id' AS source_id,
       sum((attributes->>'upsert')::int) AS upserts,
       sum((attributes->>'delete')::int) AS deletes,
       sum((attributes->>'unchanged')::int) AS unchanged,
       max(attributes->>'skip_reason') AS last_skip_reason
FROM records
WHERE service_name='haiku-ingester' AND span_name='ingester.poller.sweep'
GROUP BY 1;
```

Tell concurrent ingesters apart (set distinct `OTEL_SERVICE_NAME` per process; a
single host still separates by `process_pid` / `service_instance_id`):

```sql
SELECT service_name, service_instance_id, process_pid, count(*) AS n
FROM records
WHERE span_name='ingester.job'
GROUP BY 1,2,3
ORDER BY n DESC;
```

## Workflow

1. Job outcomes by source shows where failures cluster.
2. Failed jobs surfaces the exception and each failing `trace_id`.
3. Trace one document end-to-end to see which stage failed and, for a docling
   source, which `docling_url` served it (and whether it failed over).
4. Retried/dead-lettered jobs show what the queue kept struggling with; a
   `ingester.worker breaker opened` event flags a source the pool paused. The
   per-job dead/reschedule narration stays in the ingester console and the queue
   API, not Logfire.
5. `logfire_link(trace_id)` for a document lets the user expand the full tree.

## When a query returns nothing

Span names or attributes may have changed. Probe:

```sql
SELECT DISTINCT span_name
FROM records WHERE service_name='haiku-ingester'
ORDER BY 1;
```
