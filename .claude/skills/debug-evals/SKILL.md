---
name: debug-evals
description: Debug haiku.rag evaluation runs in Logfire. Use when asked to look at Logfire for an eval run, find failing or low-scoring eval cases, compare runs, check citation quality (cited_map) or judge pass rate (answer_equivalent), or explain why an eval case failed. Drives the Logfire MCP against the `evals` service.
---

# Debug eval runs in Logfire

Eval runs (`evaluations/`) ship spans to Logfire under `service_name = 'evals'`.
This skill finds a run, surfaces its metrics and failures, and drills into a
single case. Read-only.

## How to query

1. Confirm the current schema with `mcp__logfire__schema_reference` (spans and
   logs share the `records` table).
2. Run SQL with `mcp__logfire__arbitrary_query` (`query` + `age` in minutes, max
   30 days). The same SQL works pasted into Logfire's Explore UI.
3. Read span attributes as JSON: `attributes->>'key'`, nested as
   `attributes->'a'->'b'->>'c'`. Cast when needed: `(...)::float`, `(...)::int`.
4. Hand back a clickable trace with `mcp__logfire__logfire_link(trace_id)`.

## Vocabulary

A run is one experiment span; its cases are direct children sharing its
`trace_id`.

- Experiment span: `span_name = 'evaluate {name}'` (scope `pydantic-evals`).
  - `attributes->>'name'` — run label (the `--name` arg, or `{dataset}_qa_evaluation` / `{dataset}_retrieval_evaluation`).
  - `attributes->>'dataset_name'` — dataset.
  - `(attributes->>'assertion_pass_rate')::float` — overall judge pass rate (QA runs).
  - `attributes->'logfire.experiment.metadata'->'metadata'` — run config: `target` (`rag-skill`|`analysis-skill`), `qa_model`, `embedder_model`, `chunk_size`, `search_limit`, `rerank_model`, `judge_model`, `skill_model`, etc.
  - `trace_id` — scopes the whole run.
- Case span: `span_name = 'case: {case_name}'` (scope `pydantic-evals`).
  - `message` — `case: <id>`.
  - `attributes->'assertions'->'answer_equivalent'->>'value'` — `'true'`/`'false'` (LLM judge verdict). `->>'reason'` — why.
  - `attributes->'scores'->'cited_map'->>'value'` — citation average precision (0..1).
  - `attributes->'scores'->'number_match'->>'value'` — numeric-answer match (datasets that use it).
  - `duration` — task time in seconds.
- Inside each case the skill under test emits agent spans (scope `pydantic-ai`):
  `execute {task}`, `agent run`, `running tool`, `chat {model}`.

The service is `evals` regardless of model, so filter on `service_name = 'evals'`
first. `otel_scope_name` separates the layers (`pydantic-evals` for run/case,
`pydantic-ai` for the agent).

## Canned queries

Recent runs (pick a `trace_id` to drill in):

```sql
SELECT attributes->>'name' AS run,
       attributes->>'dataset_name' AS dataset,
       service_version,
       (attributes->>'assertion_pass_rate')::float AS pass_rate,
       start_timestamp, trace_id
FROM records
WHERE service_name='evals' AND span_name='evaluate {name}'
ORDER BY start_timestamp DESC
LIMIT 20;
```

Run summary (pass rate, mean citation score, mean task time):

```sql
SELECT count(*) AS cases,
       avg(CASE WHEN attributes->'assertions'->'answer_equivalent'->>'value'='true'
                THEN 1.0 ELSE 0.0 END) AS pass_rate,
       avg((attributes->'scores'->'cited_map'->>'value')::float) AS mean_cited_map,
       avg(duration) AS mean_task_seconds
FROM records
WHERE service_name='evals' AND span_name='case: {case_name}'
  AND trace_id='<TRACE_ID>';
```

Failing cases (judge said not equivalent), newest first, with the reason:

```sql
SELECT message AS case_name, duration,
       attributes->'assertions'->'answer_equivalent'->>'reason' AS reason
FROM records
WHERE service_name='evals' AND span_name='case: {case_name}'
  AND trace_id='<TRACE_ID>'
  AND attributes->'assertions'->'answer_equivalent'->>'value'='false'
ORDER BY start_timestamp;
```

Low-citation cases (answer may be right but grounding is weak):

```sql
SELECT message AS case_name,
       (attributes->'scores'->'cited_map'->>'value')::float AS cited_map
FROM records
WHERE service_name='evals' AND span_name='case: {case_name}'
  AND trace_id='<TRACE_ID>'
  AND (attributes->'scores'->'cited_map'->>'value')::float < 0.5
ORDER BY cited_map;
```

Error / null cases in a run (an exception aborted the case):

```sql
SELECT message, span_name, exception_type, exception_message
FROM records
WHERE service_name='evals' AND trace_id='<TRACE_ID>' AND is_exception=true
ORDER BY start_timestamp
LIMIT 50;
```

Slowest cases (task time drives run cost):

```sql
SELECT message AS case_name, duration
FROM records
WHERE service_name='evals' AND span_name='case: {case_name}'
  AND trace_id='<TRACE_ID>'
ORDER BY duration DESC
LIMIT 20;
```

Drill into one case's agent activity (all cases share the run `trace_id`, so
bound by the case's own time window):

```sql
SELECT span_name, message, duration, is_exception
FROM records
WHERE service_name='evals' AND trace_id='<TRACE_ID>'
  AND otel_scope_name='pydantic-ai'
  AND start_timestamp BETWEEN '<CASE_START>' AND '<CASE_END>'
ORDER BY start_timestamp;
```

## Workflow

1. List recent runs, pick the one to inspect by `name` + `start_timestamp`, note
   its `trace_id`.
2. Run the summary query for the headline numbers (pass rate, mean_cited_map,
   mean_task_seconds — always report task time).
3. Pull failing and low-citation cases, read the judge `reason`.
4. To understand one case, take its start/end from the case query and run the
   agent-activity query, then `logfire_link(trace_id)` so the user can expand
   that case in the UI.

## When a query returns nothing

Span names or attributes may have changed. Probe:

```sql
SELECT DISTINCT otel_scope_name, span_name
FROM records WHERE service_name='evals'
ORDER BY 1,2;
```
