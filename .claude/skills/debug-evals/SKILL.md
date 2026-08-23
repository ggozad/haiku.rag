---
name: debug-evals
description: Debug haiku.rag evaluation runs in Logfire. Use when asked to look at Logfire for an eval run, find failing or low-scoring eval cases, compare runs, check citation quality (cited_map) or judge pass rate (answer_equivalent), or explain why an eval case failed. Drives the Logfire MCP against the `evals` service, or the Logfire HTTP query API when the MCP is not loaded. Also covers monitoring a run that is still in flight.
---

# Debug eval runs in Logfire

Eval runs (`evaluations/`) ship spans to Logfire under `service_name = 'evals'`.
This skill finds a run, surfaces its metrics and failures, and drills into a
single case. Read-only.

## How to query

1. Confirm the current schema with `mcp__logfire__query_schema_reference` (spans
   and logs share the `records` table).
2. Run SQL with `mcp__logfire__query_run` (`query` + `project: "haiku"` +
   `start_timestamp`/`end_timestamp`, max 14 days). The remote MCP is
   org-scoped, so `project` is required; eval runs land in project `haiku`.
   The same SQL works pasted into Logfire's Explore UI.
3. Read span attributes as JSON: `attributes->>'key'`, nested as
   `attributes->'a'->'b'->>'c'`. Cast when needed: `(...)::float`, `(...)::int`.
4. Hand back a clickable trace with
   `mcp__logfire__project_logfire_link(trace_id, project="haiku")`.

## When the Logfire MCP is not available

The `mcp__logfire__*` tools are not loaded in every session. The HTTP query API is
the fallback and needs no MCP:

```
POST https://logfire-eu.pydantic.dev/v2/query      # EU projects
POST https://logfire-us.pydantic.dev/v2/query      # US projects
Authorization: Bearer <api-key>
Content-Type: application/json
{"sql": "...", "min_timestamp": "2026-08-23T00:00:00Z"}
```

- **`min_timestamp` is mandatory** and silently bounds every result. Too recent a
  value is indistinguishable from "no data".
- Read tokens are being replaced by **API keys** (`pylf_v2_<region>_...`). Same
  `Authorization: Bearer` header; the region is in the prefix.
- Keep the key in `~/.logfire-read-key` (mode 600) and read it from there so it never
  lands in a transcript. Helper next to this skill: `.claude/skills/debug-evals/lf-query.sh "<SQL>" [min_ts]`.
- The API is **project-scoped**. A key for the wrong project authenticates fine and
  returns zero rows — it does not error. Diagnose in this order:
  1. wrong region → `HTTP 401 Invalid read token` on the other host;
  2. wrong project → auth succeeds, `count(*)` over months is 0;
  3. right project → `SELECT service_name, count(*) ... GROUP BY service_name` shows
     `evals`, `haiku-rag`, `haiku-ingester`.
  Eval runs live in project **`haiku`**. There is an empty project named `evals`,
  which is the natural wrong guess.
- **The API caps returned rows and does not say so.** Aggregate server-side
  (`count(*)`, `avg(...)`, `sum(CASE WHEN ...)`) rather than pulling rows and counting
  them locally. A pass rate computed from a clipped page is wrong and looks fine.

### JSON access via the HTTP API

`assertions`, `scores`, `metrics` and `case_name` are **keys inside the `attributes`
column, not columns** — `SELECT assertions` fails with `column not found`. Both of
these work:

```sql
attributes->'assertions'->'answer_equivalent'          -- returns JSON
json_get_bool(attributes,'assertions','answer_equivalent','value')
json_get_float(attributes,'scores','cited_map','value')
json_get_int(attributes,'attributes','n_searches')
json_get_str(attributes,'attributes','citation_status')
```

Prefer the `json_get_*` form inside aggregates — it yields a typed value, so no cast
is needed and `sum(CASE WHEN ...)` behaves.

## Counting cases, not spans

**One exception appears once per span level.** A single failing case emits the same
`exception_type` on `case: {case_name}`, `execute {task}` and `invoke_agent agent`
(and often `chat {model}`), so a raw count over-reports by 3-4x. Always add
`AND span_name='case: {case_name}'` when counting failures. Cross-check that the
number equals the count of unjudged cases.

## Always report floor as well as judged

`assertion_pass_rate` **excludes unjudged cases from the denominator**, so cases that
died produce no verdict and silently inflate the headline. Report both:

- judged rate = passed / (cases - unjudged)
- floor = passed / cases

A run with 4.5% deaths reads ~3pp better than it is. Quote them together, always.

## Vocabulary

A run is one experiment span; its cases are direct children sharing its
`trace_id`.

- Experiment span: `span_name = 'evaluate {name}'` (scope `pydantic-evals`).
  - `attributes->>'name'` — run label (the `--name` arg, or `{dataset}_qa_evaluation` / `{dataset}_retrieval_evaluation`).
  - `attributes->>'dataset_name'` — dataset.
  - `(attributes->>'assertion_pass_rate')::float` — overall judge pass rate (QA runs).
  - `attributes->'logfire.experiment.metadata'->'metadata'` — run config: `target` (`rag-capability`|`analysis-capability`), `qa_model`, `embedder_model`, `chunk_size`, `search_limit`, `rerank_model`, `judge_model`, `qa_max_searches`, etc.
  - `trace_id` — scopes the whole run.
- Case span: `span_name = 'case: {case_name}'` (scope `pydantic-evals`).
  - `message` — `case: <id>`.
  - `attributes->'assertions'->'answer_equivalent'->>'value'` — `'true'`/`'false'` (LLM judge verdict). `->>'reason'` — why.
  - `attributes->'scores'->'cited_map'->>'value'` — citation average precision (0..1).
  - `attributes->'scores'->'number_match'->>'value'` — numeric-answer match (datasets that use it).
  - `duration` — task time in seconds.
- Inside each case the capability under test emits agent spans (scope `pydantic-ai`):
  `execute {task}`, `invoke_agent agent`, `execute_tool {tool_name}`, `chat {model}`.

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
   agent-activity query, then `project_logfire_link(trace_id)` so the user can
   expand that case in the UI.

## When a query returns nothing

Span names or attributes may have changed. Probe:

```sql
SELECT DISTINCT otel_scope_name, span_name
FROM records WHERE service_name='evals'
ORDER BY 1,2;
```

## Monitoring a run that is still in flight

An eval prints nothing until it finishes, so a live run's only progress signal is its
case spans. Everything below works mid-run.

Progress and ETA:

```sql
SELECT count(*) AS cases_done,
       min(start_timestamp) AS first_case,
       max(start_timestamp) AS latest_case,
       avg(duration) AS avg_case_s
FROM records
WHERE service_name='evals' AND span_name='case: {case_name}'
  AND start_timestamp > '<RUN_LAUNCH_TS>';
```

**Cases run serially**, so `ETA_total = total_cases * avg_case_s`. Verify rather than
assume: wall-clock per case (`latest_case - first_case` over `cases_done`) should equal
`avg_case_s`. If it does, concurrency is 1 and the multiplication is valid. Concurrent
requests seen on the model endpoint (`vllm:num_requests_running` > 1) are parallel
searches *within* one case, not parallel cases.

**Never estimate a run's length from a `--limit N` smoke.** Its "avg task time per
case" is a per-case duration on the easiest N cases of a deterministic prefix; the full
set ran 32% slower (72.7s vs 55.2s) on FRAMES. Smokes validate wiring, not wall-clock.

Live headline, behaviour and failure composition in one pass:

```sql
SELECT count(*) AS cases,
       sum(CASE WHEN json_get_bool(attributes,'assertions','answer_equivalent','value')
                THEN 1 ELSE 0 END) AS passed,
       sum(CASE WHEN json_get(attributes,'assertions','answer_equivalent') IS NULL
                THEN 1 ELSE 0 END) AS unjudged,
       avg(json_get_float(attributes,'scores','cited_map','value')) AS cited_map,
       avg(json_get_int(attributes,'attributes','n_requests')) AS req_per_case,
       avg(json_get_int(attributes,'attributes','n_searches')) AS searches,
       avg(json_get_int(attributes,'attributes','n_executions')) AS execs,
       sum(json_get_int(attributes,'attributes','n_rejected_searches')) AS rejected,
       sum(CASE WHEN json_get_str(attributes,'attributes','citation_status')='grounded'
                THEN 1 ELSE 0 END) AS grounded
FROM records
WHERE service_name='evals' AND span_name='case: {case_name}'
  AND start_timestamp > '<RUN_LAUNCH_TS>';
```

Why a case died, counted correctly:

```sql
SELECT sum(CASE WHEN exception_message LIKE '%token limit%' THEN 1 ELSE 0 END) AS token_limit,
       sum(CASE WHEN exception_message NOT LIKE '%token limit%' THEN 1 ELSE 0 END) AS other,
       count(*) AS dead_cases
FROM records
WHERE service_name='evals' AND is_exception
  AND span_name='case: {case_name}'
  AND start_timestamp > '<RUN_LAUNCH_TS>';
```

`ToolFailedError` is mostly **not** a defect — an exhausted search or code budget
reports failure to the model on purpose. `UnexpectedModelBehavior` is the one that
kills a case.

### The per-case diagnostic attributes

`attributes->'attributes'` on a case span carries what the capability actually did:
`n_requests`, `n_searches`, `n_search_calls`, `n_rejected_searches`, `n_failed_tools`,
`n_executions`, `cited_uris`, `cited_chunk_ids`, `searched_uris`, `citation_status`
(`grounded` | `missing` | `ungrounded`). `attributes->'metrics'` carries `requests`,
`input_tokens`, `output_tokens` for the case.

Cite rate comes from `citation_status`, not from `cited_map` — they answer different
questions, and conflating them has produced wrong claims before. And never steer on
raw cite rate: it is confounded by task success, so measure it among *correct* answers.
