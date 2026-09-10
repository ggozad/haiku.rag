@AGENTS.md

The project guide is `AGENTS.md`. This file carries only what applies to Claude Code.

## Debugging with Logfire

Two skills live in `.claude/skills/` and drive the Logfire MCP:
- `debug-evals` — eval runs (`service_name='evals'`): recent runs, per-case pass rate / `cited_map`, failing and slowest cases.
- `debug-ingestion` — ingestion (`service_name='haiku-ingester'`): failed/dead jobs, docling-serve failover, per-source sweeps, worker breaker trips.

Both start from `mcp__logfire__query_schema_reference` and query the `records` table via `mcp__logfire__query_run`. Ingester stdlib logs (dead/reschedule narration) are NOT shipped to Logfire — only spans and the explicit `ingester.worker breaker opened` event are.

## Recording cassettes under the Bash sandbox

The sandbox blocks external hosts (HuggingFace included), so disable it for a recording command. It does NOT block localhost — a test that embeds against local Ollama passes sandboxed, so running sandboxed does not prove a test avoids the embedder.
