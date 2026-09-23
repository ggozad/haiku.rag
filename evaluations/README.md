# haiku.rag - Evaluations

Internal benchmarking and evaluation scripts for haiku.rag.

This package is not published to PyPI and is only used for development and testing purposes.

## Overview

Contains evaluation scripts for benchmarking RAG retrieval and QA performance. Available datasets:

- HotpotQA (`hotpotqa`) — multi-hop QA over Wikipedia paragraphs (distractor validation split, 7,405 questions, two gold documents per question)
- MTRAG ClapNQ (`mtrag_clapnq`, `mtrag_clapnq_rewrite`) — IBM's multi-turn RAG benchmark, ClapNQ (Wikipedia) domain: 183,408 passages, 208 retrieval queries with binary qrels, 224 generation tasks. The base key retrieves with the raw last user turn; the `_rewrite` variant uses the human standalone rewrites (both share one database). Retrieval reports Recall@5/@10, nDCG@5/@10, and MAP against IBM's published setup. QA replays each task's reference conversation prefix as message history and answers the final turn; the judge sees the conversation as a transcript, citation MAP is scored only on turns with gold passages, and refusal precision/recall is reported against the answerability labels. Generation scores are internal (our judge and rubric), not comparable with IBM's published generation numbers. The `mtrag_clapnq_live` key replays whole conversations (one case per conversation, `--limit` counts conversations) through a single capability session, carrying the model's own answers and tool history across turns; it reports the same outcomes per turn plus micro (per-turn) and macro (per-conversation) aggregates.
- FRAMES (`frames`) — multi-hop QA (822 questions, 2-23 gold Wikipedia articles per question; 2 of the original 824 questions are excluded because a linked article has been deleted from Wikipedia). The corpus is the union of the 2,521 linked articles, fetched from the Wikipedia REST API at current revision (revision id and fetch date recorded in the article cache) with navigation chrome stripped. There is no official FRAMES evaluation setup; numbers here correspond to the paper's multi-step retrieval setting (fixed corpus, agentic retrieval, judged accuracy) and are not comparable to its closed-book, oracle-prompt, or web-search settings. Answers were authored against ~2024 revisions and may have drifted with article content.
- OpenRAG Bench, three variants:
  - `orb_text` — text embedder (`qwen3-embedding:4b`, 2560-dim) with VLM picture descriptions baked into chunk content at ingest. Use for text-only retrieval/QA against figure-rich corpora.
  - `orb_multimodal` — multimodal embedder (`qwen3-vl-embedding-8b`, 4096-dim) with picture vectors in the same space as text. Use for cross-modal retrieval (text-as-query → figure hits, image-as-query) and vision QA where the figure itself is the answer.
  - `orb_multimodal_nemotron` — multimodal embedder `nvidia/llama-nemotron-embed-vl-1b-v2`.
- T²-RAGBench (`t2_finqa`, `t2_tatdqa`) — financial-report QA scored by numeric match instead of an LLM judge.
- `mtrag_clapnq_live_uncompacted` — the `mtrag_clapnq_live` replay without evidence compaction.

## Usage

After installing the package, you can run evaluations using the `evaluations` command:

```bash
# Run retrieval + QA benchmarks
evaluations run hotpotqa
evaluations run orb_text

# Use a custom config file
evaluations run hotpotqa --config /path/to/haiku.rag.yaml

# Override the database path
evaluations run hotpotqa --db /path/to/custom.lancedb

# Skip database population and run only benchmarks
evaluations run hotpotqa --skip-db

# Skip specific benchmarks
evaluations run hotpotqa --skip-retrieval
evaluations run hotpotqa --skip-qa

# Limit the number of test cases
evaluations run hotpotqa --limit 100

# Restrict every search to a subset of the database
evaluations run orb_text --skip-db --filter "uri LIKE '2407%'"

# Re-run only the QA cases listed in a file, one id per line
evaluations run hotpotqa --skip-db --filter-ids failed.txt

# Only the queries that need image understanding
evaluations run orb_multimodal --multimodal-only

# Name the run; vacuum every N documents while populating (default 100)
evaluations run hotpotqa --name my-arm --vacuum-interval 500
```

### Choosing the capability model

`evaluations run` benchmarks the RAG capability end to end:

```bash
evaluations run hotpotqa
```

The capability runs on `qa.model` and the judge on `evaluations.judge`, both
from the config. A citation retrieval metric (`cited_map`) is computed
alongside QA accuracy from the URIs the capability registered via the `cite` tool.

### Per-case results

A QA run writes one JSON line per case to
`<data dir>/evaluations/results/<name>.<trace id>.jsonl` (`--results DIR`
to place it elsewhere): case name, pairing key, verdict, citation flag,
`cited_map`, abort flag, trace id, answer, judge reason, the per-case attributes
and task duration. Rows are appended to `<name>.<run id>.partial.jsonl` as
each case finishes, so a run that is killed keeps the cases it completed. The
run id keeps concurrent runs of one name apart, and a result file is never
overwritten. Live
conversation runs write no file. `--name` must be a file name: letters, digits,
dot, dash and underscore.

Two result files over the same cases pair with

```bash
evaluations pair <treated>.jsonl <baseline>.jsonl
```

which joins them on each case's pairing key (`DatasetSpec.pair_key`) and prints
accuracy, floor, cite rate, mean `cited_map`, aborts and unjudged for each arm,
the discordant counts with the exact McNemar p-value, the `cited_map` sign test,
and the smallest discordance split the exact test would reject. It refuses a
file with a missing or duplicate key, and two files with no case in common.

A run refuses to start when Logfire finds no token, whether from
`LOGFIRE_TOKEN` or a credentials file. `--no-telemetry` runs without Logfire,
leaving the result file as the only record.

### Debugging runs in Logfire

With `LOGFIRE_TOKEN` set, runs ship spans under `service_name = 'evals'`. The
`debug-evals` skill in `.claude/skills/` turns these into ready-made Logfire
queries (recent runs, per-case pass rate and `cited_map`, failing and slowest
cases) for use from Claude Code.

### Pre-built Databases

Download pre-built evaluation databases from HuggingFace:

```bash
evaluations download hotpotqa
evaluations download all
evaluations download hotpotqa --force
```

Upload databases (maintainer only):

```bash
evaluations upload hotpotqa
evaluations upload all
```

## Database Storage

By default, evaluation databases are stored in the haiku.rag data directory:
- **Linux**: `~/.local/share/haiku.rag/evaluations/dbs/`
- **macOS**: `~/Library/Application Support/haiku.rag/evaluations/dbs/`
- **Windows**: `C:/Users/<USER>/AppData/Roaming/haiku.rag/evaluations/dbs/`

You can override this with the `--db` option.

### Evaluating over Multiple Databases

With [`lancedb.databases`](https://ggozad.github.io/haiku.rag/configuration/storage/#multiple-databases) configured, `evaluations run <dataset> --skip-db` benchmarks the full set. Retrieval, QA, and live conversations preserve the database name on results and citations. A configured set of one follows the same path and retains its name.

Population writes one database, so it runs only without `lancedb.databases`. `--db` beside a configured set is an error:

```bash
evaluations run hotpotqa --db /path/to/one.lancedb   # no lancedb.databases: populate, then benchmark
evaluations run hotpotqa --skip-db                   # lancedb.databases: benchmark the configured set
```
