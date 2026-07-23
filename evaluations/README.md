# Haiku RAG - Evaluations

Internal benchmarking and evaluation scripts for haiku.rag.

This package is not published to PyPI and is only used for development and testing purposes.

## Overview

Contains evaluation scripts for benchmarking RAG retrieval and QA performance. Available datasets:

- WiX (`wix`)
- HotpotQA (`hotpotqa`) — multi-hop QA over Wikipedia paragraphs (distractor validation split, 7,405 questions, two gold documents per question)
- MTRAG ClapNQ (`mtrag_clapnq`, `mtrag_clapnq_rewrite`) — IBM's multi-turn RAG benchmark, ClapNQ (Wikipedia) domain: 183,408 passages, 208 retrieval queries with binary qrels, 224 generation tasks. The base key retrieves with the raw last user turn; the `_rewrite` variant uses the human standalone rewrites (both share one database). Retrieval reports Recall@5/@10, nDCG@5/@10, and MAP against IBM's published setup. QA replays each task's reference conversation prefix as message history and answers the final turn; the judge sees the conversation as a transcript, citation MAP is scored only on turns with gold passages, and refusal precision/recall is reported against the answerability labels. Generation scores are internal (our judge and rubric), not comparable with IBM's published generation numbers. The `mtrag_clapnq_live` key replays whole conversations (one case per conversation, `--limit` counts conversations) through a single capability session, carrying the model's own answers and tool history across turns; it reports the same outcomes per turn plus micro (per-turn) and macro (per-conversation) aggregates.
- OpenRAG Bench, two variants:
  - `orb_text` — text embedder (`qwen3-embedding:4b`, 2560-dim) with VLM picture descriptions baked into chunk content at ingest. Use for text-only retrieval/QA against figure-rich corpora.
  - `orb_multimodal` — multimodal embedder (`qwen3-vl-embedding-8b`, 4096-dim) with picture vectors in the same space as text. Use for cross-modal retrieval (text-as-query → figure hits, image-as-query) and vision QA where the figure itself is the answer.

## Usage

After installing the package, you can run evaluations using the `evaluations` command:

```bash
# Run retrieval + QA benchmarks
evaluations run wix
evaluations run orb_text

# Use a custom config file
evaluations run wix --config /path/to/haiku.rag.yaml

# Override the database path
evaluations run wix --db /path/to/custom.lancedb

# Skip database population and run only benchmarks
evaluations run wix --skip-db

# Skip specific benchmarks
evaluations run wix --skip-retrieval
evaluations run wix --skip-qa

# Limit the number of test cases
evaluations run wix --limit 100
```

### Choosing the target

`evaluations run` benchmarks `--target rag-capability` by default. Use
`--target analysis-capability` to benchmark the analysis capability against the same
datasets and judge:

```bash
evaluations run wix --target rag-capability
evaluations run wix --target analysis-capability --capability-model ollama:gpt-oss
```

`--capability-model "provider:name"` overrides the capability model independently from
the judge (defaults to `qa.model`, or `analysis.model` when set for the
analysis-capability target). A citation retrieval metric (`cited_map`) is computed
alongside QA accuracy from the URIs the capability registered via the `cite` tool.

### Debugging runs in Logfire

With `LOGFIRE_TOKEN` set, runs ship spans under `service_name = 'evals'`. The
`debug-evals` skill in `.claude/skills/` turns these into ready-made Logfire
queries (recent runs, per-case pass rate and `cited_map`, failing and slowest
cases) for use from Claude Code.

### Pre-built Databases

Download pre-built evaluation databases from HuggingFace:

```bash
evaluations download wix
evaluations download all
evaluations download wix --force
```

Upload databases (maintainer only):

```bash
evaluations upload wix
evaluations upload all
```

## Database Storage

By default, evaluation databases are stored in the haiku.rag data directory:
- **Linux**: `~/.local/share/haiku.rag/evaluations/dbs/`
- **macOS**: `~/Library/Application Support/haiku.rag/evaluations/dbs/`
- **Windows**: `C:/Users/<USER>/AppData/Roaming/haiku.rag/evaluations/dbs/`

You can override this with the `--db` option.
