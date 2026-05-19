# Haiku RAG - Evaluations

Internal benchmarking and evaluation scripts for haiku.rag.

This package is not published to PyPI and is only used for development and testing purposes.

## Overview

Contains evaluation scripts for benchmarking RAG retrieval and QA performance. Available datasets:

- RepliQA (`repliqa`)
- WiX (`wix`)
- HotpotQA (`hotpotqa`)
- OpenRAG Bench, two variants:
  - `orb_text` — text embedder (`qwen3-embedding:4b`, 2560-dim) with VLM picture descriptions baked into chunk content at ingest. Use for text-only retrieval/QA against figure-rich corpora.
  - `orb_multimodal` — multimodal embedder (`qwen3-vl-embedding-8b`, 4096-dim) with picture vectors in the same space as text. Use for cross-modal retrieval (text-as-query → figure hits, image-as-query) and vision QA where the figure itself is the answer.

## Usage

After installing the package, you can run evaluations using the `evaluations` command:

```bash
# Run retrieval + QA benchmarks
evaluations run repliqa
evaluations run wix

# Use a custom config file
evaluations run repliqa --config /path/to/haiku.rag.yaml

# Override the database path
evaluations run repliqa --db /path/to/custom.lancedb

# Skip database population and run only benchmarks
evaluations run repliqa --skip-db

# Skip specific benchmarks
evaluations run repliqa --skip-retrieval
evaluations run repliqa --skip-qa

# Limit the number of test cases
evaluations run repliqa --limit 100
```

### Choosing the target

`evaluations run` benchmarks `--target rag-skill` by default. Use
`--target analysis-skill` to benchmark the analysis skill against the same
datasets and judge:

```bash
evaluations run wix --target rag-skill
evaluations run wix --target analysis-skill --skill-model ollama:gpt-oss
```

`--skill-model "provider:name"` overrides the skill model independently from
the judge (defaults to `qa.model`, or `analysis.model` when set for the
analysis-skill target). A citation retrieval metric (`cited_mrr` / `cited_map`)
is computed alongside QA accuracy from the URIs the skill registered via the
`cite` tool.

### Pre-built Databases

Download pre-built evaluation databases from HuggingFace:

```bash
evaluations download repliqa
evaluations download all
evaluations download repliqa --force
```

Upload databases (maintainer only):

```bash
evaluations upload repliqa
evaluations upload all
```

## Database Storage

By default, evaluation databases are stored in the haiku.rag data directory:
- **Linux**: `~/.local/share/haiku.rag/evaluations/dbs/`
- **macOS**: `~/Library/Application Support/haiku.rag/evaluations/dbs/`
- **Windows**: `C:/Users/<USER>/AppData/Roaming/haiku.rag/evaluations/dbs/`

You can override this with the `--db` option.
