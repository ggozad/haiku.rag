# Haiku RAG - Evaluations

Internal benchmarking and evaluation scripts for haiku.rag.

This package is not published to PyPI and is only used for development and testing purposes.

## Overview

Contains evaluation scripts for benchmarking RAG retrieval and QA performance, plus GEPA-based prompt optimization. Available datasets:

- RepliQA
- WiX
- HotpotQA
- OpenRAG Bench

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

### Benchmarking the skills

By default `evaluations run` benchmarks the QA agent. Pass `--target` to
benchmark the RAG or analysis skill instead, against the same datasets and judge:

```bash
evaluations run wix --target rag-skill
evaluations run wix --target analysis-skill --skill-model ollama:gpt-oss
```

`--skill-model "provider:name"` overrides the skill model independently from
the judge (defaults to `qa.model`). For skill targets, a citation retrieval
metric (`cited_mrr` / `cited_map`) is computed alongside QA accuracy from the
URIs the skill registered via the `cite` tool.

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

### Prompt Optimization

Optimize QA system prompts using GEPA (Generalized Evolutionary Prompt Algorithm):

```bash
evaluations optimize wix
evaluations optimize repliqa --limit 40 --num-candidates 30
evaluations optimize wix --output optimized_prompt.txt
```

See [Tuning docs](https://ggozad.github.io/haiku.rag/tuning/#prompt-optimization-gepa) for details on applying results.

## Database Storage

By default, evaluation databases are stored in the haiku.rag data directory:
- **Linux**: `~/.local/share/haiku.rag/evaluations/dbs/`
- **macOS**: `~/Library/Application Support/haiku.rag/evaluations/dbs/`
- **Windows**: `C:/Users/<USER>/AppData/Roaming/haiku.rag/evaluations/dbs/`

You can override this with the `--db` option.
