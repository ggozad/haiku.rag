# Haiku RAG - Evaluations

Internal benchmarking and evaluation scripts for haiku.rag.

This package is not published to PyPI and is only used for development and testing purposes.

## Overview

Contains evaluation scripts for benchmarking RAG performance using datasets like:
- RepliQA
- WiX
- HotpotQA

## Commands

### Run Benchmarks

```bash
# Run evaluations with default settings
evaluations run repliqa

# Use a custom config file
evaluations run repliqa --config /path/to/haiku.rag.yaml

# Override the database path
evaluations run repliqa --db /path/to/custom.lancedb

# Skip database population and run only benchmarks
evaluations run repliqa --skip-db

# Limit the number of test cases
evaluations run repliqa --limit 100
```

### Optimize Prompts

Uses DSPy MIPROv2 to tune the QA system prompt:

```bash
# Basic optimization
evaluations optimize repliqa -o optimized_prompt.yaml

# Different intensity levels
evaluations optimize wix --auto medium

# Manual control over trials
evaluations optimize repliqa --auto none --trials 50

# Limit training examples for faster iteration
evaluations optimize hotpotqa --train-limit 100
```

See the [Prompt Optimization guide](https://ggozad.github.io/haiku.rag/optimization/) for guidance on dataset selection and interpreting results.

## Database Storage

By default, evaluation databases are stored in the haiku.rag data directory:
- **Linux**: `~/.local/share/haiku.rag/evaluations/dbs/`
- **macOS**: `~/Library/Application Support/haiku.rag/evaluations/dbs/`
- **Windows**: `C:/Users/<USER>/AppData/Roaming/haiku.rag/evaluations/dbs/`

You can override this with the `--db` option.
