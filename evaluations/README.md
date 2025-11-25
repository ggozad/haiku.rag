# Haiku RAG - Evaluations

Internal benchmarking and evaluation scripts for haiku.rag.

This package is not published to PyPI and is only used for development and testing purposes.

## Overview

Contains evaluation scripts for benchmarking RAG performance using datasets like:
- RepliQA
- WiX

## Usage

After installing the package, you can run evaluations using the `evaluations` command:

```bash
# Run evaluations with default settings
evaluations repliqa

# Use a custom config file
evaluations repliqa --config /path/to/haiku.rag.yaml

# Override the database path
evaluations repliqa --db /path/to/custom.lancedb

# Skip database population and run only benchmarks
evaluations repliqa --skip-db

# Limit the number of test cases
evaluations repliqa --limit 100
```

## Database Storage

By default, evaluation databases are stored in the haiku.rag data directory:
- **Linux**: `~/.local/share/haiku.rag/evaluations/dbs/`
- **macOS**: `~/Library/Application Support/haiku.rag/evaluations/dbs/`
- **Windows**: `C:/Users/<USER>/AppData/Roaming/haiku.rag/evaluations/dbs/`

You can override this with the `--db` option.
