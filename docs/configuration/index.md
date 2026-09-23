# Configuration

haiku.rag reads its configuration from a YAML file. Every setting has a default, so a file sets only what differs.

!!! note
    The embedding model in the configuration must match the one a database was built with. See [Operational constraints](storage.md#operational-constraints) for what a mismatch does and how to reconcile it.

## Getting started

Generate a configuration file with defaults:

```bash
haiku-rag init-config
```

This writes `haiku.rag.yaml` in the current directory with every setting and its default. `haiku-rag settings` prints the configuration in effect.

## Configuration file locations

`haiku.rag` searches for configuration files in this order:

1. Path specified via `--config` flag: `haiku-rag --config /path/to/config.yaml <command>`, or the `HAIKU_RAG_CONFIG_PATH` environment variable, which library use reads too. A path that does not exist is an error.
2. `./haiku.rag.yaml` (current directory)
3. Platform-specific user directory:
    - **Linux**: `~/.local/share/haiku.rag/haiku.rag.yaml`
    - **macOS**: `~/Library/Application Support/haiku.rag/haiku.rag.yaml`
    - **Windows**: `C:/Users/<USER>/AppData/Roaming/haiku.rag/haiku.rag.yaml`

## Environment variables

Any string value can reference an environment variable, so secrets stay out of the file and one config can serve multiple deployments:

```yaml
ingester:
  queue:
    dburi: postgresql+asyncpg://haiku:${POSTGRES_PASSWORD}@db:5432/haiku_rag
```

- `${VAR}` is replaced with the value of `VAR`. If `VAR` is unset or empty, loading fails with an error naming the variable.
- `${VAR:-default}` uses `default` when `VAR` is unset or empty.
- `$$` produces a literal `$`.

Substitution happens after the YAML is parsed, so a value containing `:`, `@`, or `#` fills the string verbatim and never changes the document structure.

`environment` defaults to `production`. In the CLI, any value other than `development` silences Python warnings and Logfire console output.

## Minimal configuration

A minimal configuration file with defaults. These `qa.model` values are the defaults only when the `qa` section is omitted. Once you write a `qa.model` block, each field you leave out takes the `ModelConfig` default: `vision: false`, and no `thinking`, `temperature` or `max_tokens`. The same holds for `processing.title_model` and `processing.conversion_options.picture_description.model`.


```yaml
# haiku.rag.yaml
environment: production

embeddings:
  model:
    provider: ollama
    name: qwen3-embedding:4b
    vector_dim: 2560

qa:
  model:
    provider: ollama
    name: qwen3.8
    thinking: true
```

## Programmatic configuration

When using haiku.rag as a Python library, you can pass configuration directly to the `HaikuRAG` client:

```python
from haiku.rag.config import AppConfig
from haiku.rag.config.models import EmbeddingModelConfig, ModelConfig, QAConfig, EmbeddingsConfig
from haiku.rag.client import HaikuRAG

# Create custom configuration
custom_config = AppConfig(
    qa=QAConfig(
        model=ModelConfig(
            provider="openai",
            name="gpt-4o",
            temperature=0.3
        )
    ),
    embeddings=EmbeddingsConfig(
        model=EmbeddingModelConfig(
            provider="ollama",
            name="qwen3-embedding:4b",
            vector_dim=2560
        )
    ),
    processing={"chunk_size": 512}
)

# Pass configuration to the client
async with HaikuRAG(config=custom_config) as client:
    ...
```

Without `config`, the client uses the configuration loaded from the YAML file, or the defaults. Clients in one process can each take a different configuration.

## Configuration topics

- [Providers](providers.md): models, embedders and rerankers
- [Search and question answering](qa.md): search, context expansion, QA budgets, sandbox limits
- [Document processing](processing.md): conversion and chunking
- [Storage](storage.md): database placement, remote storage, vacuum, vector indexing
- [Multiple databases](multiple-databases.md): searching several databases together
- [Prompts](prompts.md): domain context and the picture-description prompt
- [Ingester](../ingester.md): continuous ingestion sources, workers and queue
