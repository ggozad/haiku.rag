# Search and Question Answering

## Search Settings

Configure search behavior and context expansion:

```yaml
search:
  limit: 5                     # Default number of results to return
  context_radius: 0            # DocItems before/after to include for text content
  max_context_items: 10        # Maximum items in expanded context
  max_context_chars: 10000     # Maximum characters in expanded context
```

- **limit**: Default number of search results to return when no limit is specified. Used by CLI, MCP server, QA, and research workflows. Default: 5
- **context_radius**: For text content (paragraphs), includes N DocItems before and after. Set to 0 to disable expansion (default).
- **max_context_items**: Limits how many document items (paragraphs, list items, etc.) can be included in expanded context. Default: 10.
- **max_context_chars**: Hard limit on total characters in expanded content. Default: 10000.

Structural content (tables, code blocks, lists) uses type-aware expansion that automatically includes the complete structure regardless of how it was chunked.

!!! note "Reranking behavior"
    When a reranker is configured, search automatically retrieves 10x the requested limit, then reranks to return the final count. This improves result quality without requiring you to adjust `limit`.

## Question Answering Configuration

Configure the QA workflow:

```yaml
qa:
  model:
    provider: ollama
    name: gpt-oss
    enable_thinking: false
  max_iterations: 2     # Maximum search iterations
  max_concurrency: 1    # Concurrent search operations
```

- **model**: LLM configuration (see [Providers](providers.md#model-settings))
- **max_iterations**: Maximum search iterations (default: 2)
- **max_concurrency**: Number of concurrent search operations (default: 1)

Deep QA mode (`haiku-rag ask --deep`) uses the research graph with a single iteration for quick, focused answers.

## Research Configuration

Configure the multi-agent research workflow:

```yaml
research:
  model:
    provider: ""            # Empty to use qa settings
    name: ""               # Empty to use qa model
    enable_thinking: false
  max_iterations: 3
  max_concurrency: 1
```

- **model**: LLM configuration. Leave provider/model empty to inherit from `qa` (see [Providers](providers.md#model-settings))
- **max_iterations**: Maximum planning/search iterations (default: 3)
- **max_concurrency**: Concurrent search operations (default: 1)

The research workflow uses an iterative feedback loop: the planner proposes one question at a time, sees the answer, then decides whether to continue or synthesize. This continues until the planner marks research as complete or `max_iterations` is reached.

## RLM Configuration

Configure the RLM (Recursive Language Model) agent:

```yaml
rlm:
  model:
    provider: anthropic
    name: claude-sonnet-4-20250514
  code_timeout: 60.0      # Max seconds for code execution
  max_output_chars: 50000 # Truncate output after this many chars
  docker_image: "ghcr.io/ggozad/haiku.rag-slim:latest"
  docker_memory_limit: "512m"
  docker_host: null        # Docker daemon URL (tcp://, ssh://, unix://)
  docker_db_path: null     # Database path on Docker host
```

- **model**: LLM configuration (see [Providers](providers.md#model-settings))
- **code_timeout**: Maximum seconds for each code execution (default: 60)
- **max_output_chars**: Truncate code output after this many characters (default: 50000)
- **docker_image**: Container image for the sandbox (default: `ghcr.io/ggozad/haiku.rag-slim:latest`)
- **docker_memory_limit**: Container memory limit (default: `512m`)
- **docker_host**: URL of a remote Docker daemon. When set, the sandbox runs on the remote host instead of locally. Supports `tcp://`, `ssh://`, and `unix://` schemes.
- **docker_db_path**: Path to the database on the Docker host. Required for remote Docker since volume mounts resolve on the host machine.

See [RLM Agent](../rlm.md) for usage details and remote Docker setup.
