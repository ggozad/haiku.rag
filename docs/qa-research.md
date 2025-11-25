# QA and Research Configuration

## Question Answering Configuration

Configure the QA workflow:

```yaml
qa:
  model:
    provider: ollama
    model: gpt-oss
    enable_thinking: false
  max_sub_questions: 3  # Maximum sub-questions for deep QA
  max_iterations: 2     # Maximum search iterations per sub-question
  max_concurrency: 1    # Sub-questions processed in parallel
```

- **model**: LLM configuration (see [Providers](providers.md#model-settings))
- **max_sub_questions**: For deep QA mode, maximum number of sub-questions to generate (default: 3)
- **max_iterations**: Maximum search/evaluate cycles per sub-question (default: 2)
- **max_concurrency**: Number of sub-questions to process in parallel (default: 1)

Deep QA mode (`haiku-rag ask --deep`) decomposes complex questions into sub-questions, processes them in parallel batches, and synthesizes the results.

## Research Configuration

Configure the multi-agent research workflow:

```yaml
research:
  model:
    provider: ""            # Empty to use qa settings
    model: ""               # Empty to use qa model
    enable_thinking: true
  max_iterations: 3
  confidence_threshold: 0.8
  max_concurrency: 1
```

- **model**: LLM configuration. Leave provider/model empty to inherit from `qa` (see [Providers](providers.md#model-settings))
- **max_iterations**: Maximum search/evaluate cycles (default: 3)
- **confidence_threshold**: Stop when confidence score meets/exceeds this (default: 0.8)
- **max_concurrency**: Sub-questions searched in parallel per iteration (default: 1)

The research workflow plans sub-questions, searches in parallel batches, evaluates findings, and iterates until reaching the confidence threshold or max iterations.

## AG-UI Server Configuration

Configure the AG-UI HTTP server for streaming graph execution events:

```yaml
agui:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]
  cors_credentials: true
  cors_methods: ["GET", "POST", "OPTIONS"]
  cors_headers: ["*"]
```

Start the AG-UI server with:

```bash
haiku-rag serve --agui
```

The server exposes:
- `GET /health` - Health check endpoint
- `POST /v1/agent/stream` - Research graph streaming endpoint (Server-Sent Events)

See [Server Mode](server.md) for more details.
