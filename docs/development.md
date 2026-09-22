# Development

This guide covers setting up a development environment and running tests. For how to report issues and open pull requests, see [CONTRIBUTING.md](https://github.com/ggozad/haiku.rag/blob/main/CONTRIBUTING.md).

## Setup

Clone the repository, install dependencies and the pre-commit hooks:

```bash
git clone https://github.com/ggozad/haiku.rag.git
cd haiku.rag
uv sync
uv run pre-commit install
```

## Running Tests

```bash
uv run pytest
```

This runs the complete core suite, including deterministic end-to-end tests and
tests marked `integration`. Integration tests skip when their services are not
available. Start them before running the suite:

```bash
docker compose -f tests/docker/docker-compose.yml up -d
```

The evaluations suite is separate: `uv run pytest evaluations/tests -n0`.

### Test Markers

Tests use pytest markers to categorize them:

- `@pytest.mark.integration` - Tests requiring local services (Docling models, etc.) that aren't available in CI
- `@pytest.mark.slow` - Deterministic end-to-end tests run in a separate CI job
- `@pytest.mark.vcr()` - Tests with HTTP call recording

Async tests are detected automatically through pytest-asyncio's auto mode.

CI runs the fast, slow and evaluations suites as separate jobs. Live integration
tests are exercised locally against the services in `tests/docker/`.

## HTTP Recording with VCR

Tests use [pytest-recording](https://github.com/kiwicom/pytest-recording) (VCR.py) to record and replay HTTP calls. This allows tests to run without external services like Ollama or API providers.

### How It Works

1. Tests marked with `@pytest.mark.vcr()` record HTTP interactions to YAML cassettes
2. On subsequent runs, HTTP calls are replayed from cassettes instead of hitting real services
3. Cassettes are committed to the repository so CI can run tests without external dependencies

Docling-serve polling and retry delays are skipped during cassette playback.
Recording and `--disable-recording` runs retain the real delays.

### Recording New Cassettes

When adding a new test that makes HTTP calls:

1. Add the `@pytest.mark.vcr()` decorator to your test
2. Run the test with the required services available and `--record-mode=once`
3. Commit the generated cassette

### Re-recording Cassettes

To update an existing cassette, delete it and re-run the test, or use `--record-mode=rewrite`.

### Running Without Cassettes (Live Mode)

To run tests against real services instead of recorded cassettes:

```bash
uv run pytest --disable-recording
```

## Writing Tests

### Common Fixtures

Available fixtures from `tests/conftest.py`:

- `temp_db_path` - Isolated temporary database
- `temp_yaml_config` - Temporary config file
- `allow_model_requests` - Enables pydantic-ai model calls

### Example: Adding a New Test with VCR

```python
import pytest
from haiku.rag.client import HaikuRAG


@pytest.mark.vcr()
async def test_my_feature(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document("Test content", uri="test://doc")
        assert doc.id is not None
```

### Integration Tests

For tests requiring local services that can't be mocked via VCR:

```python
@pytest.mark.integration
async def test_pdf_visualization(temp_db_path):
    # Test code that needs local PDF processing
    pass
```

Integration tests are skipped in CI. Start the required services and they run as
part of the core suite; they skip when a service is unreachable.

## Linting and Formatting

```bash
uv run ruff check
uv run ruff format
uv run ty check
```

## Mock API Keys

Tests automatically set mock API keys for providers that require them during client initialization. When running with VCR playback, these mock keys are sufficient since no real API calls are made.

Recording reaches the real service, so the recording command needs network
access and the keys that service reads. Name the exact test and pass `-n0`:
a module-wide `--record-mode=rewrite` re-records every cassette in it,
including ones whose service you do not have running.

```bash
# Ollama-backed cassettes need no key, only a running Ollama
uv run pytest tests/embeddings/test_embedder.py::test_ollama_embedder -n0 --record-mode=rewrite

# A keyed provider reads its own variable. Cohere's SDK reads CO_API_KEY
CO_API_KEY=... uv run pytest tests/reranking/test_reranker.py::test_cohere_reranker -n0 --record-mode=rewrite
```
