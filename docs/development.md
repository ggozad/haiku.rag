# Development

This guide covers setting up a development environment and running tests.

## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/ggozad/haiku.rag.git
cd haiku.rag
uv sync
```

## Running Tests

```bash
uv run pytest
```

### Test Markers

Tests use pytest markers to categorize them:

- `@pytest.mark.integration` - Tests requiring local services (Docling models, etc.) that aren't available in CI
- `@pytest.mark.asyncio` - Async tests (applied automatically via pytest-asyncio)
- `@pytest.mark.vcr()` - Tests with HTTP call recording

CI runs `pytest -m "not integration"` to skip integration tests.

## HTTP Recording with VCR

Tests use [pytest-recording](https://github.com/kiwicom/pytest-recording) (VCR.py) to record and replay HTTP calls. This allows tests to run without external services like Ollama or API providers.

### How It Works

1. Tests marked with `@pytest.mark.vcr()` record HTTP interactions to YAML cassettes
2. On subsequent runs, HTTP calls are replayed from cassettes instead of hitting real services
3. Cassettes are committed to the repository so CI can run tests without external dependencies

### Recording New Cassettes

When adding a new test that makes HTTP calls:

1. Add the `@pytest.mark.vcr()` decorator to your test
2. Run the test with the required services available (e.g., Ollama running)
3. The cassette is automatically created on first run

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


@pytest.mark.asyncio
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
@pytest.mark.asyncio
async def test_pdf_visualization(temp_db_path):
    # Test code that needs local PDF processing
    pass
```

Integration tests are skipped in CI but run locally when you have the required services.

## Linting and Formatting

```bash
uv run ruff check
uv run ruff format
uv run pyright
```

## Mock API Keys

Tests automatically set mock API keys for providers that require them during client initialization. When running with VCR playback, these mock keys are sufficient since no real API calls are made.

When recording new cassettes, set real API keys via environment variables:

```bash
ANTHROPIC_API_KEY=sk-ant-... uv run pytest tests/test_qa.py::test_qa_anthropic --record-mode=rewrite
```
