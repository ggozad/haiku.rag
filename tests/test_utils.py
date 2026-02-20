import importlib.util

import pytest
from pydantic_ai.models.openai import OpenAIChatModel

from haiku.rag.config import Config
from haiku.rag.config.models import ModelConfig
from haiku.rag.converters import get_converter
from haiku.rag.utils import get_model

# Check for optional dependencies
HAS_ANTHROPIC = importlib.util.find_spec("anthropic") is not None
HAS_GOOGLE = importlib.util.find_spec("google.genai") is not None
HAS_GROQ = importlib.util.find_spec("groq") is not None
HAS_BEDROCK = importlib.util.find_spec("botocore") is not None


@pytest.mark.asyncio
async def test_text_to_docling_document():
    """Test text to DoclingDocument conversion."""
    # Test basic text conversion
    simple_text = "This is a simple text document."
    converter = get_converter(Config)
    doc = await converter.convert_text(simple_text)

    # Verify it returns a DoclingDocument
    from docling_core.types.doc.document import DoclingDocument

    assert isinstance(doc, DoclingDocument)

    # Verify the content can be exported back to markdown
    markdown = doc.export_to_markdown()
    assert "This is a simple text document." in markdown


@pytest.mark.asyncio
async def test_text_to_docling_document_with_custom_name():
    """Test text to DoclingDocument conversion with custom name parameter."""
    code_text = """# Python Code

```python
def hello():
    print("Hello, World!")
    return True
```"""

    converter = get_converter(Config)
    doc = await converter.convert_text(code_text, name="hello.md")

    # Verify it's a valid DoclingDocument
    from docling_core.types.doc.document import DoclingDocument

    assert isinstance(doc, DoclingDocument)

    # Verify the content is preserved
    markdown = doc.export_to_markdown()
    assert "def hello():" in markdown
    assert "Hello, World!" in markdown


@pytest.mark.asyncio
async def test_text_to_docling_document_markdown_content():
    """Test text to DoclingDocument conversion with markdown content."""
    markdown_text = """# Test Document

This is a test document with:

- List item 1
- List item 2

## Code Example

```python
def test():
    return "Hello"
```

**Bold text** and *italic text*."""

    converter = get_converter(Config)
    doc = await converter.convert_text(markdown_text, name="test.md")

    # Verify it's a DoclingDocument
    from docling_core.types.doc.document import DoclingDocument

    assert isinstance(doc, DoclingDocument)

    # Verify the markdown structure is preserved
    result_markdown = doc.export_to_markdown()
    assert "# Test Document" in result_markdown
    assert "List item 1" in result_markdown
    assert "def test():" in result_markdown


@pytest.mark.asyncio
async def test_text_to_docling_document_empty_content():
    """Test text to DoclingDocument conversion with empty content."""
    converter = get_converter(Config)
    doc = await converter.convert_text("")

    # Should still create a valid DoclingDocument
    from docling_core.types.doc.document import DoclingDocument

    assert isinstance(doc, DoclingDocument)

    # Export should work even with empty content
    markdown = doc.export_to_markdown()
    assert isinstance(markdown, str)


@pytest.mark.asyncio
async def test_text_to_docling_document_unicode_content():
    """Test text to DoclingDocument conversion with unicode content."""
    unicode_text = """# 测试文档

这是一个包含中文的测试文档。

## Código en Español
```javascript
function saludar() {
    return "¡Hola mundo!";
}
```

Emoji test: 🚀 ✅ 📝"""

    converter = get_converter(Config)
    doc = await converter.convert_text(unicode_text, name="unicode.md")

    # Verify it's a DoclingDocument
    from docling_core.types.doc.document import DoclingDocument

    assert isinstance(doc, DoclingDocument)

    # Verify unicode content is preserved
    result_markdown = doc.export_to_markdown()
    assert "测试文档" in result_markdown
    assert "¡Hola mundo!" in result_markdown
    assert "🚀" in result_markdown


def test_get_model_ollama():
    """Test get_model returns OpenAIChatModel for Ollama."""
    model_config = ModelConfig(provider="ollama", name="llama3")
    result = get_model(model_config)
    assert isinstance(result, OpenAIChatModel)


def test_get_model_ollama_without_thinking():
    """Test get_model configures thinking for gpt-oss on Ollama."""
    model_config = ModelConfig(provider="ollama", name="gpt-oss", enable_thinking=False)
    result = get_model(model_config)
    assert isinstance(result, OpenAIChatModel)


def test_get_model_ollama_with_settings():
    """Test get_model applies temperature and max_tokens for Ollama."""
    model_config = ModelConfig(
        provider="ollama", name="llama3", temperature=0.5, max_tokens=100
    )
    result = get_model(model_config)
    assert isinstance(result, OpenAIChatModel)


def test_get_model_openai():
    """Test get_model returns OpenAIChatModel for OpenAI."""
    model_config = ModelConfig(provider="openai", name="gpt-4o")
    result = get_model(model_config)
    assert isinstance(result, OpenAIChatModel)


def test_get_model_openai_with_thinking():
    """Test get_model configures thinking for OpenAI reasoning models."""
    model_config = ModelConfig(provider="openai", name="o1", enable_thinking=True)
    result = get_model(model_config)
    assert isinstance(result, OpenAIChatModel)


def test_get_model_openai_non_reasoning_model_ignores_thinking():
    """Test that non-reasoning OpenAI models don't get reasoning_effort setting."""
    model_config = ModelConfig(
        provider="openai", name="gpt-4o-mini", enable_thinking=False
    )
    result = get_model(model_config)
    assert isinstance(result, OpenAIChatModel)
    # Non-reasoning models should not have reasoning_effort set
    assert result._settings is None


@pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic not installed")
def test_get_model_anthropic():
    """Test get_model returns AnthropicModel for Anthropic."""
    from pydantic_ai.models.anthropic import AnthropicModel

    model_config = ModelConfig(provider="anthropic", name="claude-3-5-sonnet-20241022")
    result = get_model(model_config)
    assert isinstance(result, AnthropicModel)


@pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic not installed")
def test_get_model_anthropic_with_thinking():
    """Test get_model configures thinking for Anthropic."""
    from pydantic_ai.models.anthropic import AnthropicModel

    model_config = ModelConfig(
        provider="anthropic",
        name="claude-3-5-sonnet-20241022",
        enable_thinking=True,
    )
    result = get_model(model_config)
    assert isinstance(result, AnthropicModel)


@pytest.mark.skipif(not HAS_GOOGLE, reason="Google not installed")
def test_get_model_gemini():
    """Test get_model returns GoogleModel for Gemini."""
    from pydantic_ai.models.google import GoogleModel

    model_config = ModelConfig(provider="gemini", name="gemini-2.0-flash-exp")
    result = get_model(model_config)
    assert isinstance(result, GoogleModel)


@pytest.mark.skipif(not HAS_GOOGLE, reason="Google not installed")
def test_get_model_gemini_with_thinking():
    """Test get_model configures thinking for Gemini."""
    from pydantic_ai.models.google import GoogleModel

    model_config = ModelConfig(
        provider="gemini", name="gemini-2.0-flash-thinking-exp", enable_thinking=True
    )
    result = get_model(model_config)
    assert isinstance(result, GoogleModel)


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
def test_get_model_groq():
    """Test get_model returns GroqModel for Groq."""
    from pydantic_ai.models.groq import GroqModel

    model_config = ModelConfig(provider="groq", name="llama-3.3-70b-versatile")
    result = get_model(model_config)
    assert isinstance(result, GroqModel)


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
def test_get_model_groq_with_thinking():
    """Test get_model configures thinking format for Groq."""
    from pydantic_ai.models.groq import GroqModel

    model_config = ModelConfig(
        provider="groq", name="llama-3.3-70b-versatile", enable_thinking=False
    )
    result = get_model(model_config)
    assert isinstance(result, GroqModel)


@pytest.mark.skipif(not HAS_BEDROCK, reason="Bedrock not installed")
def test_get_model_bedrock():
    """Test get_model returns BedrockConverseModel for Bedrock."""
    from pydantic_ai.models.bedrock import BedrockConverseModel

    model_config = ModelConfig(
        provider="bedrock", name="anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
    result = get_model(model_config)
    assert isinstance(result, BedrockConverseModel)


@pytest.mark.skipif(not HAS_BEDROCK, reason="Bedrock not installed")
def test_get_model_bedrock_with_thinking():
    """Test get_model configures thinking for Bedrock Claude models."""
    from pydantic_ai.models.bedrock import BedrockConverseModel

    model_config = ModelConfig(
        provider="bedrock",
        name="anthropic.claude-3-5-sonnet-20241022-v2:0",
        enable_thinking=True,
    )
    result = get_model(model_config)
    assert isinstance(result, BedrockConverseModel)


def test_get_model_unknown_provider():
    """Test get_model returns string format for unknown providers."""
    model_config = ModelConfig(provider="mistral", name="mistral-large-latest")
    result = get_model(model_config)
    assert isinstance(result, str)
    assert result == "mistral:mistral-large-latest"


def test_get_model_with_all_settings():
    """Test get_model applies all settings together."""
    model_config = ModelConfig(
        provider="openai",
        name="gpt-4o",
        enable_thinking=False,
        temperature=0.7,
        max_tokens=500,
    )
    result = get_model(model_config)
    assert isinstance(result, OpenAIChatModel)


def test_get_package_versions():
    """Test get_package_versions returns expected keys."""
    from haiku.rag.utils import get_package_versions

    versions = get_package_versions()

    assert "haiku_rag" in versions
    assert "lancedb" in versions
    assert "docling" in versions
    assert "pydantic_ai" in versions
    assert "docling_document_schema" in versions

    # All should be non-empty strings
    for key, value in versions.items():
        assert isinstance(value, str)
        assert len(value) > 0


# --- parse_datetime tests ---


def test_parse_datetime_iso8601():
    from haiku.rag.utils import parse_datetime

    dt = parse_datetime("2025-01-15T14:30:00")
    assert dt.year == 2025
    assert dt.month == 1
    assert dt.day == 15
    assert dt.hour == 14
    assert dt.minute == 30


def test_parse_datetime_date_only():
    from haiku.rag.utils import parse_datetime

    dt = parse_datetime("2025-01-15")
    assert dt.year == 2025
    assert dt.month == 1
    assert dt.day == 15


def test_parse_datetime_with_timezone():
    from haiku.rag.utils import parse_datetime

    dt = parse_datetime("2025-01-15T14:30:00+00:00")
    assert dt.year == 2025
    assert dt.tzinfo is not None


def test_parse_datetime_invalid():
    from haiku.rag.utils import parse_datetime

    with pytest.raises(ValueError, match="Could not parse datetime"):
        parse_datetime("not-a-date")


# --- to_utc tests ---


def test_to_utc_naive_datetime():
    from datetime import datetime

    from haiku.rag.utils import to_utc

    naive = datetime(2025, 6, 15, 12, 0, 0)
    result = to_utc(naive)
    assert result.tzinfo is not None


def test_to_utc_utc_datetime():
    from datetime import UTC, datetime

    from haiku.rag.utils import to_utc

    utc_dt = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)
    result = to_utc(utc_dt)
    assert result is utc_dt


def test_to_utc_aware_non_utc():
    from datetime import UTC, datetime, timedelta, timezone

    from haiku.rag.utils import to_utc

    eastern = timezone(timedelta(hours=-5))
    aware = datetime(2025, 6, 15, 12, 0, 0, tzinfo=eastern)
    result = to_utc(aware)
    assert result.tzinfo == UTC
    assert result.hour == 17


# --- apply_common_settings tests ---


def test_apply_common_settings_no_settings():
    from haiku.rag.config.models import ModelConfig
    from haiku.rag.utils import apply_common_settings

    mc = ModelConfig(provider="openai", name="gpt-4o")
    result = apply_common_settings(None, dict, mc)
    assert result is None


def test_apply_common_settings_temperature():
    from haiku.rag.config.models import ModelConfig
    from haiku.rag.utils import apply_common_settings

    mc = ModelConfig(provider="openai", name="gpt-4o", temperature=0.7)
    result = apply_common_settings(None, dict, mc)
    assert result is not None
    assert result["temperature"] == 0.7


def test_apply_common_settings_max_tokens():
    from haiku.rag.config.models import ModelConfig
    from haiku.rag.utils import apply_common_settings

    mc = ModelConfig(provider="openai", name="gpt-4o", max_tokens=500)
    result = apply_common_settings(None, dict, mc)
    assert result is not None
    assert result["max_tokens"] == 500


def test_apply_common_settings_existing():
    from haiku.rag.config.models import ModelConfig
    from haiku.rag.utils import apply_common_settings

    mc = ModelConfig(provider="openai", name="gpt-4o", temperature=0.5)
    existing = {"some_key": "value"}
    result = apply_common_settings(existing, dict, mc)
    assert result is not None
    assert result["temperature"] == 0.5
    assert result["some_key"] == "value"


# --- format_bytes tests ---


def test_format_bytes():
    from haiku.rag.utils import format_bytes

    assert format_bytes(0) == "0.0 B"
    assert format_bytes(512) == "512.0 B"
    assert format_bytes(1024) == "1.0 KB"
    assert format_bytes(1048576) == "1.0 MB"
    assert format_bytes(1073741824) == "1.0 GB"
    assert format_bytes(1099511627776) == "1.0 TB"
    assert format_bytes(1125899906842624) == "1.0 PB"


# --- format_citations tests ---


def test_format_citations_empty():
    from haiku.rag.utils import format_citations

    assert format_citations([]) == ""


def test_format_citations_with_citation():
    from haiku.rag.agents.research.models import Citation
    from haiku.rag.utils import format_citations

    citation = Citation(
        document_id="doc1",
        chunk_id="chunk1",
        document_uri="test://doc",
        document_title="Test Doc",
        content="Some content",
        page_numbers=[1],
        headings=["Intro"],
    )
    result = format_citations([citation])
    assert "[doc1:chunk1]" in result
    assert "Test Doc" in result
    assert "p. 1" in result
    assert "Section: Intro" in result
    assert "Some content" in result


def test_format_citations_multiple_pages():
    from haiku.rag.agents.research.models import Citation
    from haiku.rag.utils import format_citations

    citation = Citation(
        document_id="doc1",
        chunk_id="chunk1",
        document_uri="test://doc",
        content="Content",
        page_numbers=[1, 2, 3],
    )
    result = format_citations([citation])
    assert "pp. 1-3" in result


def test_format_citations_no_title():
    from haiku.rag.agents.research.models import Citation
    from haiku.rag.utils import format_citations

    citation = Citation(
        document_id="doc1",
        chunk_id="chunk1",
        document_uri="test://doc",
        content="Content",
    )
    result = format_citations([citation])
    assert "test://doc" in result


# --- format_citations_rich tests ---


def test_format_citations_rich_empty():
    from haiku.rag.utils import format_citations_rich

    assert format_citations_rich([]) == []


def test_format_citations_rich_with_citation():
    from rich.panel import Panel
    from rich.text import Text

    from haiku.rag.agents.research.models import Citation
    from haiku.rag.utils import format_citations_rich

    citation = Citation(
        document_id="doc1",
        chunk_id="chunk1",
        document_uri="test://doc",
        document_title="Test Doc",
        content="Some content",
        page_numbers=[1, 2],
        headings=["Intro"],
    )
    result = format_citations_rich([citation])
    assert len(result) == 2
    assert isinstance(result[0], Text)
    assert isinstance(result[1], Panel)


# --- get_default_data_dir tests ---


def test_get_default_data_dir():
    from pathlib import Path

    from haiku.rag.utils import get_default_data_dir

    result = get_default_data_dir()
    assert isinstance(result, Path)
    assert "haiku.rag" in str(result)


# --- build_prompt tests ---


def test_build_prompt_without_preamble():
    from haiku.rag.config.models import AppConfig
    from haiku.rag.utils import build_prompt

    config = AppConfig()
    result = build_prompt("Base prompt", config)
    assert result == "Base prompt"


def test_build_prompt_with_preamble():
    from haiku.rag.config.models import AppConfig, PromptsConfig
    from haiku.rag.utils import build_prompt

    config = AppConfig(prompts=PromptsConfig(domain_preamble="You are a legal expert."))
    result = build_prompt("Base prompt", config)
    assert result == "You are a legal expert.\n\nBase prompt"


# --- is_up_to_date tests ---


def test_cosine_similarity_zero_norm():
    from haiku.rag.utils import cosine_similarity

    assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0
    assert cosine_similarity([1, 2, 3], [0, 0, 0]) == 0.0
    assert cosine_similarity([0, 0], [0, 0]) == 0.0


@pytest.mark.asyncio
async def test_is_up_to_date(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    import httpx

    from haiku.rag.utils import is_up_to_date

    mock_response = MagicMock()
    mock_response.json.return_value = {"info": {"version": "0.0.1"}}

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    is_current, running, latest = await is_up_to_date()
    assert is_current is True
    assert running >= latest
