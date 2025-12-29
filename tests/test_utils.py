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
