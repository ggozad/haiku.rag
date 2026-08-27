import importlib.util
from unittest.mock import AsyncMock

import pytest
from pydantic_ai.models.openai import OpenAIChatModel

from haiku.rag.config import get_config
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
    converter = get_converter(get_config())
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

    converter = get_converter(get_config())
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

    converter = get_converter(get_config())
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
    converter = get_converter(get_config())
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

    converter = get_converter(get_config())
    doc = await converter.convert_text(unicode_text, name="unicode.md")

    # Verify it's a DoclingDocument
    from docling_core.types.doc.document import DoclingDocument

    assert isinstance(doc, DoclingDocument)

    # Verify unicode content is preserved
    result_markdown = doc.export_to_markdown()
    assert "测试文档" in result_markdown
    assert "¡Hola mundo!" in result_markdown
    assert "🚀" in result_markdown


@pytest.mark.parametrize(
    "kwargs,expected_settings",
    [
        ({"provider": "ollama", "name": "llama3"}, None),
        (
            {"provider": "ollama", "name": "gpt-oss", "enable_thinking": False},
            {"openai_reasoning_effort": "low"},
        ),
        (
            {"provider": "ollama", "name": "gpt-oss", "enable_thinking": True},
            {"openai_reasoning_effort": "high"},
        ),
        (
            {
                "provider": "ollama",
                "name": "llama3",
                "temperature": 0.5,
                "max_tokens": 100,
            },
            {"temperature": 0.5, "max_tokens": 100},
        ),
        ({"provider": "openai", "name": "gpt-4o"}, None),
        (
            {"provider": "openai", "name": "o1", "enable_thinking": True},
            {"openai_reasoning_effort": "high"},
        ),
        (
            {"provider": "openai", "name": "o1", "enable_thinking": False},
            {"openai_reasoning_effort": "low"},
        ),
        (
            {
                "provider": "openai",
                "name": "gpt-4o",
                "enable_thinking": False,
                "temperature": 0.7,
                "max_tokens": 500,
            },
            # gpt-4o is not a reasoning model, so only the common settings land.
            {"temperature": 0.7, "max_tokens": 500},
        ),
    ],
    ids=[
        "ollama",
        "ollama_thinking_off",
        "ollama_thinking_on",
        "ollama_with_settings",
        "openai",
        "openai_reasoning_thinking_on",
        "openai_reasoning_thinking_off",
        "openai_all_settings",
    ],
)
def test_get_model_openai_chat_settings(kwargs, expected_settings):
    """Each ollama/openai configuration maps onto the expected model settings."""
    result = get_model(ModelConfig(**kwargs))

    assert isinstance(result, OpenAIChatModel)
    if expected_settings is None:
        assert result.settings is None
        return
    assert result.settings is not None
    for key, value in expected_settings.items():
        assert result.settings.get(key) == value


def test_get_model_ollama_appends_v1_to_per_model_base_url():
    """Per-model base_url without /v1 should get it appended."""
    model_config = ModelConfig(
        provider="ollama", name="qwen3.6", base_url="http://my-ollama:11434"
    )
    result = get_model(model_config)
    assert isinstance(result, OpenAIChatModel)
    assert str(result.client.base_url).rstrip("/").endswith("/v1")


def test_get_model_ollama_does_not_double_append_v1():
    """If the per-model base_url already ends with /v1, leave it alone."""
    model_config = ModelConfig(
        provider="ollama", name="qwen3.6", base_url="http://my-ollama:11434/v1"
    )
    result = get_model(model_config)
    url = str(result.client.base_url).rstrip("/")
    assert url.endswith("/v1")
    assert not url.endswith("/v1/v1")


def test_get_model_openai_non_reasoning_model_ignores_thinking():
    """Test that non-reasoning OpenAI models don't get reasoning_effort setting."""
    model_config = ModelConfig(
        provider="openai", name="gpt-4o-mini", enable_thinking=False
    )
    result = get_model(model_config)
    assert isinstance(result, OpenAIChatModel)
    # Non-reasoning models should not have reasoning_effort set
    assert result._settings is None


def test_get_model_vllm_model_without_reasoning_profile_sends_no_thinking():
    """A vLLM-served model with no reasoning profile carries no thinking settings.

    Its chat template reads the switch from `chat_template_kwargs`, which only
    `extra_body` can reach, and the endpoint rejects `reasoning_effort`.
    """
    model_config = ModelConfig(
        provider="openai",
        name="Qwen/Qwen3-32B",
        base_url="http://vllm:8000/v1",
        enable_thinking=True,
        temperature=0.2,
    )
    result = get_model(model_config)

    assert isinstance(result, OpenAIChatModel)
    assert result._settings is not None
    assert "thinking" not in result._settings
    assert "openai_reasoning_effort" not in result._settings


def test_get_model_openai_extra_body_forwarded():
    """`extra_body` on ModelConfig is forwarded to ModelSettings.extra_body.

    pydantic-ai's OpenAI model branch reads `model_settings["extra_body"]`
    and passes it verbatim to the OpenAI SDK. Enables vLLM-specific keys
    like `chat_template_kwargs.enable_thinking` without coupling them to
    the high-level `enable_thinking` flag.
    """
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    model_config = ModelConfig(
        provider="openai",
        name="qwen3.6-35b",
        base_url="http://localhost:11430/v1",
        extra_body=extra,
    )
    result = get_model(model_config)
    assert isinstance(result, OpenAIChatModel)
    assert result._settings is not None
    assert result._settings.get("extra_body") == extra


@pytest.mark.asyncio
async def test_get_model_merges_system_messages_for_openai_compatible():
    """Instruction parts from multiple sources (agent preamble, capability
    instructions, dynamic notices) each map to their own system message.
    Strict OpenAI-compatible templates (e.g. Qwen on vLLM) reject more than
    one leading system message, so OpenAI-compatible endpoints merge them;
    plain OpenAI keeps them separate."""
    from pydantic_ai.messages import InstructionPart, ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters

    parts = [
        InstructionPart(content="Base instructions."),
        InstructionPart(content="Limit notice.", dynamic=True),
    ]
    messages = [ModelRequest(parts=[UserPromptPart("hi")])]

    async def mapped_for(model):
        return await model._map_messages(
            messages, ModelRequestParameters(instruction_parts=parts)
        )

    vllm = get_model(
        ModelConfig(provider="openai", name="qwen3.6", base_url="http://vllm:1/v1")
    )
    mapped = await mapped_for(vllm)
    assert [m["role"] for m in mapped] == ["system", "user"]
    assert mapped[0]["content"] == "Base instructions.\n\nLimit notice."

    ollama = get_model(ModelConfig(provider="ollama", name="llama3"))
    assert [m["role"] for m in await mapped_for(ollama)] == ["system", "user"]

    openai_native = get_model(ModelConfig(provider="openai", name="gpt-4o"))
    assert [m["role"] for m in await mapped_for(openai_native)] == [
        "system",
        "system",
        "user",
    ]


def test_get_model_ollama_extra_body_forwarded():
    """`extra_body` is forwarded through the Ollama (openai-compatible) branch too."""
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    model_config = ModelConfig(provider="ollama", name="qwen3", extra_body=extra)
    result = get_model(model_config)
    assert isinstance(result, OpenAIChatModel)
    assert result._settings is not None
    assert result._settings.get("extra_body") == extra


def test_get_model_extra_body_absent_when_unset():
    """No `extra_body` key appears on the settings when the config omits it."""
    model_config = ModelConfig(provider="openai", name="gpt-4o-mini", temperature=0.3)
    result = get_model(model_config)
    assert isinstance(result, OpenAIChatModel)
    # temperature triggers settings construction; extra_body should not be there.
    assert result._settings is not None
    assert "extra_body" not in result._settings


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
    assert result.settings is not None
    assert result.settings.get("thinking") is True


@pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic not installed")
def test_get_model_anthropic_thinking_off_disables_adaptive_models():
    """Adaptive-thinking models think by default, so off must be explicit.

    The unified `thinking=False` omits the request field, which leaves Sonnet
    4.6+ and Opus 4.6+ thinking.
    """
    from pydantic_ai.models.anthropic import AnthropicModel

    model_config = ModelConfig(
        provider="anthropic", name="claude-sonnet-4-6", enable_thinking=False
    )
    result = get_model(model_config)

    assert isinstance(result, AnthropicModel)
    assert result.settings is not None
    assert result.settings.get("anthropic_thinking") == {"type": "disabled"}
    # The explicit disable replaces the unified key rather than joining it.
    assert "thinking" not in result.settings


@pytest.mark.skipif(not HAS_GOOGLE, reason="Google not installed")
def test_get_model_gemini():
    """Test get_model returns GoogleModel for Gemini."""
    from pydantic_ai.models.google import GoogleModel

    model_config = ModelConfig(provider="gemini", name="gemini-2.0-flash-exp")
    result = get_model(model_config)
    assert isinstance(result, GoogleModel)


@pytest.mark.skipif(not HAS_GOOGLE, reason="Google not installed")
@pytest.mark.parametrize("enable_thinking", [True, False])
def test_get_model_gemini_with_thinking(enable_thinking):
    """Test get_model configures thinking for Gemini."""
    from pydantic_ai.models.google import GoogleModel

    model_config = ModelConfig(
        provider="gemini",
        name="gemini-2.0-flash-thinking-exp",
        enable_thinking=enable_thinking,
    )
    result = get_model(model_config)

    assert isinstance(result, GoogleModel)
    assert result.settings is not None
    assert result.settings.get("thinking") == enable_thinking


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
def test_get_model_groq():
    """Test get_model returns GroqModel for Groq."""
    from pydantic_ai.models.groq import GroqModel

    model_config = ModelConfig(provider="groq", name="llama-3.3-70b-versatile")
    result = get_model(model_config)
    assert isinstance(result, GroqModel)


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
@pytest.mark.parametrize("enable_thinking", [True, False])
def test_get_model_groq_with_thinking(enable_thinking):
    """Test get_model configures thinking for Groq."""
    from pydantic_ai.models.groq import GroqModel

    model_config = ModelConfig(
        provider="groq",
        name="llama-3.3-70b-versatile",
        enable_thinking=enable_thinking,
    )
    result = get_model(model_config)

    assert isinstance(result, GroqModel)
    assert result.settings is not None
    assert result.settings.get("thinking") == enable_thinking


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
@pytest.mark.parametrize(
    "name",
    [
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "openai.gpt-oss-120b-1:0",
        "qwen.qwen3-32b-v1:0",
        "meta.llama3-70b-instruct-v1:0",
    ],
    ids=["claude", "gpt_oss", "qwen", "unmapped"],
)
def test_get_model_bedrock_with_thinking(name):
    """Every Bedrock family carries the unified thinking setting."""
    from pydantic_ai.models.bedrock import BedrockConverseModel

    model_config = ModelConfig(
        provider="bedrock",
        name=name,
        enable_thinking=True,
    )
    result = get_model(model_config)

    assert isinstance(result, BedrockConverseModel)
    assert result.settings is not None
    assert result.settings.get("thinking") is True


@pytest.mark.skipif(not HAS_BEDROCK, reason="Bedrock not installed")
@pytest.mark.parametrize(
    "name",
    [
        "anthropic.claude-sonnet-4-6-20260514-v1:0",
        "us.anthropic.claude-sonnet-4-6-20260514-v1:0",
    ],
    ids=["plain", "cross_region"],
)
def test_get_model_bedrock_thinking_off_disables_adaptive_claude(name):
    """Bedrock omits the field for adaptive Claude, which leaves it thinking."""
    from pydantic_ai.models.bedrock import BedrockConverseModel

    model_config = ModelConfig(provider="bedrock", name=name, enable_thinking=False)
    result = get_model(model_config)

    assert isinstance(result, BedrockConverseModel)
    assert result.settings is not None
    assert result.settings.get("bedrock_additional_model_requests_fields") == {
        "thinking": {"type": "disabled"}
    }
    # The explicit disable replaces the unified key rather than joining it.
    assert "thinking" not in result.settings


@pytest.mark.skipif(not HAS_BEDROCK, reason="Bedrock not installed")
def test_get_model_bedrock_thinking_off_leaves_non_claude_families_alone():
    """Only the Anthropic variant takes a `thinking` request field."""
    from pydantic_ai.models.bedrock import BedrockConverseModel

    model_config = ModelConfig(
        provider="bedrock", name="qwen.qwen3-32b-v1:0", enable_thinking=False
    )
    result = get_model(model_config)

    assert isinstance(result, BedrockConverseModel)
    assert result.settings is not None
    assert "bedrock_additional_model_requests_fields" not in result.settings
    assert result.settings.get("thinking") is False


@pytest.mark.skipif(not HAS_BEDROCK, reason="Bedrock not installed")
def test_get_model_bedrock_rejects_mantle_only_model():
    """Proprietary OpenAI models are Bedrock Mantle-only, not served by Converse."""
    from pydantic_ai.exceptions import UserError

    model_config = ModelConfig(provider="bedrock", name="openai.o3-mini-v1:0")

    with pytest.raises(UserError):
        get_model(model_config)


def test_get_model_unknown_provider():
    """Test get_model returns string format for unknown providers."""
    model_config = ModelConfig(provider="mistral", name="mistral-large-latest")
    result = get_model(model_config)
    assert isinstance(result, str)
    assert result == "mistral:mistral-large-latest"


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
    for value in versions.values():
        assert isinstance(value, str)
        assert len(value) > 0


# --- apply_common_settings tests ---


def test_apply_common_settings_no_settings():
    from haiku.rag.config.models import ModelConfig
    from haiku.rag.utils import apply_common_settings

    mc = ModelConfig(provider="openai", name="gpt-4o")
    result = apply_common_settings(None, mc)
    assert result is None


def test_apply_common_settings_temperature():
    from haiku.rag.config.models import ModelConfig
    from haiku.rag.utils import apply_common_settings

    mc = ModelConfig(provider="openai", name="gpt-4o", temperature=0.7)
    result = apply_common_settings(None, mc)
    assert result is not None
    assert result["temperature"] == 0.7


def test_apply_common_settings_max_tokens():
    from haiku.rag.config.models import ModelConfig
    from haiku.rag.utils import apply_common_settings

    mc = ModelConfig(provider="openai", name="gpt-4o", max_tokens=500)
    result = apply_common_settings(None, mc)
    assert result is not None
    assert result["max_tokens"] == 500


def test_apply_common_settings_existing():
    from haiku.rag.config.models import ModelConfig
    from haiku.rag.utils import apply_common_settings

    mc = ModelConfig(provider="openai", name="gpt-4o", temperature=0.5)
    existing = {"some_key": "value"}
    result = apply_common_settings(existing, mc)
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
    from haiku.rag.store.models.citation import Citation
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
    assert "[1] Test Doc" in result
    assert "doc1" not in result
    assert "chunk1" not in result
    assert "test://doc" in result
    assert "p. 1" in result
    assert "Section: Intro" in result
    assert "Some content" in result


def test_format_citations_multiple_pages():
    from haiku.rag.store.models.citation import Citation
    from haiku.rag.utils import format_citations

    citation = Citation(
        document_id="doc1",
        chunk_id="chunk1",
        document_uri="test://doc",
        content="Content",
        page_numbers=[1, 2, 3],
    )
    result = format_citations([citation])
    assert "[1] test://doc" in result
    assert "pp. 1-3" in result
    # No title: the URI stands in, and the document id never leaks.
    assert "doc1" not in result


def test_format_citations_with_index():
    from haiku.rag.store.models.citation import Citation
    from haiku.rag.utils import format_citations

    citation = Citation(
        index=5,
        document_id="doc1",
        chunk_id="chunk1",
        document_uri="test://doc",
        document_title="Test Doc",
        content="Content",
    )
    result = format_citations([citation])
    assert "[5] Test Doc" in result


def test_format_citations_sequential_indices():
    from haiku.rag.store.models.citation import Citation
    from haiku.rag.utils import format_citations

    citations = [
        Citation(
            document_id="doc1",
            chunk_id="chunk1",
            document_uri="test://doc1",
            document_title="First",
            content="Content 1",
        ),
        Citation(
            document_id="doc2",
            chunk_id="chunk2",
            document_uri="test://doc2",
            document_title="Second",
            content="Content 2",
        ),
    ]
    result = format_citations(citations)
    assert "[1] First" in result
    assert "[2] Second" in result


# --- format_citations tests (pictures) ---


def test_format_citations_picture_refs_render_as_markers():
    from haiku.rag.store.models.citation import Citation
    from haiku.rag.utils import format_citations

    citation = Citation(
        document_id="doc1",
        chunk_id="chunk1",
        document_uri="test://doc",
        document_title="Test Doc",
        content="text body",
        picture_refs=["#/pictures/0", "#/pictures/3"],
    )
    result = format_citations([citation])
    assert "[Figure: #/pictures/0]" in result
    assert "[Figure: #/pictures/3]" in result


# --- format_citations_rich tests ---


def _render_rich(renderables: list) -> str:
    from rich.console import Console

    console = Console(record=True, width=200)
    for r in renderables:
        console.print(r)
    return console.export_text()


async def test_format_citations_rich_empty():
    from haiku.rag.utils import format_citations_rich

    assert await format_citations_rich([]) == []


async def test_format_citations_rich_header_and_footer():
    from haiku.rag.store.models.citation import Citation
    from haiku.rag.utils import format_citations_rich

    citation = Citation(
        document_id="doc-uuid-1",
        chunk_id="chunk-uuid-1",
        document_uri="test://doc",
        document_title="Test Doc",
        content="Body",
        page_numbers=[1, 2, 3],
        headings=["Intro", "Background"],
    )
    output = _render_rich(await format_citations_rich([citation]))
    assert "Citations" in output
    assert "[1] Test Doc (test://doc)" in output
    assert "pp. 1-3" in output
    assert "§Background" in output
    assert "doc: doc-uuid-1" in output
    assert "chunk: chunk-uuid-1" in output


async def test_format_citations_rich_names_the_database_when_federating():
    """Across databases, a citation has to say which one it came from."""
    from unittest.mock import AsyncMock

    from haiku.rag.store.models.citation import Citation
    from haiku.rag.utils import format_citations_rich

    citation = Citation(
        document_id="doc-uuid-1",
        chunk_id="chunk-uuid-1",
        document_uri="test://doc",
        document_title="Test Doc",
        content="Body",
        source="papers",
    )
    client = AsyncMock()
    client.covers_multiple = True
    client.source_names = ("papers", "notes")

    output = _render_rich(await format_citations_rich([citation], client))

    assert "papers" in output


async def test_an_unattributable_picture_renders_its_marker(tmp_path):
    """Evidence recorded before databases could be named carries no source, so
    across databases nothing says which holds the picture. One unrenderable
    figure must not cost the answer."""
    from rich.console import Console

    from haiku.rag.store.models.citation import Citation
    from haiku.rag.utils import format_citations_rich

    covering = AsyncMock()
    covering.covers_multiple = True
    citation = Citation(
        document_id="d1",
        chunk_id="c1",
        content="body",
        document_uri="test://doc",
        picture_refs=["#/pictures/0"],
    )

    renderables = await format_citations_rich([citation], covering)

    console = Console(record=True, width=200)
    for renderable in renderables:
        console.print(renderable)
    assert "[Figure: #/pictures/0]" in console.export_text()
    covering.get_picture_bytes.assert_not_awaited()


def test_truncated_marks_what_it_dropped():
    """An unmarked cut reads as the value: a sentence ending "in 1991" becomes
    one ending "in 1"."""
    from haiku.rag.utils import truncated

    sentence = "Station Kestrel sits at 980 metres and was commissioned in 1991."

    assert truncated(sentence, 60) == sentence[:60] + "…"
    assert truncated(sentence, len(sentence)) == sentence
    assert truncated("short", 60) == "short"
    # Trailing space before the mark reads as a gap in the text.
    assert truncated("a bc", 2) == "a…"


async def test_format_citations_rich_omits_the_database_for_one_database():
    """A single database is not worth naming on every citation."""
    from unittest.mock import AsyncMock

    from haiku.rag.store.models.citation import Citation
    from haiku.rag.utils import format_citations_rich

    citation = Citation(
        document_id="doc-uuid-1",
        chunk_id="chunk-uuid-1",
        document_uri="test://doc",
        document_title="Test Doc",
        content="Body",
        source="papers",
    )
    client = AsyncMock()
    client.covers_multiple = False
    client.source_names = ()

    output = _render_rich(await format_citations_rich([citation], client))

    assert "papers" not in output


async def test_format_citations_rich_truncates_long_content():
    from haiku.rag.store.models.citation import Citation
    from haiku.rag.utils import CITATION_PREVIEW_CHARS, format_citations_rich

    citation = Citation(
        document_id="doc1",
        chunk_id="chunk1",
        document_uri="test://doc",
        content="A" * (CITATION_PREVIEW_CHARS + 200),
    )
    output = _render_rich(await format_citations_rich([citation]))
    assert "…" in output
    assert "A" * (CITATION_PREVIEW_CHARS + 1) not in output


async def test_format_citations_rich_picture_marker_without_client():
    from haiku.rag.store.models.citation import Citation
    from haiku.rag.utils import format_citations_rich

    citation = Citation(
        document_id="doc1",
        chunk_id="chunk1",
        document_uri="test://doc",
        content="body",
        picture_refs=["#/pictures/0"],
    )
    output = _render_rich(await format_citations_rich([citation]))
    assert "[Figure: #/pictures/0]" in output


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


# --- parse_model_option tests ---


def test_parse_model_option():
    from haiku.rag.utils import parse_model_option

    result = parse_model_option("anthropic:claude-sonnet-4-20250514")
    assert result.provider == "anthropic"
    assert result.name == "claude-sonnet-4-20250514"

    # Colons in name are preserved
    assert parse_model_option("openai:gpt-4o:latest").name == "gpt-4o:latest"

    for bad in ["just-a-name", ":model", "provider:"]:
        with pytest.raises(ValueError, match="Invalid model format"):
            parse_model_option(bad)


def test_cosine_similarity_identical_vectors():
    from haiku.rag.utils import cosine_similarity

    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


async def test_format_citations_rich_separates_multiple_citations():
    from haiku.rag.store.models.citation import Citation
    from haiku.rag.utils import format_citations_rich

    citations = [
        Citation(
            document_id=f"doc{i}",
            chunk_id=f"chunk{i}",
            document_uri=f"test://doc{i}",
            document_title=f"Doc {i}",
            content=f"Body {i}",
        )
        for i in (1, 2)
    ]

    output = _render_rich(await format_citations_rich(citations))

    assert "[1] Doc 1 (test://doc1)" in output
    assert "[2] Doc 2 (test://doc2)" in output


@pytest.mark.parametrize(
    "stored,renders",
    [
        (None, False),
        (b"not a real image", False),
        ("png", True),
    ],
    ids=["no_bytes", "undecodable_bytes", "valid_png"],
)
async def test_render_picture_handles_stored_bytes(stored, renders):
    from unittest.mock import AsyncMock

    from haiku.rag.utils import _render_picture

    if stored == "png":
        from io import BytesIO

        from PIL import Image as PILImage

        buf = BytesIO()
        PILImage.new("RGB", (4, 4), "red").save(buf, format="PNG")
        stored = buf.getvalue()

    client = AsyncMock()
    client.covers_multiple = False
    client.get_picture_bytes = AsyncMock(return_value=stored)

    result = await _render_picture(client, "doc1", "#/pictures/0")

    if renders:
        from textual_image.renderable import Image as RichImage

        assert isinstance(result, RichImage)
    else:
        assert result is None


async def test_render_picture_without_client_returns_none():
    from haiku.rag.utils import _render_picture

    assert await _render_picture(None, "doc1", "#/pictures/0") is None


def test_get_package_versions_reports_missing_docling(monkeypatch):
    from importlib import metadata as importlib_metadata

    from haiku.rag.utils import get_package_versions

    real_version = importlib_metadata.version

    def fake_version(name):
        if name == "docling":
            raise importlib_metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(importlib_metadata, "version", fake_version)

    assert get_package_versions()["docling"] == "not installed"


def test_get_model_openai_api_key_from_config():
    """A config-supplied api_key reaches the client, so several
    openai-compatible endpoints can each carry their own key."""
    result = get_model(
        ModelConfig(
            provider="openai",
            name="qwen3.6",
            base_url="http://vllm:8000/v1",
            api_key="sk-vendor-a",
        )
    )
    assert result.client.api_key == "sk-vendor-a"


def test_get_model_openai_api_key_without_base_url_overrides_env():
    result = get_model(
        ModelConfig(provider="openai", name="gpt-4o", api_key="sk-vendor-b")
    )
    assert result.client.api_key == "sk-vendor-b"


def test_get_model_ollama_api_key_from_config():
    result = get_model(
        ModelConfig(
            provider="ollama",
            name="gpt-oss",
            base_url="http://remote-ollama:11434/v1",
            api_key="sk-proxy",
        )
    )
    assert result.client.api_key == "sk-proxy"


def test_get_model_api_key_rejected_on_unplumbed_provider():
    """Providers whose client we never build read their own vendor variable;
    an api_key there would be silently dropped."""
    with pytest.raises(ValueError, match="api_key is not supported"):
        get_model(
            ModelConfig(provider="anthropic", name="claude-sonnet-4-5", api_key="sk-x")
        )
