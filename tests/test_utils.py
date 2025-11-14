from haiku.rag.config import Config
from haiku.rag.converters import get_converter


def test_text_to_docling_document():
    """Test text to DoclingDocument conversion."""
    # Test basic text conversion
    simple_text = "This is a simple text document."
    converter = get_converter(Config)
    doc = converter.convert_text(simple_text)

    # Verify it returns a DoclingDocument
    from docling_core.types.doc.document import DoclingDocument

    assert isinstance(doc, DoclingDocument)

    # Verify the content can be exported back to markdown
    markdown = doc.export_to_markdown()
    assert "This is a simple text document." in markdown


def test_text_to_docling_document_with_custom_name():
    """Test text to DoclingDocument conversion with custom name parameter."""
    code_text = """# Python Code

```python
def hello():
    print("Hello, World!")
    return True
```"""

    converter = get_converter(Config)
    doc = converter.convert_text(code_text, name="hello.md")

    # Verify it's a valid DoclingDocument
    from docling_core.types.doc.document import DoclingDocument

    assert isinstance(doc, DoclingDocument)

    # Verify the content is preserved
    markdown = doc.export_to_markdown()
    assert "def hello():" in markdown
    assert "Hello, World!" in markdown


def test_text_to_docling_document_markdown_content():
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
    doc = converter.convert_text(markdown_text, name="test.md")

    # Verify it's a DoclingDocument
    from docling_core.types.doc.document import DoclingDocument

    assert isinstance(doc, DoclingDocument)

    # Verify the markdown structure is preserved
    result_markdown = doc.export_to_markdown()
    assert "# Test Document" in result_markdown
    assert "List item 1" in result_markdown
    assert "def test():" in result_markdown


def test_text_to_docling_document_empty_content():
    """Test text to DoclingDocument conversion with empty content."""
    converter = get_converter(Config)
    doc = converter.convert_text("")

    # Should still create a valid DoclingDocument
    from docling_core.types.doc.document import DoclingDocument

    assert isinstance(doc, DoclingDocument)

    # Export should work even with empty content
    markdown = doc.export_to_markdown()
    assert isinstance(markdown, str)


def test_text_to_docling_document_unicode_content():
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
    doc = converter.convert_text(unicode_text, name="unicode.md")

    # Verify it's a DoclingDocument
    from docling_core.types.doc.document import DoclingDocument

    assert isinstance(doc, DoclingDocument)

    # Verify unicode content is preserved
    result_markdown = doc.export_to_markdown()
    assert "测试文档" in result_markdown
    assert "¡Hola mundo!" in result_markdown
    assert "🚀" in result_markdown
