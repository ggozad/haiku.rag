from haiku.rag.utils import (
    int_to_semantic_version,
    semantic_version_to_int,
    text_to_docling_document,
)


def test_sqlite_user_version():
    version = "0.1.5"
    assert semantic_version_to_int(version) == 261
    assert int_to_semantic_version(261) == version

    version = "0.0.0"
    assert semantic_version_to_int(version) == 0
    assert int_to_semantic_version(0) == version

    version = "255.255.255"
    assert semantic_version_to_int(version) == 16777215
    assert int_to_semantic_version(16777215) == version


def test_text_to_docling_document():
    """Test the text_to_docling_document utility function."""
    # Test basic text conversion
    simple_text = "This is a simple text document."
    doc = text_to_docling_document(simple_text)

    # Verify it returns a DoclingDocument
    from docling_core.types.doc.document import DoclingDocument

    assert isinstance(doc, DoclingDocument)

    # Verify the content can be exported back to markdown
    markdown = doc.export_to_markdown()
    assert "This is a simple text document." in markdown


def test_text_to_docling_document_with_custom_name():
    """Test text_to_docling_document with custom name parameter."""
    code_text = """# Python Code

```python
def hello():
    print("Hello, World!")
    return True
```"""

    doc = text_to_docling_document(code_text, name="hello.md")

    # Verify it's a valid DoclingDocument
    from docling_core.types.doc.document import DoclingDocument

    assert isinstance(doc, DoclingDocument)

    # Verify the content is preserved
    markdown = doc.export_to_markdown()
    assert "def hello():" in markdown
    assert "Hello, World!" in markdown


def test_text_to_docling_document_markdown_content():
    """Test text_to_docling_document with markdown content."""
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

    doc = text_to_docling_document(markdown_text, name="test.md")

    # Verify it's a DoclingDocument
    from docling_core.types.doc.document import DoclingDocument

    assert isinstance(doc, DoclingDocument)

    # Verify the markdown structure is preserved
    result_markdown = doc.export_to_markdown()
    assert "# Test Document" in result_markdown
    assert "List item 1" in result_markdown
    assert "def test():" in result_markdown


def test_text_to_docling_document_empty_content():
    """Test text_to_docling_document with empty content."""
    doc = text_to_docling_document("")

    # Should still create a valid DoclingDocument
    from docling_core.types.doc.document import DoclingDocument

    assert isinstance(doc, DoclingDocument)

    # Export should work even with empty content
    markdown = doc.export_to_markdown()
    assert isinstance(markdown, str)


def test_text_to_docling_document_unicode_content():
    """Test text_to_docling_document with unicode content."""
    unicode_text = """# 测试文档

这是一个包含中文的测试文档。

## Código en Español
```javascript
function saludar() {
    return "¡Hola mundo!";
}
```

Emoji test: 🚀 ✅ 📝"""

    doc = text_to_docling_document(unicode_text, name="unicode.md")

    # Verify it's a DoclingDocument
    from docling_core.types.doc.document import DoclingDocument

    assert isinstance(doc, DoclingDocument)

    # Verify unicode content is preserved
    result_markdown = doc.export_to_markdown()
    assert "测试文档" in result_markdown
    assert "¡Hola mundo!" in result_markdown
    assert "🚀" in result_markdown
