from haiku.rag.config import get_config
from haiku.rag.converters import get_converter


async def test_code_file_wrapped_in_code_block(tmp_path):
    """Test that code files are wrapped in markdown code blocks."""
    python_code = '''def hello_world():
    print("Hello, World!")
    return "success"'''

    temp_path = tmp_path / "snippet.py"
    temp_path.write_text(python_code, encoding="utf-8")

    converter = get_converter(get_config())
    document = await converter.convert_file(temp_path)
    result = document.export_to_markdown()

    assert result.startswith("```\n")
    assert result.endswith("\n```")
    assert "def hello_world():" in result
