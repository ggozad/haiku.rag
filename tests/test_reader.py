import tempfile
from pathlib import Path

from haiku.rag.reader import FileReader


def test_code_file_wrapped_in_code_block():
    """Test that code files are wrapped in markdown code blocks."""
    python_code = '''def hello_world():
    print("Hello, World!")
    return "success"'''

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(python_code)
        f.flush()
        temp_path = Path(f.name)

        result = FileReader.parse_file(temp_path)

        assert result.startswith("```python\n")
        assert result.endswith("\n```")
        assert "def hello_world():" in result
