"""Local docling converter implementation."""

from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from haiku.rag.converters.base import DocumentConverter

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


class DoclingLocalConverter(DocumentConverter):
    """Converter that uses local docling for document conversion.

    This converter runs docling locally in-process to convert documents.
    It handles various document formats including PDF, DOCX, HTML, and plain text.
    """

    # Extensions supported by docling
    docling_extensions: ClassVar[list[str]] = [
        ".adoc",
        ".asc",
        ".asciidoc",
        ".bmp",
        ".csv",
        ".docx",
        ".html",
        ".xhtml",
        ".jpeg",
        ".jpg",
        ".md",
        ".pdf",
        ".png",
        ".pptx",
        ".tiff",
        ".xlsx",
        ".xml",
        ".webp",
    ]

    # Plain text extensions that we'll read directly
    text_extensions: ClassVar[list[str]] = [
        ".astro",
        ".c",
        ".cpp",
        ".css",
        ".go",
        ".h",
        ".hpp",
        ".java",
        ".js",
        ".json",
        ".kt",
        ".mdx",
        ".mjs",
        ".php",
        ".py",
        ".rb",
        ".rs",
        ".svelte",
        ".swift",
        ".ts",
        ".tsx",
        ".txt",
        ".vue",
        ".yaml",
        ".yml",
    ]

    # Code file extensions with their markdown language identifiers for syntax highlighting
    code_markdown_identifier: ClassVar[dict[str, str]] = {
        ".astro": "astro",
        ".c": "c",
        ".cpp": "cpp",
        ".css": "css",
        ".go": "go",
        ".h": "c",
        ".hpp": "cpp",
        ".java": "java",
        ".js": "javascript",
        ".json": "json",
        ".kt": "kotlin",
        ".mjs": "javascript",
        ".php": "php",
        ".py": "python",
        ".rb": "ruby",
        ".rs": "rust",
        ".svelte": "svelte",
        ".swift": "swift",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".vue": "vue",
        ".yaml": "yaml",
        ".yml": "yaml",
    }

    @property
    def supported_extensions(self) -> list[str]:
        """Return list of file extensions supported by this converter."""
        return self.docling_extensions + self.text_extensions

    def convert_file(self, path: Path) -> "DoclingDocument":
        """Convert a file to DoclingDocument using local docling.

        Args:
            path: Path to the file to convert.

        Returns:
            DoclingDocument representation of the file.

        Raises:
            ValueError: If the file cannot be converted.
        """
        from docling.document_converter import DocumentConverter as DoclingDocConverter

        try:
            file_extension = path.suffix.lower()

            if file_extension in self.docling_extensions:
                # Use docling for complex document formats
                converter = DoclingDocConverter()
                result = converter.convert(path)
                return result.document
            elif file_extension in self.text_extensions:
                # Read plain text files directly
                content = path.read_text(encoding="utf-8")

                # Wrap code files (but not plain txt) in markdown code blocks for better presentation
                if file_extension in self.code_markdown_identifier:
                    language = self.code_markdown_identifier[file_extension]
                    content = f"```{language}\n{content}\n```"

                # Convert text to DoclingDocument by wrapping as markdown
                return self.convert_text(content, name=f"{path.stem}.md")
            else:
                # Fallback: try to read as text and convert to DoclingDocument
                content = path.read_text(encoding="utf-8")
                return self.convert_text(content, name=f"{path.stem}.md")
        except Exception:
            raise ValueError(f"Failed to parse file: {path}")

    def convert_text(self, text: str, name: str = "content.md") -> "DoclingDocument":
        """Convert text content to DoclingDocument using local docling.

        Args:
            text: The text content to convert.
            name: The name to use for the document (defaults to "content.md").

        Returns:
            DoclingDocument representation of the text.

        Raises:
            ValueError: If the text cannot be converted.
        """
        from docling.document_converter import DocumentConverter as DoclingDocConverter
        from docling_core.types.io import DocumentStream

        try:
            bytes_io = BytesIO(text.encode("utf-8"))
            doc_stream = DocumentStream(name=name, stream=bytes_io)
            converter = DoclingDocConverter()
            result = converter.convert(doc_stream)
            return result.document
        except Exception as e:
            raise ValueError(f"Failed to convert text to DoclingDocument: {e}")