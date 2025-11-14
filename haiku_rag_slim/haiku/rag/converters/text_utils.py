"""Shared utilities for text file handling in converters."""

from io import BytesIO
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


class TextFileHandler:
    """Handles conversion of text files to DoclingDocument format.

    This class provides shared functionality for converting plain text and code files
    to DoclingDocument format, with proper code block wrapping for syntax highlighting.
    """

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

    # Code file extensions with their markdown language identifiers
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

    @staticmethod
    def prepare_text_content(content: str, file_extension: str) -> str:
        """Prepare text content for conversion to DoclingDocument.

        Wraps code files in markdown code blocks with appropriate language identifiers.

        Args:
            content: The text content.
            file_extension: File extension (including dot, e.g., ".py").

        Returns:
            Prepared text content, possibly wrapped in code blocks.
        """
        if file_extension in TextFileHandler.code_markdown_identifier:
            language = TextFileHandler.code_markdown_identifier[file_extension]
            return f"```{language}\n{content}\n```"
        return content

    @staticmethod
    def text_to_docling_document(
        text: str, name: str = "content.md"
    ) -> "DoclingDocument":
        """Convert text to DoclingDocument using docling's markdown parser.

        Args:
            text: The text content to convert.
            name: The name to use for the document.

        Returns:
            DoclingDocument representation of the text.

        Raises:
            ValueError: If the conversion fails.
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
