"""Shared utilities for text file handling in converters."""

import asyncio
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
        ".bash",
        ".c",
        ".clj",
        ".cljs",
        ".cpp",
        ".cs",
        ".css",
        ".dart",
        ".elm",
        ".ex",
        ".exs",
        ".fs",
        ".fsx",
        ".go",
        ".gql",
        ".graphql",
        ".groovy",
        ".h",
        ".hcl",
        ".hpp",
        ".hs",
        ".java",
        ".jl",
        ".js",
        ".json",
        ".kt",
        ".less",
        ".lua",
        ".mdx",
        ".mjs",
        ".ml",
        ".mli",
        ".nim",
        ".nix",
        ".php",
        ".pl",
        ".pm",
        ".proto",
        ".ps1",
        ".py",
        ".r",
        ".rb",
        ".rs",
        ".sass",
        ".scala",
        ".scss",
        ".sh",
        ".sql",
        ".svelte",
        ".swift",
        ".tf",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".vue",
        ".xml",
        ".yaml",
        ".yml",
        ".zig",
    ]

    # Code file extensions with their markdown language identifiers
    code_markdown_identifier: ClassVar[dict[str, str]] = {
        ".astro": "astro",
        ".bash": "bash",
        ".c": "c",
        ".clj": "clojure",
        ".cljs": "clojure",
        ".cpp": "cpp",
        ".cs": "csharp",
        ".css": "css",
        ".dart": "dart",
        ".elm": "elm",
        ".ex": "elixir",
        ".exs": "elixir",
        ".fs": "fsharp",
        ".fsx": "fsharp",
        ".go": "go",
        ".gql": "graphql",
        ".graphql": "graphql",
        ".groovy": "groovy",
        ".h": "c",
        ".hcl": "hcl",
        ".hpp": "cpp",
        ".hs": "haskell",
        ".java": "java",
        ".jl": "julia",
        ".js": "javascript",
        ".json": "json",
        ".kt": "kotlin",
        ".less": "less",
        ".lua": "lua",
        ".mjs": "javascript",
        ".ml": "ocaml",
        ".mli": "ocaml",
        ".nim": "nim",
        ".nix": "nix",
        ".php": "php",
        ".pl": "perl",
        ".pm": "perl",
        ".proto": "protobuf",
        ".ps1": "powershell",
        ".py": "python",
        ".r": "r",
        ".rb": "ruby",
        ".rs": "rust",
        ".sass": "sass",
        ".scala": "scala",
        ".scss": "scss",
        ".sh": "bash",
        ".sql": "sql",
        ".svelte": "svelte",
        ".swift": "swift",
        ".tf": "hcl",
        ".toml": "toml",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".vue": "vue",
        ".xml": "xml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".zig": "zig",
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

    SUPPORTED_FORMATS = ("md", "html", "plain")

    @staticmethod
    def _create_simple_docling_document(text: str, name: str) -> "DoclingDocument":
        """Create a simple DoclingDocument directly from text.

        Used as fallback when docling's format detection fails for plain text
        that doesn't contain markdown syntax.
        """
        from docling_core.types.doc.document import DoclingDocument
        from docling_core.types.doc.labels import DocItemLabel

        doc_name = name.rsplit(".", 1)[0] if "." in name else name
        doc = DoclingDocument(name=doc_name)
        doc.add_text(label=DocItemLabel.TEXT, text=text)
        return doc

    @staticmethod
    def _sync_text_to_docling_document(
        text: str, name: str = "content.md", format: str = "md"
    ) -> "DoclingDocument":
        """Synchronous implementation of text to DoclingDocument conversion."""
        from docling.document_converter import DocumentConverter as DoclingDocConverter
        from docling.exceptions import ConversionError
        from docling_core.types.io import DocumentStream

        if format not in TextFileHandler.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {format}. "
                f"Supported formats: {', '.join(TextFileHandler.SUPPORTED_FORMATS)}"
            )

        # Derive document name from format to tell docling which parser to use
        doc_name = f"content.{format}" if name == "content.md" else name

        # Plain text doesn't need parsing - create document directly
        if format == "plain":
            return TextFileHandler._create_simple_docling_document(text, doc_name)

        bytes_io = BytesIO(text.encode("utf-8"))
        doc_stream = DocumentStream(name=doc_name, stream=bytes_io)
        converter = DoclingDocConverter()
        try:
            result = converter.convert(doc_stream)
            return result.document
        except ConversionError:
            # Docling's format detection fails for plain text without markdown syntax.
            # Fall back to creating a simple document directly.
            return TextFileHandler._create_simple_docling_document(text, doc_name)

    @staticmethod
    async def text_to_docling_document(
        text: str, name: str = "content.md", format: str = "md"
    ) -> "DoclingDocument":
        """Convert text to DoclingDocument using docling's parser.

        Args:
            text: The text content to convert.
            name: The name to use for the document.
            format: The format of the text content ("md", "html", or "plain").
                Defaults to "md". Use "plain" for plain text without parsing.

        Returns:
            DoclingDocument representation of the text.

        Raises:
            ValueError: If the conversion fails or format is unsupported.
        """
        try:
            return await asyncio.to_thread(
                TextFileHandler._sync_text_to_docling_document, text, name, format
            )
        except Exception as e:
            raise ValueError(f"Failed to convert text to DoclingDocument: {e}")
