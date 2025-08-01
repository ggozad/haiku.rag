from pathlib import Path
from typing import ClassVar

from docling.document_converter import DocumentConverter


class FileReader:
    # Extensions supported by docling
    docling_extensions: ClassVar[list[str]] = [
        ".asciidoc",
        ".bmp",
        ".csv",
        ".docx",
        ".html",
        ".xhtml",
        ".jpeg",
        ".jpg",
        ".md",
        ".pdf.png",
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

    extensions: ClassVar[list[str]] = docling_extensions + text_extensions

    @staticmethod
    def parse_file(path: Path) -> str:
        try:
            file_extension = path.suffix.lower()

            if file_extension in FileReader.docling_extensions:
                # Use docling for complex document formats
                converter = DocumentConverter()
                result = converter.convert(path)
                return result.document.export_to_markdown()
            elif file_extension in FileReader.text_extensions:
                # Read plain text files directly
                return path.read_text(encoding="utf-8")
            else:
                # Fallback: try to read as text
                return path.read_text(encoding="utf-8")
        except Exception:
            raise ValueError(f"Failed to parse file: {path}")
