from pathlib import Path
from typing import ClassVar

from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument

from haiku.rag.utils import text_to_docling_document


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

    extensions: ClassVar[list[str]] = docling_extensions + text_extensions

    @staticmethod
    def parse_file(path: Path) -> DoclingDocument:
        try:
            file_extension = path.suffix.lower()

            if file_extension in FileReader.docling_extensions:
                # Use docling for complex document formats
                converter = DocumentConverter()
                result = converter.convert(path)
                return result.document
            elif file_extension in FileReader.text_extensions:
                # Read plain text files directly
                content = path.read_text(encoding="utf-8")

                # Wrap code files (but not plain txt) in markdown code blocks for better presentation
                if file_extension in FileReader.code_markdown_identifier:
                    language = FileReader.code_markdown_identifier[file_extension]
                    content = f"```{language}\n{content}\n```"

                # Convert text to DoclingDocument by wrapping as markdown
                return text_to_docling_document(content, name=f"{path.stem}.md")
            else:
                # Fallback: try to read as text and convert to DoclingDocument
                content = path.read_text(encoding="utf-8")
                return text_to_docling_document(content, name=f"{path.stem}.md")
        except Exception:
            raise ValueError(f"Failed to parse file: {path}")
