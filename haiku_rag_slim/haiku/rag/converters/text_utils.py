"""Shared utilities for text file handling in converters."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

logger = logging.getLogger(__name__)

# Formats docling reads as UTF-8 only. LaTeX is not here: docling resolves its
# `\input` and `\includegraphics` only from a file path, never from a stream.
DOCLING_TEXT_EXTENSIONS = frozenset(
    {".md", ".qmd", ".rmd", ".csv", ".adoc", ".asc", ".asciidoc"}
)


def to_utf8(raw: bytes, source: Path) -> bytes:
    """Return `raw` as UTF-8, detecting its encoding when it is not UTF-8 already.

    UTF-8 input is returned as the same object. Raises `ValueError` when no
    encoding is detected.
    """
    try:
        raw.decode("utf-8")
        return raw
    except UnicodeDecodeError:
        pass

    from charset_normalizer import from_bytes

    match = from_bytes(raw).best()
    if match is None:
        raise ValueError(f"No text encoding detected for {source}")
    logger.warning("%s is not UTF-8, decoding it as %s", source, match.encoding)
    return str(match).encode("utf-8")


def _universal_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def read_text(path: Path) -> str:
    """Read a text file in any detectable encoding, with universal newlines."""
    return _universal_newlines(to_utf8(path.read_bytes(), path).decode("utf-8"))


def transcode(path: Path) -> bytes | None:
    """UTF-8 bytes with universal newlines of a file that is not UTF-8, else None."""
    raw = path.read_bytes()
    utf8 = to_utf8(raw, path)
    if utf8 is raw:
        return None
    return _universal_newlines(utf8.decode("utf-8")).encode("utf-8")


def docling_safe_name(name: str) -> str:
    """Strip leading dots from a filename handed to docling.

    Docling ignores the extension of any name starting with a dot, so a name
    derived from a dotfile (".customrc" -> ".customrc.md") is classified as
    an unknown format and parsed as one unstructured text block. A name that
    is only dots before its extension keeps the extension ("..md" ->
    "document.md").
    """
    path = Path(name)
    stem = path.stem.lstrip(".")
    if stem:
        return stem + path.suffix
    return f"document{path.suffix}" if path.suffix else name


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
        ".plantuml",
        ".pm",
        ".proto",
        ".pu",
        ".puml",
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
        ".plantuml": "plantuml",
        ".pm": "perl",
        ".proto": "protobuf",
        ".pu": "plantuml",
        ".puml": "plantuml",
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
