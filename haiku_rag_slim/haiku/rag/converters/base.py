"""Base class for document converters."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


class DocumentConverter(ABC):
    """Abstract base class for document converters.

    Document converters are responsible for converting various document formats
    (PDF, DOCX, HTML, etc.) into DoclingDocument format for further processing.
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return list of file extensions supported by this converter.

        Returns:
            List of file extensions (including the dot, e.g., [".pdf", ".docx"]).
        """
        pass

    @abstractmethod
    async def convert_file(
        self, path: Path, source_uri: str | None = None
    ) -> "DoclingDocument":
        """Convert a file to DoclingDocument format.

        Args:
            path: Path to the file to convert.
            source_uri: Optional origin URI (e.g. the URL the file was
                downloaded from) used by docling's HTML/Markdown backends to
                resolve relative `<img src="/path">` references. Ignored by
                converters that have no equivalent backend option (notably
                docling-serve).

        Returns:
            DoclingDocument representation of the file.

        Raises:
            ValueError: If the file cannot be converted.
        """
        pass

    SUPPORTED_FORMATS = ("md", "html", "plain")

    @abstractmethod
    async def convert_text(
        self,
        text: str,
        name: str = "content.md",
        format: str = "md",
        source_uri: str | None = None,
    ) -> "DoclingDocument":
        """Convert text content to DoclingDocument format.

        Args:
            text: The text content to convert.
            name: The name to use for the document (defaults to "content.md").
            format: The format of the text content ("md", "html", or "plain").
                Defaults to "md". Use "plain" for plain text without parsing.
            source_uri: Optional origin URI used by docling's HTML/Markdown
                backends to resolve relative image references.

        Returns:
            DoclingDocument representation of the text.

        Raises:
            ValueError: If the text cannot be converted or format is unsupported.
        """
        pass
