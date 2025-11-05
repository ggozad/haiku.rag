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
    def convert_file(self, path: Path) -> "DoclingDocument":
        """Convert a file to DoclingDocument format.

        Args:
            path: Path to the file to convert.

        Returns:
            DoclingDocument representation of the file.

        Raises:
            ValueError: If the file cannot be converted.
        """
        pass

    @abstractmethod
    def convert_text(self, text: str, name: str = "content.md") -> "DoclingDocument":
        """Convert text content to DoclingDocument format.

        Args:
            text: The text content to convert.
            name: The name to use for the document (defaults to "content.md").

        Returns:
            DoclingDocument representation of the text.

        Raises:
            ValueError: If the text cannot be converted.
        """
        pass
