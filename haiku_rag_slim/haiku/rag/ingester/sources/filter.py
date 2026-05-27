import pathspec
from watchfiles import Change, DefaultFilter


def _default_supported_extensions() -> list[str]:
    from haiku.rag.converters.docling_local import DoclingLocalConverter
    from haiku.rag.converters.text_utils import TextFileHandler

    return DoclingLocalConverter.docling_extensions + TextFileHandler.text_extensions


class FileFilter(DefaultFilter):
    def __init__(
        self,
        *,
        ignore_patterns: list[str] | None = None,
        include_patterns: list[str] | None = None,
        supported_extensions: list[str] | None = None,
    ) -> None:
        if supported_extensions is None:
            supported_extensions = _default_supported_extensions()

        self.extensions = tuple(supported_extensions)
        self.ignore_spec = (
            pathspec.PathSpec.from_lines("gitwildmatch", ignore_patterns)
            if ignore_patterns
            else None
        )
        self.include_spec = (
            pathspec.PathSpec.from_lines("gitwildmatch", include_patterns)
            if include_patterns
            else None
        )
        super().__init__()

    def __call__(self, change: Change, path: str) -> bool:
        if not self.include_file(path):
            return False
        return super().__call__(change, path)

    def include_file(self, path: str) -> bool:
        if not path.endswith(self.extensions):
            return False

        if self.include_spec and not self.include_spec.match_file(path):
            return False

        if self.ignore_spec and self.ignore_spec.match_file(path):
            return False

        return True
