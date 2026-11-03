import os
import re
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import Field

from haiku.rag.config.models import DEFAULT_DATABASE_NAME, ConfigModel

if TYPE_CHECKING:
    from haiku.rag.config import AppConfig

_HEADER = (
    "# The folders haiku-ingester watches and indexes. Managed with\n"
    "# `haiku-rag folders`, the service API and MCP; hand edits are kept.\n"
)


class FolderRegisterError(ValueError):
    """The register cannot take the change, or is not configured."""


class FolderConflictError(FolderRegisterError):
    """The id or the directory is registered already."""


class UnknownFolderError(FolderRegisterError, KeyError):
    """No folder has the id."""

    def __str__(self) -> str:
        return str(self.args[0])


class Folder(ConfigModel):
    """A local directory indexed into `database`, as the FS source `id`.

    `include` and `exclude` are gitignore-style patterns matched against each
    file's absolute path; an empty `include` takes every supported file.
    """

    id: str
    database: str
    path: Path
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)

    def source(self) -> dict[str, Any]:
        """The folder as an `ingester.databases` FS source."""
        return {
            "type": "fs",
            "id": self.id,
            "root": str(self.path),
            "include_patterns": list(self.include),
            "ignore_patterns": list(self.exclude),
        }


class RetiredFolder(ConfigModel):
    """A removed folder whose documents leave the index when its database
    next starts."""

    id: str
    database: str


class RegisterContents(ConfigModel):
    folders: list[Folder] = Field(default_factory=list)
    retired: list[RetiredFolder] = Field(default_factory=list)


def merge_register(data: dict, path: Path) -> None:
    """Add the register's folders to `data` as FS sources of their databases,
    and its retired folders as retired sources."""
    contents = read_register(path)
    ingester = data.setdefault("ingester", {}) or {}
    data["ingester"] = ingester
    databases = ingester.get("databases") or {}
    ingester["databases"] = databases

    def entry(name: str) -> dict:
        databases[name] = databases.get(name) or {}
        return databases[name]

    for folder in contents.folders:
        target = entry(folder.database)
        target["sources"] = [*(target.get("sources") or []), folder.source()]
    for retired in contents.retired:
        target = entry(retired.database)
        target["retired_sources"] = [
            *(target.get("retired_sources") or []),
            retired.id,
        ]


def read_register(path: Path) -> RegisterContents:
    """The register at `path`; empty when the file does not exist."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return RegisterContents()
    try:
        return RegisterContents.model_validate(yaml.safe_load(text) or {})
    except (yaml.YAMLError, ValueError) as error:
        raise FolderRegisterError(f"{path}: {error}") from None


class FolderRegister:
    """The register file of folders the ingester indexes.

    Every change is a locked read-modify-write and replaces the file whole, so
    a reader sees the old register or the new one. A running ingester watching
    the file applies the change.
    """

    def __init__(self, path: Path, config: "AppConfig"):
        self.path = path
        self.config = config

    @classmethod
    def of(cls, config: "AppConfig") -> "FolderRegister":
        path = config.ingester.folders_register
        if path is None:
            raise FolderRegisterError(
                "No folder register is configured. Set ingester.folders_register "
                "to the YAML file that lists the indexed folders"
            )
        return cls(path, config)

    def folders(self) -> list[Folder]:
        return read_register(self.path).folders

    def get(self, folder_id: str) -> Folder:
        for folder in self.folders():
            if folder.id == folder_id:
                return folder
        raise UnknownFolderError(f"No folder {folder_id!r} is registered")

    def add(
        self,
        path: Path | str,
        *,
        database: str | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        folder_id: str | None = None,
    ) -> Folder:
        directory = Path(path).expanduser().resolve()
        if not directory.is_dir():
            raise FolderRegisterError(f"{directory} is not a directory")
        database = self._database(database)
        with self._editing() as contents:
            for folder in contents.folders:
                if folder.path == directory:
                    raise FolderConflictError(
                        f"{directory} is registered already as {folder.id!r}"
                    )
                if folder.path.is_relative_to(directory) or directory.is_relative_to(
                    folder.path
                ):
                    raise FolderConflictError(
                        f"{directory} overlaps folder {folder.id!r} at "
                        f"{folder.path}: a file must belong to one folder"
                    )
            taken = {folder.id for folder in contents.folders} | self._source_ids()
            if folder_id is None:
                folder_id = _unique(_slug(directory.name), taken)
            elif folder_id in taken:
                raise FolderConflictError(f"The id {folder_id!r} is taken")
            folder = Folder(
                id=folder_id,
                database=database,
                path=directory,
                include=list(include or []),
                exclude=list(exclude or []),
            )
            contents.folders.append(folder)
            contents.retired = [r for r in contents.retired if r.id != folder_id]
        return folder

    def update(
        self,
        folder_id: str,
        *,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> Folder:
        with self._editing() as contents:
            folder = _find(contents, folder_id)
            if include is not None:
                folder.include = list(include)
            if exclude is not None:
                folder.exclude = list(exclude)
        return folder

    def remove(self, folder_id: str, *, keep_documents: bool = False) -> Folder:
        """Unregister a folder. Unless `keep_documents`, its documents leave
        the index when the ingester next starts its database."""
        with self._editing() as contents:
            folder = _find(contents, folder_id)
            contents.folders.remove(folder)
            if not keep_documents:
                contents.retired.append(
                    RetiredFolder(id=folder.id, database=folder.database)
                )
        return folder

    def _database(self, name: str | None) -> str:
        configured = list(self.config.lancedb.databases) or [DEFAULT_DATABASE_NAME]
        if name is None:
            if len(configured) > 1:
                raise FolderRegisterError(
                    "Several databases are configured; name the one to index "
                    f"the folder into: {', '.join(configured)}"
                )
            return configured[0]
        if name not in configured:
            raise FolderRegisterError(
                f"Unknown database {name!r}; configured: {', '.join(configured)}"
            )
        return name

    def _source_ids(self) -> set[str]:
        """Explicit ids of the configured sources that are not folders."""
        folders = {folder.id for folder in self.folders()}
        return {
            source.id
            for entry in self.config.ingester.databases.values()
            for source in entry.sources
            if source.id is not None and source.id not in folders
        }

    @contextmanager
    def _editing(self) -> Iterator[RegisterContents]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _locked(self.path.with_name(f".{self.path.name}.lock")):
            contents = read_register(self.path)
            yield contents
            _write(self.path, contents)


def _find(contents: RegisterContents, folder_id: str) -> Folder:
    for folder in contents.folders:
        if folder.id == folder_id:
            return folder
    raise UnknownFolderError(f"No folder {folder_id!r} is registered")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "folder"


def _unique(base: str, taken: set[str]) -> str:
    candidate, n = base, 2
    while candidate in taken:
        candidate, n = f"{base}-{n}", n + 1
    return candidate


def _write(path: Path, contents: RegisterContents) -> None:
    body = yaml.safe_dump(
        contents.model_dump(mode="json"), sort_keys=False, allow_unicode=True
    )
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_HEADER + body)
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    with open(path, "a+b") as handle:
        if sys.platform == "win32":  # pragma: no cover - Windows has no coverage
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
        yield
