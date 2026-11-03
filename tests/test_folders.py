from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from haiku.rag.config import AppConfig, load_yaml_config
from haiku.rag.config.models import (
    FSSourceConfig,
    IngestedDatabase,
    IngesterConfig,
    LanceDBConfig,
)
from haiku.rag.folders import (
    FolderConflictError,
    FolderRegister,
    FolderRegisterError,
    UnknownFolderError,
    read_register,
)


def _config(tmp_path: Path, *databases: str, **ingester) -> AppConfig:
    return AppConfig(
        lancedb=LanceDBConfig(
            databases={name: str(tmp_path / f"{name}.lancedb") for name in databases}
        ),
        ingester=IngesterConfig(folders_register=tmp_path / "folders.yaml", **ingester),
    )


def _dir(tmp_path: Path, *parts: str) -> Path:
    path = tmp_path.joinpath(*parts)
    path.mkdir(parents=True)
    return path


@pytest.fixture
def register(tmp_path) -> FolderRegister:
    return FolderRegister.of(_config(tmp_path, "files"))


def test_a_folder_is_added_with_a_derived_id(tmp_path, register):
    books = _dir(tmp_path, "Guitar Books & Sheets")

    folder = register.add(books, include=["*.pdf"], exclude=["drafts/"])

    assert folder.id == "guitar-books-sheets"
    assert folder.database == "files"
    assert folder.path == books.resolve()
    assert register.folders() == [folder]
    assert register.get("guitar-books-sheets") == folder
    text = (tmp_path / "folders.yaml").read_text(encoding="utf-8")
    assert text.startswith("# The folders haiku-ingester watches")


def test_derived_ids_are_unique(tmp_path, register):
    first = register.add(_dir(tmp_path, "a", "notes"))
    second = register.add(_dir(tmp_path, "b", "notes"))
    third = register.add(_dir(tmp_path, "c", "notes"))
    assert [first.id, second.id, third.id] == ["notes", "notes-2", "notes-3"]


def test_a_name_without_letters_gets_a_generic_id(tmp_path, register):
    assert register.add(_dir(tmp_path, "___")).id == "folder"


def test_an_id_can_be_given(tmp_path, register):
    assert register.add(_dir(tmp_path, "x"), folder_id="mine").id == "mine"
    with pytest.raises(FolderConflictError, match="taken"):
        register.add(_dir(tmp_path, "y"), folder_id="mine")


def test_a_configured_source_id_is_taken(tmp_path):
    config = _config(
        tmp_path,
        "files",
        databases={
            "files": IngestedDatabase(
                sources=[FSSourceConfig(type="fs", id="notes", root=tmp_path)]
            )
        },
    )
    register = FolderRegister.of(config)
    assert register.add(_dir(tmp_path, "notes")).id == "notes-2"


def test_a_folder_registered_twice_or_overlapping_conflicts(tmp_path, register):
    books = _dir(tmp_path, "books")
    register.add(books)
    with pytest.raises(FolderConflictError, match="registered already"):
        register.add(books)
    with pytest.raises(FolderConflictError, match="overlaps"):
        register.add(_dir(tmp_path, "books", "jazz"))
    with pytest.raises(FolderConflictError, match="overlaps"):
        register.add(tmp_path)


def test_a_missing_directory_is_refused(tmp_path, register):
    with pytest.raises(FolderRegisterError, match="not a directory"):
        register.add(tmp_path / "nope")


def test_the_database_defaults_to_the_only_one(tmp_path):
    register = FolderRegister.of(_config(tmp_path))
    assert register.add(_dir(tmp_path, "x")).database == "haiku.rag"


def test_several_databases_need_one_named(tmp_path):
    register = FolderRegister.of(_config(tmp_path, "files", "work"))
    x = _dir(tmp_path, "x")
    with pytest.raises(FolderRegisterError, match="name the one"):
        register.add(x)
    with pytest.raises(FolderRegisterError, match="Unknown database 'other'"):
        register.add(x, database="other")
    assert register.add(_dir(tmp_path, "y"), database="work").database == "work"


def test_patterns_are_updated(tmp_path, register):
    register.add(_dir(tmp_path, "x"), include=["*.pdf"])
    folder = register.update("x", exclude=["old/"])
    assert (folder.include, folder.exclude) == (["*.pdf"], ["old/"])
    folder = register.update("x", include=[])
    assert (folder.include, folder.exclude) == ([], ["old/"])
    with pytest.raises(UnknownFolderError) as missing:
        register.update("nope", include=[])
    assert str(missing.value) == "No folder 'nope' is registered"
    with pytest.raises(UnknownFolderError):
        register.get("nope")


def test_a_removed_folder_is_retired_unless_its_documents_are_kept(tmp_path, register):
    register.add(_dir(tmp_path, "x"))
    register.add(_dir(tmp_path, "y"))
    register.remove("x")
    register.remove("y", keep_documents=True)
    contents = read_register(tmp_path / "folders.yaml")
    assert contents.folders == []
    assert [(r.id, r.database) for r in contents.retired] == [("x", "files")]
    with pytest.raises(UnknownFolderError):
        register.remove("x")


def test_adding_a_retired_id_again_unretires_it(tmp_path, register):
    x = _dir(tmp_path, "x")
    register.add(x)
    register.remove("x")
    register.add(x)
    assert read_register(tmp_path / "folders.yaml").retired == []


def test_a_failed_write_leaves_the_register_as_it_was(tmp_path, register, monkeypatch):
    register.add(_dir(tmp_path, "x"))
    before = (tmp_path / "folders.yaml").read_text(encoding="utf-8")

    def refuse(*args):
        raise OSError("disk full")

    monkeypatch.setattr("haiku.rag.folders.os.replace", refuse)
    with pytest.raises(OSError, match="disk full"):
        register.add(_dir(tmp_path, "y"))
    assert (tmp_path / "folders.yaml").read_text(encoding="utf-8") == before
    assert sorted(
        p.name for p in tmp_path.iterdir() if p.name.startswith(".folders")
    ) == [".folders.yaml.lock"]


def test_an_unreadable_register_is_reported(tmp_path, register):
    (tmp_path / "folders.yaml").write_text("folders: [{id: x}]\n", encoding="utf-8")
    with pytest.raises(FolderRegisterError, match="folders.yaml"):
        register.folders()
    (tmp_path / "folders.yaml").write_text("folders: [\n", encoding="utf-8")
    with pytest.raises(FolderRegisterError, match="folders.yaml"):
        register.folders()


def test_a_register_must_be_configured():
    with pytest.raises(FolderRegisterError, match="folders_register"):
        FolderRegister.of(AppConfig())


def test_the_loader_adds_the_folders_to_their_databases(tmp_path):
    books = _dir(tmp_path, "books")
    notes = _dir(tmp_path, "notes")
    (tmp_path / "haiku.rag.yaml").write_text(
        yaml.safe_dump(
            {
                "lancedb": {"databases": {"files": str(tmp_path / "files.lancedb")}},
                "ingester": {
                    "folders_register": "state/folders.yaml",
                    "databases": {
                        "files": {
                            "sources": [{"type": "fs", "id": "other", "root": "/x"}]
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    config = AppConfig.model_validate(load_yaml_config(tmp_path / "haiku.rag.yaml"))
    register = FolderRegister.of(config)
    assert register.path == tmp_path / "state" / "folders.yaml"
    register.add(books, include=["*.pdf"])
    register.add(notes)
    register.remove("notes")

    config = AppConfig.model_validate(load_yaml_config(tmp_path / "haiku.rag.yaml"))
    entry = config.ingester.databases["files"]
    assert [(s.id, getattr(s, "root", None)) for s in entry.sources] == [
        ("other", Path("/x")),
        ("books", books.resolve()),
    ]
    assert entry.sources[1].include_patterns == ["*.pdf"]  # ty: ignore[unresolved-attribute]
    assert entry.retired_sources == ["notes"]


def test_the_loader_takes_an_absolute_register_and_an_empty_one(tmp_path):
    register = tmp_path / "elsewhere" / "folders.yaml"
    (tmp_path / "haiku.rag.yaml").write_text(
        f"ingester:\n  folders_register: {register}\n", encoding="utf-8"
    )
    data = load_yaml_config(tmp_path / "haiku.rag.yaml")
    assert data["ingester"] == {"folders_register": str(register), "databases": {}}


def test_a_database_left_with_only_retired_sources_is_valid():
    entry = IngestedDatabase(retired_sources=["gone"])
    assert entry.sources == []
    with pytest.raises(ValidationError, match="at least one source"):
        IngestedDatabase()
