import pytest
from typer.testing import CliRunner

from haiku.rag.cli import _cli as cli
from haiku.rag.config.models import AppConfig, IngesterConfig, LanceDBConfig
from haiku.rag.folders import read_register

runner = CliRunner()


@pytest.fixture
def config(tmp_path, monkeypatch) -> AppConfig:
    config = AppConfig(
        lancedb=LanceDBConfig(databases={"files": str(tmp_path / "files.lancedb")}),
        ingester=IngesterConfig(folders_register=tmp_path / "folders.yaml"),
    )
    monkeypatch.setattr("haiku.rag.cli.get_config", lambda: config)
    return config


def test_folders_are_added_listed_updated_and_removed(tmp_path, config):  # noqa: ARG001
    books = tmp_path / "books"
    books.mkdir()

    result = runner.invoke(cli, ["folders", "list"])
    assert result.exit_code == 0
    assert "No folders are registered." in result.output

    result = runner.invoke(
        cli,
        ["folders", "add", str(books), "--include", "*.pdf", "--exclude", "old/"],
    )
    assert result.exit_code == 0, result.output
    assert f"books\tfiles\t{books.resolve()}" in result.output
    assert "include: *.pdf" in result.output
    assert "exclude: old/" in result.output

    result = runner.invoke(cli, ["folders", "list"])
    assert f"books\tfiles\t{books.resolve()}" in result.output

    result = runner.invoke(
        cli, ["folders", "update", "books", "--clear-include", "--clear-exclude"]
    )
    assert result.exit_code == 0
    assert "include:" not in result.output
    result = runner.invoke(cli, ["folders", "update", "books", "--include", "*.md"])
    assert "include: *.md" in result.output

    result = runner.invoke(cli, ["folders", "remove", "books"])
    assert result.exit_code == 0
    assert f"Removed books ({books.resolve()})" in result.output
    assert [r.id for r in read_register(tmp_path / "folders.yaml").retired] == ["books"]


def test_removing_can_keep_the_documents(tmp_path, config):  # noqa: ARG001
    books = tmp_path / "books"
    books.mkdir()
    runner.invoke(cli, ["folders", "add", str(books), "--id", "b"])
    result = runner.invoke(cli, ["folders", "remove", "b", "--keep-documents"])
    assert result.exit_code == 0
    assert read_register(tmp_path / "folders.yaml").retired == []


@pytest.mark.parametrize(
    "args",
    [
        ["folders", "add", "/no/such/dir"],
        ["folders", "update", "nope", "--include", "*.pdf"],
        ["folders", "remove", "nope"],
    ],
)
def test_register_errors_exit_1(config, args):  # noqa: ARG001
    result = runner.invoke(cli, args)
    assert result.exit_code == 1
    assert "Error:" in result.output


def test_folders_need_a_register(monkeypatch):
    monkeypatch.setattr("haiku.rag.cli.get_config", AppConfig)
    result = runner.invoke(cli, ["folders", "list"])
    assert result.exit_code == 1
    assert "folders_register" in result.output
