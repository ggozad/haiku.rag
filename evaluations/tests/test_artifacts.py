from pathlib import Path
from unittest.mock import MagicMock, patch

from evaluations.artifacts import (
    HF_REPO_ID,
    download_dataset_db,
    upload_dataset_db,
)
from evaluations.config import DatasetSpec


def _spec() -> DatasetSpec:
    return DatasetSpec(
        key="sample",
        db_filename="sample.lancedb",
        document_loader=lambda: None,  # ty: ignore[invalid-argument-type]
        document_mapper=lambda doc: None,
        qa_loader=lambda: None,  # ty: ignore[invalid-argument-type]
        qa_case_builder=lambda index, doc: None,  # ty: ignore[invalid-argument-type]
    )


def _db_path(root: Path) -> Path:
    return root / "evaluations" / "dbs" / "sample.lancedb"


class TestDownloadDatasetDb:
    def test_existing_database_is_preserved(self, tmp_path: Path) -> None:
        db = _db_path(tmp_path)
        db.mkdir(parents=True)

        with (
            patch("haiku.rag.utils.get_default_data_dir", return_value=tmp_path),
            patch("evaluations.artifacts.snapshot_download") as download,
            patch("evaluations.artifacts.console"),
        ):
            download_dataset_db(_spec())

        download.assert_not_called()

    def test_download_replaces_database_when_forced(self, tmp_path: Path) -> None:
        db = _db_path(tmp_path)
        db.mkdir(parents=True)
        (db / "stale").write_text("old")
        snapshot = tmp_path / "snapshot"
        source = snapshot / "sample.lancedb"
        source.mkdir(parents=True)
        (source / "data").write_text("new")

        with (
            patch("haiku.rag.utils.get_default_data_dir", return_value=tmp_path),
            patch(
                "evaluations.artifacts.snapshot_download", return_value=str(snapshot)
            ) as download,
            patch("evaluations.artifacts.console"),
        ):
            download_dataset_db(_spec(), force=True)

        download.assert_called_once_with(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            allow_patterns="sample.lancedb/*",
        )
        assert (db / "data").read_text() == "new"
        assert not (db / "stale").exists()

    def test_download_failure_leaves_no_database(self, tmp_path: Path) -> None:
        with (
            patch("haiku.rag.utils.get_default_data_dir", return_value=tmp_path),
            patch(
                "evaluations.artifacts.snapshot_download",
                side_effect=RuntimeError("offline"),
            ),
            patch("evaluations.artifacts.console") as console,
        ):
            download_dataset_db(_spec())

        assert "offline" in console.print.call_args.args[0]
        assert not _db_path(tmp_path).exists()

    def test_missing_database_in_snapshot_is_reported(self, tmp_path: Path) -> None:
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()

        with (
            patch("haiku.rag.utils.get_default_data_dir", return_value=tmp_path),
            patch(
                "evaluations.artifacts.snapshot_download", return_value=str(snapshot)
            ),
            patch("evaluations.artifacts.console") as console,
        ):
            download_dataset_db(_spec())

        messages = [call.args[0] for call in console.print.call_args_list]
        assert any("Database sample not found" in message for message in messages)
        assert not _db_path(tmp_path).exists()


class TestUploadDatasetDb:
    def test_missing_database_is_not_uploaded(self, tmp_path: Path) -> None:
        with (
            patch("haiku.rag.utils.get_default_data_dir", return_value=tmp_path),
            patch("evaluations.artifacts.HfApi") as api,
            patch("evaluations.artifacts.console"),
        ):
            upload_dataset_db(_spec())

        api.assert_not_called()

    def test_upload_stages_database_under_its_remote_name(self, tmp_path: Path) -> None:
        db = _db_path(tmp_path)
        (db / "nested").mkdir(parents=True)
        (db / "root.lance").write_text("root")
        (db / "nested" / "part.lance").write_text("part")
        uploaded: dict[str, str] = {}

        def capture_upload(*, folder_path: str, **kwargs: str) -> None:
            staged = Path(folder_path) / "sample.lancedb"
            uploaded["root"] = (staged / "root.lance").read_text()
            uploaded["part"] = (staged / "nested" / "part.lance").read_text()

        api = MagicMock()
        api.upload_large_folder.side_effect = capture_upload
        with (
            patch("haiku.rag.utils.get_default_data_dir", return_value=tmp_path),
            patch("evaluations.artifacts.HfApi", return_value=api),
            patch("evaluations.artifacts.console"),
        ):
            upload_dataset_db(_spec())

        api.delete_folder.assert_called_once_with(
            path_in_repo="sample.lancedb",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
        )
        assert uploaded == {"root": "root", "part": "part"}
        assert api.upload_large_folder.call_args.kwargs["repo_id"] == HF_REPO_ID
        assert api.upload_large_folder.call_args.kwargs["repo_type"] == "dataset"

    def test_missing_remote_folder_does_not_block_upload(self, tmp_path: Path) -> None:
        db = _db_path(tmp_path)
        db.mkdir(parents=True)
        (db / "data").write_text("content")
        api = MagicMock()
        api.delete_folder.side_effect = RuntimeError("not found")

        with (
            patch("haiku.rag.utils.get_default_data_dir", return_value=tmp_path),
            patch("evaluations.artifacts.HfApi", return_value=api),
            patch("evaluations.artifacts.console"),
        ):
            upload_dataset_db(_spec())

        api.upload_large_folder.assert_called_once()
