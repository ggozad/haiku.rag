"""Pre-built evaluation databases on HuggingFace."""

import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from rich.console import Console

from evaluations.config import DatasetSpec

console = Console()

HF_REPO_ID = "ggozad/haiku-rag-eval-dbs"


def download_dataset_db(spec: DatasetSpec, force: bool = False) -> None:
    """Fetch one dataset's database from HuggingFace into its local path."""
    db = spec.db_path()
    if db.exists() and not force:
        console.print(
            f"[yellow]Skipping {spec.key}: database already exists at {db}[/yellow]"
        )
        console.print("Use --force to overwrite.")
        return

    console.print(f"[blue]Downloading {spec.key}...[/blue]")

    try:
        downloaded_path = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            allow_patterns=f"{spec.db_filename}/*",
        )
    except Exception as e:
        console.print(f"[red]Failed to download {spec.key}: {e}[/red]")
        return

    source_path = Path(downloaded_path) / spec.db_filename
    if not source_path.exists():
        console.print(f"[red]Database {spec.key} not found in HuggingFace repo.[/red]")
        console.print(
            f"[yellow]The database may not have been uploaded yet. "
            f"Try running 'evaluations build {spec.key}' to create it locally.[/yellow]"
        )
        return

    if db.exists():
        shutil.rmtree(db)

    db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_path, db)

    console.print(f"[green]Downloaded {spec.key} to {db}[/green]")


def upload_dataset_db(spec: DatasetSpec) -> None:
    """Push one dataset's database to HuggingFace (maintainer only).

    Uses ``upload_large_folder`` for resumable, parallel transfer — important
    for the multi-GB ORB databases which would otherwise abort on any transient
    network failure under plain ``upload_folder``.

    ``upload_large_folder`` has no ``path_in_repo`` — it ships the contents of
    ``folder_path`` to the repo root. Stage the db under a temp parent with
    hardlinks so the basename becomes the remote path, leaving everything else
    at the root undisturbed.
    """
    db = spec.db_path()
    if not db.exists():
        console.print(f"[red]Database not found at {db}[/red]")
        return

    api = HfApi()

    # Wipe the existing remote path so we don't accumulate orphaned files from
    # prior uploads. upload_large_folder doesn't accept delete_patterns, so we
    # do this as a separate commit. Safe to run if the path is missing.
    try:
        api.delete_folder(
            path_in_repo=spec.db_filename,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
        )
    except Exception:
        pass

    with tempfile.TemporaryDirectory() as staging:
        target = Path(staging) / spec.db_filename
        target.mkdir()
        for src in db.rglob("*"):
            if not src.is_file():
                continue
            dest = target / src.relative_to(db)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.link(src, dest)

        console.print(f"[blue]Uploading {spec.key} ({db})...[/blue]")
        api.upload_large_folder(
            folder_path=staging,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
        )

    console.print(f"[green]Uploaded {spec.key} to {HF_REPO_ID}[/green]")
