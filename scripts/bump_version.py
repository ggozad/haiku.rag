#!/usr/bin/env python3
"""
Version bumping script for haiku.rag workspace.

Updates version in all pyproject.toml files and CHANGELOG.md.
"""

import re
import subprocess
import sys
from datetime import date
from pathlib import Path


def get_current_version(file_path: Path) -> str:
    """Extract current version from pyproject.toml."""
    content = file_path.read_text()
    match = re.search(r'^version = "([^"]+)"', content, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find version in {file_path}")
    return match.group(1)


def update_version_in_file(file_path: Path, new_version: str) -> None:
    """Update version in a pyproject.toml file."""
    content = file_path.read_text()
    updated = re.sub(
        r'^version = "[^"]+"', f'version = "{new_version}"', content, flags=re.MULTILINE
    )
    file_path.write_text(updated)
    print(f"✓ Updated {file_path.relative_to(Path.cwd())}")


def update_changelog(changelog_path: Path, new_version: str) -> None:
    """Update CHANGELOG.md with new version."""
    content = changelog_path.read_text()
    today = date.today().isoformat()

    # Replace [Unreleased] with new version
    updated = re.sub(
        r"## \[Unreleased\]",
        f"## [Unreleased]\n\n## [{new_version}] - {today}",
        content,
        count=1,
    )

    # Update comparison links
    # Find the old [Unreleased] link
    old_unreleased_match = re.search(
        r"\[Unreleased\]: https://github\.com/ggozad/haiku\.rag/compare/([^.]+)\.\.\.HEAD",
        updated,
    )

    if old_unreleased_match:
        prev_version = old_unreleased_match.group(1)

        # Update [Unreleased] link
        updated = re.sub(
            r"\[Unreleased\]: https://github\.com/ggozad/haiku\.rag/compare/[^.]+\.\.\.HEAD",
            f"[Unreleased]: https://github.com/ggozad/haiku.rag/compare/{new_version}...HEAD",
            updated,
        )

        # Add new version link after [Unreleased]
        updated = re.sub(
            r"(\[Unreleased\]: https://github\.com/ggozad/haiku\.rag/compare/[^\n]+\n)",
            f"\\1[{new_version}]: https://github.com/ggozad/haiku.rag/compare/{prev_version}...{new_version}\n",
            updated,
        )

    changelog_path.write_text(updated)
    print(f"✓ Updated {changelog_path.relative_to(Path.cwd())}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/bump_version.py <new_version>")
        print("Example: python scripts/bump_version.py 0.14.0")
        sys.exit(1)

    new_version = sys.argv[1]

    # Validate version format
    if not re.match(r"^\d+\.\d+\.\d+$", new_version):
        print(f"Error: Invalid version format '{new_version}'")
        print("Version must be in format: X.Y.Z (e.g., 0.14.0)")
        sys.exit(1)

    root = Path(__file__).parent.parent

    # Files to update
    pyproject_files = [
        root / "pyproject.toml",
        root / "haiku_rag_slim" / "pyproject.toml",
        root / "evaluations" / "pyproject.toml",
    ]

    changelog_file = root / "CHANGELOG.md"

    # Check all files exist
    for file in pyproject_files + [changelog_file]:
        if not file.exists():
            print(f"Error: {file} not found")
            sys.exit(1)

    # Get current version from root pyproject.toml
    current_version = get_current_version(pyproject_files[0])
    print(f"Current version: {current_version}")
    print(f"New version: {new_version}")
    print()

    # Confirm
    response = input("Proceed with version bump? [y/N] ")
    if response.lower() != "y":
        print("Aborted.")
        sys.exit(0)

    print()

    # Update all pyproject.toml files
    for file in pyproject_files:
        update_version_in_file(file, new_version)

    # Update CHANGELOG.md
    update_changelog(changelog_file, new_version)

    # Run uv sync to update lock file
    print()
    print("Running uv sync...")
    try:
        subprocess.run(["uv", "sync"], check=True, cwd=root)
        print("✓ Lock file updated")
    except subprocess.CalledProcessError as e:
        print(f"Error: uv sync failed with exit code {e.returncode}")
        sys.exit(1)

    print()
    print(f"✓ Version bumped from {current_version} to {new_version}")


if __name__ == "__main__":
    main()
