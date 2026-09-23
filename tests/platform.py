import tempfile
from pathlib import Path

import pytest


def _symlinks_available() -> bool:
    """Whether this process can create a symlink.

    Windows grants the privilege only to an administrator or with Developer
    Mode on, and raises OSError(WinError 1314) otherwise.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / "target"
        target.write_text("x", encoding="utf-8")
        try:
            (root / "link").symlink_to(target)
        except (OSError, NotImplementedError):
            return False
    return True


requires_symlinks = pytest.mark.skipif(
    not _symlinks_available(),
    reason="creating a symlink needs privileges this process does not hold",
)
