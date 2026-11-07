import tomllib
from pathlib import Path


def test_the_lock_installs_only_the_headless_opencv():
    lock = tomllib.loads(
        (Path(__file__).parents[1] / "uv.lock").read_text(encoding="utf-8")
    )
    names = {package["name"] for package in lock["package"]}
    assert "opencv-python-headless" in names
    assert "opencv-python" not in names
