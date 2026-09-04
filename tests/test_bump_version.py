import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "bump_version", Path(__file__).resolve().parents[1] / "scripts" / "bump_version.py"
)
assert _spec is not None and _spec.loader is not None
bump_version = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bump_version)


def test_update_plugin_version_rewrites_only_the_version_field(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / "plugin.json"
    manifest.write_text(
        '{\n  "name": "haiku-rag",\n  "version": "0.1.0",\n  "license": "MIT"\n}\n'
    )

    bump_version.update_plugin_version(manifest, "0.2.0")

    assert json.loads(manifest.read_text()) == {
        "name": "haiku-rag",
        "version": "0.2.0",
        "license": "MIT",
    }
    assert manifest.read_text().endswith("}\n")


def test_the_shipped_plugin_manifest_carries_the_package_version():
    root = Path(__file__).resolve().parents[1]
    plugin = json.loads(
        (root / "claude-plugin" / ".claude-plugin" / "plugin.json").read_text()
    )

    assert plugin["version"] == bump_version.get_current_version(
        root / "haiku_rag_slim" / "pyproject.toml"
    )
