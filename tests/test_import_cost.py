import subprocess
import sys

import pytest

HEAVY = "{'lancedb', 'pyarrow', 'pydantic_ai'}"

ENTRY_POINTS = [
    "import haiku.rag.client.scope",
    "import haiku.rag.client.exceptions",
    "import haiku.rag.sources.registry",
    "from haiku.rag.config import AppConfig\n"
    "from haiku.rag.ingester.app import IngesterApp\n"
    "IngesterApp(config=AppConfig())",
]


@pytest.mark.parametrize("entry_point", ENTRY_POINTS, ids=lambda p: p.splitlines()[-1])
def test_entry_point_does_not_load_heavy_dependencies(entry_point: str):
    """Reaching a database scope, a source or the ingester must not load the
    runtime dependencies that cost seconds of startup. Runs in a subprocess
    because another test in the same session may already have imported them."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"{entry_point}\n"
            "import sys\n"
            f"loaded = {HEAVY} & sys.modules.keys()\n"
            "assert not loaded, loaded\n",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
