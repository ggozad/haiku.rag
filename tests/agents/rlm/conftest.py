import os
import subprocess
from pathlib import Path

import pytest

from haiku.rag.agents.rlm.dependencies import RLMContext
from haiku.rag.agents.rlm.docker_sandbox import DockerSandbox
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import RLMConfig

TEST_DOCKER_IMAGE = os.environ.get("HAIKU_TEST_DOCKER_IMAGE", "haiku-rag-slim:test")


@pytest.fixture(scope="session")
def test_docker_image():
    """Build and return the Docker image for testing."""
    if os.environ.get("CI"):
        return TEST_DOCKER_IMAGE

    project_root = Path(__file__).parent.parent.parent.parent
    dockerfile = project_root / "docker" / "Dockerfile.slim"

    if not dockerfile.exists():
        pytest.skip(f"Dockerfile.slim not found at {dockerfile}")

    result = subprocess.run(
        ["docker", "build", "-t", TEST_DOCKER_IMAGE, "-f", str(dockerfile), "."],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to build Docker image:\n{result.stderr}")

    return TEST_DOCKER_IMAGE


@pytest.fixture
async def empty_client(temp_db_path):
    """Create an empty HaikuRAG client without documents."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        yield client


@pytest.fixture
async def docker_sandbox(empty_client, test_docker_image):
    """Create a Docker sandbox for testing."""
    config = RLMConfig(docker_image=test_docker_image)
    context = RLMContext()
    async with DockerSandbox(
        client=empty_client, config=config, context=context, image=test_docker_image
    ) as sandbox:
        yield sandbox
