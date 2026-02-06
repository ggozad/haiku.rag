"""Docker-based sandboxed execution."""

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from haiku.rag.agents.rlm.dependencies import RLMContext
from haiku.rag.config.models import RLMConfig

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG


@dataclass
class SandboxResult:
    """Result of executing code in the sandbox."""

    stdout: str
    stderr: str
    success: bool


class DockerSandbox:  # pragma: no cover
    """Execute code in a persistent Docker container.

    Use as an async context manager to manage container lifecycle:

        async with DockerSandbox(client, config, context) as sandbox:
            result = await sandbox.execute("print('hello')")
            result = await sandbox.execute("print('world')")
    """

    DEFAULT_IMAGE = "ghcr.io/ggozad/haiku.rag-slim:latest"

    haiku_client: "HaikuRAG"
    config: RLMConfig
    context: RLMContext
    image: str
    _process: subprocess.Popen[bytes] | None

    def __init__(
        self,
        client: "HaikuRAG",
        config: RLMConfig,
        context: RLMContext,
        image: str | None = None,
    ):
        self.haiku_client = client
        self.config = config
        self.context = context
        self.image = image or self.DEFAULT_IMAGE
        self._process = None

    def _build_docker_cmd(self) -> list[str]:
        """Build the docker run command."""
        db_path = str(self.haiku_client.store.db_path)

        env_list = ["-e", "HAIKU_DB_PATH=/data/db.lancedb"]
        if self.context.filter:
            env_list.extend(["-e", f"HAIKU_FILTER={self.context.filter}"])

        ollama_host = os.environ.get("OLLAMA_HOST", "")
        ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "")

        if sys.platform == "darwin":
            if not ollama_host or "localhost" in ollama_host:
                ollama_host = "http://host.docker.internal:11434"
            if not ollama_base_url or "localhost" in ollama_base_url:
                ollama_base_url = "http://host.docker.internal:11434"

        if ollama_host:
            env_list.extend(["-e", f"OLLAMA_HOST={ollama_host}"])
        if ollama_base_url:
            env_list.extend(["-e", f"OLLAMA_BASE_URL={ollama_base_url}"])

        for key in [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "VOYAGE_API_KEY",
            "COHERE_API_KEY",
        ]:
            if value := os.environ.get(key):
                env_list.extend(["-e", f"{key}={value}"])

        return [
            "docker",
            "run",
            "--rm",
            "-i",
            "-v",
            f"{db_path}:/data/db.lancedb:ro",
            f"--memory={self.config.docker_memory_limit}",
            "--network=host",
            *env_list,
            self.image,
            "python",
            "-m",
            "haiku.rag.agents.rlm.runner",
        ]

    async def __aenter__(self) -> "DockerSandbox":
        """Start the container."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._start_container)
        return self

    async def __aexit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        """Stop the container."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._stop_container)

    def _start_container(self) -> None:
        """Start the persistent container process."""
        if self._process is not None:
            return

        cmd = self._build_docker_cmd()
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _stop_container(self) -> None:
        """Stop the container process."""
        if self._process is None:
            return

        try:
            if self._process.stdin:
                try:
                    self._process.stdin.close()
                except BrokenPipeError:
                    pass
            self._process.terminate()
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        finally:
            self._process = None

    async def execute(self, code: str) -> SandboxResult:
        """Execute code in the container."""
        if self._process is None:
            return SandboxResult(
                stdout="",
                stderr="Container not started. Use 'async with' context manager.",
                success=False,
            )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._execute_sync, code)

    def _execute_sync(self, code: str) -> SandboxResult:
        """Send code to container and read result."""
        assert self._process is not None and self._process.stdin is not None

        try:
            message = json.dumps({"code": code})
            length_line = f"{len(message)}\n".encode()
            self._process.stdin.write(length_line)
            self._process.stdin.write(message.encode())
            self._process.stdin.flush()

            if self._process.stdout is None:
                return SandboxResult(
                    stdout="", stderr="No stdout from container.", success=False
                )

            length_line = self._process.stdout.readline()
            if not length_line:
                stderr = ""
                if self._process.stderr:
                    stderr = self._process.stderr.read().decode()
                return SandboxResult(
                    stdout="",
                    stderr=stderr or "Container closed unexpectedly.",
                    success=False,
                )

            length = int(length_line.strip())
            response = self._process.stdout.read(length).decode()
            result_data = json.loads(response)

            return SandboxResult(
                stdout=result_data.get("stdout", ""),
                stderr=result_data.get("stderr", ""),
                success=result_data.get("success", False),
            )

        except subprocess.TimeoutExpired:
            return SandboxResult(
                stdout="",
                stderr=f"Execution timed out after {self.config.code_timeout} seconds",
                success=False,
            )
        except json.JSONDecodeError as e:
            return SandboxResult(
                stdout="",
                stderr=f"Invalid response from container: {e}",
                success=False,
            )
        except Exception as e:
            return SandboxResult(
                stdout="",
                stderr=f"Execution error: {e}",
                success=False,
            )
