"""Docker-based sandboxed execution."""

import asyncio
import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import docker
import docker.errors

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
    CONTAINER_PORT = 19876

    haiku_client: "HaikuRAG"
    config: RLMConfig
    context: RLMContext
    image: str
    _docker_client: Any
    _container: Any
    _socket: socket.socket | None

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
        self._docker_client = None
        self._container = None
        self._socket = None

    def _use_host_network(self) -> bool:
        """Host networking only works for TCP on Linux with local Docker."""
        return sys.platform == "linux" and not self.config.docker_host

    def _build_environment(self) -> dict[str, str]:
        """Build environment variables for the container."""
        env: dict[str, str] = {"HAIKU_DB_PATH": "/data/db.lancedb"}

        if self.context.filter:
            env["HAIKU_FILTER"] = self.context.filter

        ollama_host = os.environ.get("OLLAMA_HOST", "")
        ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "")

        if not self._use_host_network():
            if not ollama_host or "localhost" in ollama_host:
                ollama_host = "http://host.docker.internal:11434"
            if not ollama_base_url or "localhost" in ollama_base_url:
                ollama_base_url = "http://host.docker.internal:11434"

        if ollama_host:
            env["OLLAMA_HOST"] = ollama_host
        if ollama_base_url:
            env["OLLAMA_BASE_URL"] = ollama_base_url

        for key in [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "VOYAGE_API_KEY",
            "COHERE_API_KEY",
        ]:
            if value := os.environ.get(key):
                env[key] = value

        return env

    def _resolve_connection_host(self) -> str:
        """Derive the host to connect to from docker_host config."""
        docker_host = self.config.docker_host
        if not docker_host:
            return "localhost"

        parsed = urlparse(docker_host)
        hostname = parsed.hostname
        if not hostname or hostname in ("", "localhost", "127.0.0.1"):
            return "localhost"
        return hostname

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
        """Start the persistent container and connect via TCP."""
        if self._container is not None:
            return

        if self.config.docker_host:
            self._docker_client = docker.DockerClient(base_url=self.config.docker_host)
        else:
            self._docker_client = docker.from_env()

        db_path = self.config.docker_db_path or str(self.haiku_client.store.db_path)
        env = self._build_environment()
        use_host = self._use_host_network()

        run_kwargs: dict[str, Any] = {
            "detach": True,
            "mem_limit": self.config.docker_memory_limit,
            "volumes": {db_path: {"bind": "/data/db.lancedb", "mode": "ro"}},
        }

        if use_host:
            run_kwargs["network_mode"] = "host"
        else:
            # Fixed container port, Docker picks a random host port
            env["HAIKU_SANDBOX_PORT"] = str(self.CONTAINER_PORT)
            run_kwargs["ports"] = {f"{self.CONTAINER_PORT}/tcp": None}
            # host.docker.internal on Linux requires extra_hosts
            if sys.platform == "linux":
                run_kwargs["extra_hosts"] = {"host.docker.internal": "host-gateway"}

        run_kwargs["environment"] = env

        self._container = self._docker_client.containers.run(
            self.image,
            command=["python", "-m", "haiku.rag.agents.rlm.runner"],
            **run_kwargs,
        )

        self._wait_for_port()

        host = self._resolve_connection_host()
        if use_host:
            port = self._read_port_from_logs()
        else:
            port = self._read_published_port()

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((host, port))

    def _wait_for_port(self, timeout: float = 30.0) -> None:
        """Wait for the container to report its TCP port (readiness signal)."""
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            self._container.reload()
            logs = self._container.logs().decode(errors="replace")

            for line in logs.splitlines():
                if line.startswith("PORT:"):
                    return

            if self._container.status != "running":
                exit_info = self._container.attrs.get("State", {})
                exit_code = exit_info.get("ExitCode", "unknown")
                oom = exit_info.get("OOMKilled", False)
                raise RuntimeError(
                    f"Container exited (code={exit_code}, OOMKilled={oom}) "
                    f"before reporting port. Logs: {logs}"
                )

            time.sleep(0.2)

        raise TimeoutError(
            f"Container did not report TCP port within {timeout}s. "
            f"Logs: {self._container.logs().decode(errors='replace')}"
        )

    def _read_port_from_logs(self) -> int:
        """Read the TCP port from container logs (host network mode)."""
        logs = self._container.logs().decode(errors="replace")
        for line in logs.splitlines():
            if line.startswith("PORT:"):
                return int(line.split(":")[1])
        raise RuntimeError(f"PORT line not found in container logs: {logs}")

    def _read_published_port(self) -> int:
        """Read the mapped host port from Docker port bindings."""
        self._container.reload()
        port_key = f"{self.CONTAINER_PORT}/tcp"
        mappings = self._container.ports.get(port_key)
        if not mappings:
            raise RuntimeError(
                f"No port mapping found for {port_key}. "
                f"Container ports: {self._container.ports}"
            )
        return int(mappings[0]["HostPort"])

    def _stop_container(self) -> None:
        """Stop the container and clean up."""
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None

        if self._container is not None:
            try:
                self._container.stop(timeout=5)
            except docker.errors.NotFound:
                pass
            try:
                self._container.remove(force=True)
            except docker.errors.NotFound:
                pass
            self._container = None

        if self._docker_client is not None:
            self._docker_client.close()
            self._docker_client = None

    async def execute(self, code: str) -> SandboxResult:
        """Execute code in the container."""
        if self._socket is None:
            return SandboxResult(
                stdout="",
                stderr="Container not started. Use 'async with' context manager.",
                success=False,
            )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._execute_sync, code)

    def _execute_sync(self, code: str) -> SandboxResult:
        """Send code to container and read result via TCP."""
        assert self._socket is not None

        try:
            self._socket.settimeout(self.config.code_timeout)

            message = json.dumps({"code": code})
            data = f"{len(message)}\n{message}".encode()
            self._socket.sendall(data)

            buf = b""
            while b"\n" not in buf:
                chunk = self._socket.recv(4096)
                if not chunk:
                    return SandboxResult(
                        stdout="",
                        stderr="Container closed connection unexpectedly.",
                        success=False,
                    )
                buf += chunk

            newline_idx = buf.index(b"\n")
            length = int(buf[:newline_idx].strip())
            buf = buf[newline_idx + 1 :]

            while len(buf) < length:
                chunk = self._socket.recv(4096)
                if not chunk:
                    return SandboxResult(
                        stdout="",
                        stderr="Container closed connection unexpectedly.",
                        success=False,
                    )
                buf += chunk

            response = buf[:length].decode()
            result_data = json.loads(response)

            return SandboxResult(
                stdout=result_data.get("stdout", ""),
                stderr=result_data.get("stderr", ""),
                success=result_data.get("success", False),
            )

        except TimeoutError:
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
