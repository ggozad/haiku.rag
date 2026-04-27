"""Shared client for docling-serve async API."""

import asyncio
from typing import Any

import httpx


class DoclingServeClient:
    """Client for docling-serve async workflow.

    Handles the submit → poll → fetch pattern used by both conversion and chunking.
    """

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        headers: dict[str, str] = {}
        if self.api_key:
            headers["X-Api-Key"] = self.api_key
        return headers

    async def _submit_and_wait(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        files: dict[str, Any],
        data: dict[str, Any],
        headers: dict[str, str],
        name: str,
    ) -> str:
        """Submit a task and poll until success. Returns the task_id.

        Shared by submit_and_poll (JSON results) and submit_and_poll_zip
        (binary zip results) — only the result-fetching step differs.
        """
        submit_url = f"{self.base_url}{endpoint}"
        response = await client.post(
            submit_url,
            files=files,
            data=data,
            headers=headers,
        )
        response.raise_for_status()
        submit_result = response.json()
        task_id = submit_result.get("task_id")

        if not task_id:
            raise ValueError("docling-serve did not return a task_id")

        poll_url = f"{self.base_url}/v1/status/poll/{task_id}"
        while True:
            poll_response = await client.get(poll_url, headers=headers)
            poll_response.raise_for_status()
            poll_result = poll_response.json()
            status = poll_result.get("task_status")

            if status == "success":
                return task_id
            elif status in ("failure", "error"):
                raise ValueError(f"docling-serve task failed for {name}: {poll_result}")

            await asyncio.sleep(1)

    async def submit_and_poll(
        self,
        endpoint: str,
        files: dict[str, Any],
        data: dict[str, Any],
        name: str = "document",
    ) -> dict[str, Any]:
        """Submit a task and poll until completion; fetch result as JSON.

        Args:
            endpoint: The async endpoint path (e.g., "/v1/convert/file/async")
            files: Files to upload
            data: Form data parameters
            name: Name for error messages

        Returns:
            The result dictionary from the completed task

        Raises:
            ValueError: If the task fails or service is unavailable
        """
        headers = self._get_headers()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                task_id = await self._submit_and_wait(
                    client, endpoint, files, data, headers, name
                )
                result_url = f"{self.base_url}/v1/result/{task_id}"
                result_response = await client.get(result_url, headers=headers)
                result_response.raise_for_status()
                return result_response.json()

        except httpx.ConnectError as e:
            raise ValueError(
                f"Could not connect to docling-serve at {self.base_url}. "
                f"Ensure the service is running and accessible. Error: {e}"
            )
        except httpx.TimeoutException as e:
            raise ValueError(
                f"Request to docling-serve timed out after {self.timeout}s. Error: {e}"
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    "Authentication failed. Check your API key configuration."
                )
            raise ValueError(f"HTTP error from docling-serve: {e}")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to process via docling-serve: {e}")

    async def submit_and_poll_zip(
        self,
        endpoint: str,
        files: dict[str, Any],
        data: dict[str, Any],
        name: str = "document",
    ) -> bytes:
        """Submit a task and poll until completion; fetch result as raw bytes.

        Used when the caller requested ``target_type=zip`` (e.g. to retrieve
        picture image bytes that docling-serve only emits as referenced files
        bundled into a zip archive). The submit/poll flow is identical to
        ``submit_and_poll``; only the result-fetching step differs.
        """
        headers = self._get_headers()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                task_id = await self._submit_and_wait(
                    client, endpoint, files, data, headers, name
                )
                result_url = f"{self.base_url}/v1/result/{task_id}"
                result_response = await client.get(result_url, headers=headers)
                result_response.raise_for_status()
                return result_response.content

        except httpx.ConnectError as e:
            raise ValueError(
                f"Could not connect to docling-serve at {self.base_url}. "
                f"Ensure the service is running and accessible. Error: {e}"
            )
        except httpx.TimeoutException as e:
            raise ValueError(
                f"Request to docling-serve timed out after {self.timeout}s. Error: {e}"
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    "Authentication failed. Check your API key configuration."
                )
            raise ValueError(f"HTTP error from docling-serve: {e}")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to process via docling-serve: {e}")
