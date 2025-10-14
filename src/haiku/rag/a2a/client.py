import asyncio
import uuid
from typing import Any

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt


class A2AClient:
    """Simple A2A protocol client."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize A2A client.

        Args:
            base_url: Base URL of the A2A server
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=60.0)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def get_agent_card(self) -> dict[str, Any]:
        """Fetch the agent card from the A2A server.

        Returns:
            Agent card dictionary with agent capabilities and metadata
        """
        response = await self.client.get(f"{self.base_url}/.well-known/agent-card.json")
        response.raise_for_status()
        return response.json()

    async def send_message(
        self,
        text: str,
        context_id: str | None = None,
        skill_id: str | None = None,
    ) -> dict[str, Any]:
        """Send a message to the A2A agent and wait for completion.

        Args:
            text: Message text to send
            context_id: Optional conversation context ID (creates new if None)
            skill_id: Optional skill ID to use (defaults to document-qa)

        Returns:
            Completed task with response messages and artifacts
        """
        if context_id is None:
            context_id = str(uuid.uuid4())

        message_id = str(uuid.uuid4())

        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "contextId": context_id,
                "message": {
                    "kind": "message",
                    "role": "user",
                    "messageId": message_id,
                    "parts": [{"kind": "text", "text": text}],
                },
            },
            "id": 1,
        }

        if skill_id:
            payload["params"]["skillId"] = skill_id

        response = await self.client.post(
            self.base_url,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        initial_response = response.json()

        # Extract task ID from response
        result = initial_response.get("result", {})
        task_id = result.get("id")

        if not task_id:
            return initial_response

        # Poll for task completion
        return await self.wait_for_task(task_id)

    async def wait_for_task(
        self, task_id: str, max_wait: int = 60, poll_interval: float = 0.5
    ) -> dict[str, Any]:
        """Poll for task completion.

        Args:
            task_id: Task ID to poll for
            max_wait: Maximum time to wait in seconds
            poll_interval: Interval between polls in seconds

        Returns:
            Completed task result
        """
        import time

        start_time = time.time()

        while time.time() - start_time < max_wait:
            payload = {
                "jsonrpc": "2.0",
                "method": "tasks/get",
                "params": {"id": task_id},
                "id": 2,
            }

            response = await self.client.post(
                self.base_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            task = response.json()

            result = task.get("result", {})
            status = result.get("status", {})
            state = status.get("state")

            if state == "completed":
                return task
            elif state == "failed":
                raise Exception(f"Task failed: {task}")

            await asyncio.sleep(poll_interval)

        raise TimeoutError(f"Task {task_id} did not complete within {max_wait}s")


def print_agent_card(card: dict[str, Any], console: Console):
    """Pretty print the agent card using Rich."""
    console.print()
    console.print("[bold]Agent Card[/bold]")
    console.rule()

    console.print(f"  [repr.attrib_name]name[/repr.attrib_name]: {card.get('name')}")
    console.print(
        f"  [repr.attrib_name]description[/repr.attrib_name]: {card.get('description')}"
    )
    console.print(
        f"  [repr.attrib_name]version[/repr.attrib_name]: {card.get('version')}"
    )
    console.print(
        f"  [repr.attrib_name]protocol version[/repr.attrib_name]: {card.get('protocolVersion')}"
    )

    skills = card.get("skills", [])
    console.print(f"\n[bold cyan]Skills ({len(skills)}):[/bold cyan]")
    for skill in skills:
        console.print(f"  • {skill.get('id')}: {skill.get('name')}")
        console.print(f"    [dim]{skill.get('description')}[/dim]")
        examples = skill.get("examples", [])
        if examples:
            console.print(f"    [dim]Examples: {', '.join(examples[:2])}[/dim]")
    console.print()


def print_response(response: dict[str, Any], console: Console):
    """Pretty print the A2A response using Rich."""
    if "error" in response:
        console.print(f"[red]Error: {response['error']}[/red]")
        return

    result = response.get("result", {})

    # Get messages from history and artifacts from completed task
    history = result.get("history", [])
    artifacts = result.get("artifacts", [])

    # Print agent messages from history with markdown rendering
    for msg in history:
        if msg.get("role") == "agent":
            for part in msg.get("parts", []):
                if part.get("kind") == "text":
                    text = part.get("text", "")
                    # Render as markdown
                    console.print()
                    console.print("[bold green]Answer:[/bold green]")
                    console.print(Markdown(text))

    # Print artifacts summary with details
    if artifacts:
        summary_lines = []

        for artifact in artifacts:
            name = artifact.get("name", "")
            parts = artifact.get("parts", [])

            if name == "search_results" and parts:
                data = parts[0].get("data", {})
                query = data.get("query", "")
                results = data.get("results", [])
                summary_lines.append(f"🔍 search: '{query}' ({len(results)} results)")

            elif name == "document" and parts:
                part = parts[0]
                if part.get("kind") == "text":
                    text = part.get("text", "")
                    length = len(text)
                    summary_lines.append(f"📄 document ({length} chars)")

            elif name == "qa_result" and parts:
                data = parts[0].get("data", {})
                skill = data.get("skill", "unknown")
                summary_lines.append(f"💬 {skill}")

        if summary_lines:
            console.print(f"[dim]{' • '.join(summary_lines)}[/dim]")

    console.print()


async def run_interactive_client(url: str = "http://localhost:8000"):
    """Run the interactive A2A client.

    Args:
        url: Base URL of the A2A server
    """
    console = Console()
    client = A2AClient(url)

    console.print("[bold]haiku.rag A2A interactive client[/bold]")
    console.print()

    # Fetch and display agent card
    console.print("[dim]Fetching agent card...[/dim]")
    try:
        card = await client.get_agent_card()
        print_agent_card(card, console)
    except Exception as e:
        console.print(f"[red]Error fetching agent card: {e}[/red]")
        await client.close()
        return

    # Create a conversation context
    context_id = str(uuid.uuid4())
    console.print(f"[dim]context id: {context_id}[/dim]")
    console.print("[dim]Type your questions (or 'quit' to exit)[/dim]\n")

    try:
        while True:
            try:
                question = Prompt.ask("[bold blue]Question[/bold blue]").strip()
                if not question:
                    continue

                if question.lower() in ("quit", "exit", "q"):
                    console.print("\n[dim]Goodbye![/dim]")
                    break

                response = await client.send_message(question, context_id=context_id)
                print_response(response, console)

            except KeyboardInterrupt:
                console.print("\n\n[dim]Exiting...[/dim]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]\n")
    finally:
        await client.close()
