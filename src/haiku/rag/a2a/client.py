import asyncio
import uuid
from typing import Any

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

try:
    from fasta2a.client import A2AClient as FastA2AClient
    from fasta2a.schema import Message, TextPart
except ImportError as e:
    raise ImportError(
        "A2A support requires the 'a2a' extra. "
        "Install with: uv pip install 'haiku.rag[a2a]'"
    ) from e


class A2AClient:
    """Interactive A2A protocol client."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize A2A client.

        Args:
            base_url: Base URL of the A2A server
        """
        self.base_url = base_url.rstrip("/")
        http_client = httpx.AsyncClient(timeout=60.0)
        self._client = FastA2AClient(base_url=base_url, http_client=http_client)

    async def close(self):
        """Close the HTTP client."""
        await self._client.http_client.aclose()

    async def get_agent_card(self) -> dict[str, Any]:
        """Fetch the agent card from the A2A server.

        Returns:
            Agent card dictionary with agent capabilities and metadata
        """
        response = await self._client.http_client.get(
            f"{self.base_url}/.well-known/agent-card.json"
        )
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

        message = Message(
            kind="message",
            role="user",
            message_id=str(uuid.uuid4()),
            parts=[TextPart(kind="text", text=text)],
        )

        metadata: dict[str, Any] = {"contextId": context_id}
        if skill_id:
            metadata["skillId"] = skill_id

        response = await self._client.send_message(message, metadata=metadata)

        if "error" in response:
            return {"error": response["error"]}

        result = response.get("result")
        if not result:
            return {"result": result}

        # Result can be either Task or Message - check if it's a Task with an id
        if result.get("kind") == "task":
            task_id = result.get("id")
            if task_id:
                # Poll for task completion
                return await self.wait_for_task(task_id)

        # Return the message directly
        return {"result": result}

    async def wait_for_task(
        self, task_id: str, max_wait: int = 120, poll_interval: float = 0.5
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
            task_response = await self._client.get_task(task_id)

            if "error" in task_response:
                return {"error": task_response["error"]}

            task = task_response.get("result")
            if not task:
                raise Exception("No task in response")

            state = task.get("status", {}).get("state")

            if state == "completed":
                return {"result": task}
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
