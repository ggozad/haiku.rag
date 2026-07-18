"""Custom agent using the native haiku.rag RAG capability.

Demonstrates composing a native Pydantic AI capability into an agent.

Requirements:
    - An Ollama instance running locally (default embedder)
    - An Anthropic API key (for the QA model) or adjust the model below
    - A haiku.rag database with documents already ingested

Usage:
    uv run python examples/custom_agent.py /path/to/db.lancedb
"""

import asyncio
import sys
from pathlib import Path

from pydantic_ai import Agent

from haiku.rag.capabilities.rag import create_capability


async def main(db_path: str) -> None:
    capability = create_capability(db_path=Path(db_path), defer_loading=False)

    agent = Agent(
        "anthropic:claude-haiku-4-5-20251001",
        capabilities=[capability],
    )

    print("Custom agent ready. Ctrl+C to exit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        result = await agent.run(user_input)
        print(f"\nAgent: {result.output}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <db_path>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
