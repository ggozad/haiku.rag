"""Custom agent using the native haiku.rag RAG capability.

Demonstrates composing native Pydantic AI capabilities into an agent, and what a
multi-turn conversation needs to carry between runs.

Requirements:
    - An Ollama instance running locally (default embedder)
    - An Anthropic API key (for the QA model) or adjust the model below
    - A haiku.rag database with documents already ingested

Usage:
    uv run python examples/custom_agent.py /path/to/db.lancedb
"""

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from haiku.rag.capabilities.compaction import create_capability as compaction
from haiku.rag.capabilities.policy import create_capability as citation_policy
from haiku.rag.capabilities.rag import create_capability as rag


@dataclass
class Deps:
    state: dict[str, Any] = field(default_factory=dict)


async def main(db_path: str) -> None:
    agent = Agent(
        "anthropic:claude-haiku-4-5-20251001",
        capabilities=[
            rag(db_path=Path(db_path), defer_loading=False),
            compaction(),
            citation_policy(),
        ],
        deps_type=Deps,
    )

    # One state dict and one history for the whole session. The capabilities read
    # both: the state holds what was retrieved and cited, and the message counts
    # are how they tell one question from the next.
    deps = Deps()
    messages: list[ModelMessage] = []

    print("Custom agent ready. Ctrl+C to exit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        result = await agent.run(user_input, deps=deps, message_history=messages)
        messages = list(result.all_messages())
        print(f"\nAgent: {result.output}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <db_path>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
