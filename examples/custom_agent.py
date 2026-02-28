"""Custom agent using the haiku.rag RAG skill.

Demonstrates how to use the RAG skill with haiku.skills SkillToolset
to build a conversational agent.

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

from haiku.rag.skills.rag import create_skill
from haiku.skills.agent import SkillToolset
from haiku.skills.prompts import build_system_prompt


async def main(db_path: str) -> None:
    skill = create_skill(db_path=Path(db_path))
    toolset = SkillToolset(skills=[skill])

    agent = Agent(
        "anthropic:claude-haiku-4-5-20251001",
        instructions=build_system_prompt(toolset.skill_catalog),
        toolsets=[toolset],
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
