"""Custom agent using haiku.rag composable toolsets.

Demonstrates how to compose search, QA, and document toolsets into a
pydantic-ai Agent using AgentDeps and prepare_context.

Requirements:
    - An Ollama instance running locally (default embedder)
    - An Anthropic API key (for the QA model) or adjust the model below
    - A haiku.rag database with documents already ingested

Usage:
    uv run python examples/custom_agent.py /path/to/db.lancedb
"""

import asyncio
import sys

from pydantic_ai import Agent

from haiku.rag.client import HaikuRAG
from haiku.rag.tools import (
    AgentDeps,
    ToolContext,
    build_tools_prompt,
    create_document_toolset,
    create_qa_toolset,
    create_search_toolset,
    prepare_context,
)


async def main(db_path: str) -> None:
    async with HaikuRAG(db_path) as client:
        # Compose toolsets into an agent
        config = client.config
        search_toolset = create_search_toolset(config)
        qa_toolset = create_qa_toolset(config)
        document_toolset = create_document_toolset(config)

        features = ["search", "documents", "qa"]
        tools_prompt = build_tools_prompt(features)

        agent = Agent(
            "anthropic:claude-haiku-4-5-20251001",
            deps_type=AgentDeps,
            output_type=str,
            instructions=(
                "You are a helpful assistant with access to a knowledge base.\n"
                f"{tools_prompt}"
            ),
            toolsets=[search_toolset, qa_toolset, document_toolset],
        )

        # Prepare a shared ToolContext
        context = ToolContext()
        prepare_context(context, features=["search", "documents", "qa"])

        deps = AgentDeps(client=client, tool_context=context)

        print("Custom agent ready. Ctrl+C to exit.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            result = await agent.run(user_input, deps=deps)
            print(f"\nAgent: {result.output}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <db_path>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
