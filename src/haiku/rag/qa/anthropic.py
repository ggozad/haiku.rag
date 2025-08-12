from collections.abc import Sequence

try:
    from anthropic import AsyncAnthropic  # type: ignore
    from anthropic.types import (  # type: ignore
        MessageParam,
        TextBlock,
        ToolParam,
        ToolUseBlock,
    )

    from haiku.rag.client import HaikuRAG
    from haiku.rag.qa.base import QuestionAnswerAgentBase

    class QuestionAnswerAnthropicAgent(QuestionAnswerAgentBase):
        def __init__(
            self,
            client: HaikuRAG,
            model: str = "claude-3-5-haiku-20241022",
            use_citations: bool = False,
        ):
            super().__init__(client, model or self._model, use_citations)
            self.tools: Sequence[ToolParam] = [
                ToolParam(
                    name="search_documents",
                    description="Search the knowledge base for relevant documents. Returns a JSON array with content, score, and document_uri for each result.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant documents",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 3,
                            },
                        },
                        "required": ["query"],
                    },
                )
            ]

        async def answer(self, question: str) -> str:
            anthropic_client = AsyncAnthropic()

            messages: list[MessageParam] = [{"role": "user", "content": question}]

            max_rounds = 5  # Prevent infinite loops

            for _ in range(max_rounds):
                response = await anthropic_client.messages.create(
                    model=self._model,
                    max_tokens=4096,
                    system=self._system_prompt,
                    messages=messages,
                    tools=self.tools,
                    temperature=0.0,
                )

                if response.stop_reason == "tool_use":
                    messages.append({"role": "assistant", "content": response.content})

                    # Process tool calls
                    tool_results = []
                    for content_block in response.content:
                        if isinstance(content_block, ToolUseBlock):
                            if content_block.name == "search_documents":
                                args = content_block.input
                                query = (
                                    args.get("query", question)
                                    if isinstance(args, dict)
                                    else question
                                )
                                limit = (
                                    int(args.get("limit", 3))
                                    if isinstance(args, dict)
                                    else 3
                                )

                                context = await self._search_and_expand(
                                    query, limit=limit
                                )

                                tool_results.append(
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": content_block.id,
                                        "content": context,
                                    }
                                )

                    if tool_results:
                        messages.append({"role": "user", "content": tool_results})
                else:
                    # No tool use, return the response
                    if response.content:
                        first_content = response.content[0]
                        if isinstance(first_content, TextBlock):
                            return first_content.text
                    return ""

            # If we've exhausted max rounds, return empty string
            return ""

except ImportError:
    pass
