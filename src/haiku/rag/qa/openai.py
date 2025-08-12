from collections.abc import Sequence

try:
    from openai import AsyncOpenAI  # type: ignore
    from openai.types.chat import (  # type: ignore
        ChatCompletionAssistantMessageParam,
        ChatCompletionMessageParam,
        ChatCompletionSystemMessageParam,
        ChatCompletionToolMessageParam,
        ChatCompletionUserMessageParam,
    )
    from openai.types.chat.chat_completion_tool_param import (  # type: ignore
        ChatCompletionToolParam,
    )

    from haiku.rag.client import HaikuRAG
    from haiku.rag.qa.base import QuestionAnswerAgentBase

    class QuestionAnswerOpenAIAgent(QuestionAnswerAgentBase):
        def __init__(
            self,
            client: HaikuRAG,
            model: str = "gpt-4o-mini",
            use_citations: bool = False,
        ):
            super().__init__(client, model or self._model, use_citations)
            self.tools: Sequence[ChatCompletionToolParam] = [
                ChatCompletionToolParam(tool) for tool in self.tools
            ]

        async def answer(self, question: str) -> str:
            openai_client = AsyncOpenAI()

            messages: list[ChatCompletionMessageParam] = [
                ChatCompletionSystemMessageParam(
                    role="system", content=self._system_prompt
                ),
                ChatCompletionUserMessageParam(role="user", content=question),
            ]

            max_rounds = 5  # Prevent infinite loops

            for _ in range(max_rounds):
                response = await openai_client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=self.tools,
                    temperature=0.0,
                )

                response_message = response.choices[0].message

                if response_message.tool_calls:
                    messages.append(
                        ChatCompletionAssistantMessageParam(
                            role="assistant",
                            content=response_message.content,
                            tool_calls=[
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in response_message.tool_calls
                            ],
                        )
                    )

                    for tool_call in response_message.tool_calls:
                        if tool_call.function.name == "search_documents":
                            import json

                            args = json.loads(tool_call.function.arguments)
                            query = args.get("query", question)
                            limit = int(args.get("limit", 3))

                            context = await self._search_and_expand(query, limit=limit)

                            messages.append(
                                ChatCompletionToolMessageParam(
                                    role="tool",
                                    content=context,
                                    tool_call_id=tool_call.id,
                                )
                            )
                else:
                    # No tool calls, return the response
                    return response_message.content or ""

            # If we've exhausted max rounds, return empty string
            return ""

except ImportError:
    pass
