from ollama import AsyncClient

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.qa.base import QuestionAnswerAgentBase

OLLAMA_OPTIONS = {"temperature": 0.0, "seed": 42, "num_ctx": 16384}


class QuestionAnswerOllamaAgent(QuestionAnswerAgentBase):
    def __init__(self, client: HaikuRAG, model: str = Config.QA_MODEL):
        super().__init__(client, model or self._model)

    async def answer(self, question: str) -> str:
        ollama_client = AsyncClient(host=Config.OLLAMA_BASE_URL)

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": question},
        ]

        max_rounds = 5  # Prevent infinite loops

        for _ in range(max_rounds):
            response = await ollama_client.chat(
                model=self._model,
                messages=messages,
                tools=self.tools,
                options=OLLAMA_OPTIONS,
                think=False,
            )

            if response.get("message", {}).get("tool_calls"):
                messages.append(response["message"])

                for tool_call in response["message"]["tool_calls"]:
                    if tool_call["function"]["name"] == "search_documents":
                        args = tool_call["function"]["arguments"]
                        query = args.get("query", question)
                        limit = int(args.get("limit", 3))

                        search_results = await self._client.search(query, limit=limit)

                        context_chunks = []
                        for chunk, score in search_results:
                            context_chunks.append(
                                f"Content: {chunk.content}\nScore: {score:.4f}"
                            )

                        context = "\n\n".join(context_chunks)

                        messages.append(
                            {
                                "role": "tool",
                                "content": context,
                                "tool_call_id": tool_call.get("id", "search_tool"),
                            }
                        )
            else:
                # No tool calls, return the response
                return response["message"]["content"]

        # If we've exhausted max rounds, return empty string
        return ""
