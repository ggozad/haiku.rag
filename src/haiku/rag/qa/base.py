import json

from haiku.rag.client import HaikuRAG
from haiku.rag.qa.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_CITATIONS


class QuestionAnswerAgentBase:
    _model: str = ""
    _system_prompt: str = SYSTEM_PROMPT

    def __init__(self, client: HaikuRAG, model: str = "", use_citations: bool = False):
        self._model = model
        self._client = client
        self._system_prompt = (
            SYSTEM_PROMPT_WITH_CITATIONS if use_citations else SYSTEM_PROMPT
        )

    async def answer(self, question: str) -> str:
        raise NotImplementedError(
            "QABase is an abstract class. Please implement the answer method in a subclass."
        )

    async def _search_and_expand(self, query: str, limit: int = 3) -> str:
        """Search for documents and expand context, then format as JSON"""
        search_results = await self._client.search(query, limit=limit)
        expanded_results = await self._client.expand_context(search_results)
        return self._format_search_results(expanded_results)

    def _format_search_results(self, search_results) -> str:
        """Format search results as JSON list of {content, score, document_uri}"""
        formatted_results = []
        for chunk, score in search_results:
            formatted_results.append(
                {
                    "content": chunk.content,
                    "score": score,
                    "document_uri": chunk.document_uri,
                }
            )
        return json.dumps(formatted_results, indent=2)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Search the knowledge base for relevant documents. Returns a JSON array of search results.",
                "parameters": {
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
                "returns": {
                    "type": "string",
                    "description": "JSON array of search results",
                    "schema": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "The document text content",
                                },
                                "score": {
                                    "type": "number",
                                    "description": "Relevance score (higher is more relevant)",
                                },
                                "document_uri": {
                                    "type": "string",
                                    "description": "Source URI/path of the document",
                                },
                            },
                        },
                    },
                },
            },
        }
    ]
