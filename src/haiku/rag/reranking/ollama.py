import json

from ollama import AsyncClient
from pydantic import BaseModel

from haiku.rag.config import Config
from haiku.rag.reranking.base import RerankerBase
from haiku.rag.store.models.chunk import Chunk

OLLAMA_OPTIONS = {"temperature": 0.0, "seed": 42, "num_ctx": 16384}


class RerankResult(BaseModel):
    """Individual rerank result with index and relevance score."""

    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    """Response from the reranking model containing ranked results."""

    results: list[RerankResult]


class OllamaReranker(RerankerBase):
    def __init__(self, model: str = Config.RERANK_MODEL):
        self._model = model
        self._client = AsyncClient(host=Config.OLLAMA_BASE_URL)

    async def rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []

        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({"index": i, "content": chunk.content})

        # Create the prompt for reranking
        system_prompt = """You are a document reranking assistant. Given a query and a list of document chunks, you must rank them by relevance to the query.

Return your response as a JSON object with a "results" array. Each result should have:
- "index": the original index of the document (integer)
- "relevance_score": a score between 0.0 and 1.0 indicating relevance (float, where 1.0 is most relevant)

Only return the top documents up to the requested limit, ordered by decreasing relevance score."""

        documents_text = ""
        for doc in documents:
            documents_text += f"Index {doc['index']}: {doc['content']}\n\n"

        user_prompt = f"""Query: {query}

Documents to rerank:
{documents_text.strip()}

Please rank these documents by relevance to the query and return the top {top_n} results as JSON."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self._client.chat(
                model=self._model,
                messages=messages,
                format=RerankResponse.model_json_schema(),
                options=OLLAMA_OPTIONS,
            )

            content = response["message"]["content"]

            parsed_response = RerankResponse.model_validate(json.loads(content))
            return [
                (chunks[result.index], result.relevance_score)
                for result in parsed_response.results[:top_n]
            ]

        except Exception:
            # Fallback: return chunks in original order with same score
            return [(chunks[i], 1.0) for i in range(min(top_n, len(chunks)))]
