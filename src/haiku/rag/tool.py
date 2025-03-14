from ollama._types import Tool

from haiku.rag.store.search import search

RAGTool = Tool(
    type="function",
    function=Tool.Function(
        name="rag",
        description="Function to search the RAG knowledge base. Returns chunks of documents relevant to the query.",
        parameters=Tool.Function.Parameters(
            type="object",
            properties={
                "query": Tool.Function.Parameters.Property(
                    type="string", description="The query to execute."
                )
            },
            required=["query"],
        ),
    ),
)


async def rag_command(query: str) -> str:
    results = await search(query, top_k=3)
    response = "\n".join([chunk.text for chunk, _ in results])
    return response
