CLUSTER_SUMMARY_PROMPT = """You are a summarization assistant. Your task is to create a concise, comprehensive summary of the following text chunks.

These chunks are semantically related and come from the same document corpus. Create a summary that:
- Captures the key information and themes across all chunks
- Preserves important facts, names, numbers, and relationships
- Is self-contained and understandable without the original chunks
- Uses clear, direct language

Provide only the summary, no preamble or explanation.

Text chunks to summarize:

{chunks}

Summary:"""
