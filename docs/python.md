# Python API

Use `haiku.rag` directly in your Python applications.

## Basic Usage

```python
from pathlib import Path
from haiku.rag.client import HaikuRAG

# Use as async context manager (recommended)
async with HaikuRAG("path/to/database.db") as client:
    # Your code here
    pass
```

## Document Management

### Creating Documents

From text:
```python
doc = await client.create_document(
    content="Your document content here",
    uri="doc://example",
    metadata={"source": "manual", "topic": "example"}
)
```

With custom externally generated chunks:
```python
from haiku.rag.store.models.chunk import Chunk

# Create custom chunks with optional embeddings
chunks = [
    Chunk(
        content="This is the first chunk",
        metadata={"section": "intro"}
    ),
    Chunk(
        content="This is the second chunk",
        metadata={"section": "body"},
        embedding=[0.1] * 1024  # Optional pre-computed embedding
    ),
]

doc = await client.create_document(
    content="Full document content",
    uri="doc://custom",
    metadata={"source": "manual"},
    chunks=chunks  # Use provided chunks instead of auto-generating
)
```

From file:
```python
doc = await client.create_document_from_source("path/to/document.pdf")
```

From URL:
```python
doc = await client.create_document_from_source("https://example.com/article.html")
```

### Retrieving Documents

By ID:
```python
doc = await client.get_document_by_id(1)
```

By URI:
```python
doc = await client.get_document_by_uri("file:///path/to/document.pdf")
```

List all documents:
```python
docs = await client.list_documents(limit=10, offset=0)
```

### Updating Documents

```python
doc.content = "Updated content"
await client.update_document(doc)
```

### Deleting Documents

```python
await client.delete_document(doc.id)
```

### Rebuilding the Database

```python
async for doc_id in client.rebuild_database():
    print(f"Processed document {doc_id}")
```

## Searching Documents

The search method performs hybrid search (vector + full-text) with **reranking enabled by default** for improved relevance:

Basic search (with reranking):
```python
results = await client.search("machine learning algorithms", limit=5)
for chunk, score in results:
    print(f"Score: {score:.3f}")
    print(f"Content: {chunk.content}")
    print(f"Document ID: {chunk.document_id}")
```

With options:
```python
results = await client.search(
    query="machine learning",
    limit=5,  # Maximum results to return
    k=60,     # RRF parameter for reciprocal rank fusion
    rerank=False  # Disable reranking for faster search
)

# Process results
for chunk, relevance_score in results:
    print(f"Relevance: {relevance_score:.3f}")
    print(f"Content: {chunk.content}")
    print(f"From document: {chunk.document_id}")
    print(f"Document URI: {chunk.document_uri}")
    print(f"Document metadata: {chunk.document_meta}")
```

### Expanding Search Context

Expand search results with adjacent chunks for more complete context:

```python
# Get initial search results
search_results = await client.search("machine learning", limit=3)

# Expand with adjacent chunks using config setting
expanded_results = await client.expand_context(search_results)

# Or specify a custom radius
expanded_results = await client.expand_context(search_results, radius=2)

# The expanded results contain chunks with combined content from adjacent chunks
for chunk, score in expanded_results:
    print(f"Expanded content: {chunk.content}")  # Now includes before/after chunks
```

**Smart Merging**: When expanded chunks overlap or are adjacent within the same document, they are automatically merged into single chunks with continuous content. This eliminates duplication and provides coherent text blocks. The merged chunk uses the highest relevance score from the original chunks.

This is automatically used by the QA system when `CONTEXT_CHUNK_RADIUS > 0` to provide better answers with more complete context.

## Question Answering

Ask questions about your documents:

```python
answer = await client.ask("Who is the author of haiku.rag?")
print(answer)
```

Ask questions with citations showing source documents:

```python
answer = await client.ask("Who is the author of haiku.rag?", cite=True)
print(answer)
```

The QA agent will search your documents for relevant information and use the configured LLM to generate a comprehensive answer. With `cite=True`, responses include citations showing which documents were used as sources.

The QA provider and model can be configured via environment variables (see [Configuration](configuration.md)).
