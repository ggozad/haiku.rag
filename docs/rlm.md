# RLM Agent (Recursive Language Model)

The RLM agent enables complex analytical tasks by writing and executing Python code in a sandboxed environment. It solves problems that traditional RAG struggles with:

- **Aggregation**: "How many documents mention security vulnerabilities?"
- **Computation**: "What's the average revenue across all quarterly reports?"
- **Multi-document analysis**: "Compare the key findings between Report A and Report B"
- **Structured data extraction**: "Extract all tables from the document and summarize them"

## How It Works

1. The agent receives a question
2. It writes Python code to explore the knowledge base
3. Code executes in a sandboxed environment with access to haiku.rag functions
4. The agent iterates: run code, examine results, refine approach
5. Final answer is synthesized from the gathered data

## CLI Usage

```bash
# Basic usage
haiku-rag rlm "How many documents are in the database?"

# With document filter (restricts what the agent can access)
haiku-rag rlm "Summarize the key points" --filter "uri LIKE '%report%'"

# Pre-load specific documents
haiku-rag rlm "Compare these two reports" --document "Q1 Report" --document "Q2 Report"
```

## Python Usage

```python
from haiku.rag.client import HaikuRAG

async with HaikuRAG(path_to_db) as client:
    # Basic question
    result = await client.rlm("How many documents mention 'security'?")
    print(result.answer)    # The answer
    print(result.program)   # The final consolidated program

    # With filter (agent can only see filtered documents)
    result = await client.rlm(
        "What is the total revenue?",
        filter="title LIKE '%Financial%'"
    )

    # Pre-load specific documents
    result = await client.rlm(
        "Compare the conclusions",
        documents=["Report A", "Report B"]
    )
```

## Available Functions

Inside the sandbox, these functions are available (no imports needed):

### search(query, limit=10)

Search the knowledge base using hybrid search (vector + full-text).

```python
results = search("climate change impacts", limit=20)
for r in results:
    print(r['document_title'], r['score'])
    print(r['content'][:200])
```

Returns list of dicts with keys: `chunk_id`, `content`, `document_id`, `document_title`, `document_uri`, `score`, `page_numbers`, `headings`

### list_documents(limit=10, offset=0)

List available documents in the knowledge base.

```python
docs = list_documents(limit=100)
for doc in docs:
    print(doc['id'], doc['title'])
```

Returns list of dicts with keys: `id`, `title`, `uri`, `created_at`

### get_document(id_or_title)

Get the full text content of a document by ID, title, or URI.

```python
content = get_document("Q1 Report")
if content:
    print(len(content), "characters")
```

Returns the document content as a string, or `None` if not found.

### get_docling_document(id_or_title)

Get the structured DoclingDocument object for advanced analysis of tables, figures, and document structure.

```python
doc = get_docling_document("Technical Manual")
if doc:
    print(f"Tables: {len(doc.tables)}")
    print(f"Pictures: {len(doc.pictures)}")

    # Extract table data
    for table in doc.tables:
        for cell in table.data.table_cells:
            print(f"Row {cell.start_row_offset_idx}, Col {cell.start_col_offset_idx}: {cell.text}")
```

### llm(prompt)

Call an LLM directly for classification, summarization, or extraction tasks.

```python
content = get_document("Q1 Report")
sentiment = llm(f"Classify the sentiment as positive, negative, or mixed: {content}")
print(sentiment)
```

Use this when you have content and need LLM reasoning without RAG search.

## Pre-loaded Documents

When documents are pre-loaded via the `documents` parameter, they're available as a `documents` variable:

```python
# Available when documents are pre-loaded
for doc in documents:
    print(doc['title'], len(doc['content']))
```

Each document dict has keys: `id`, `title`, `uri`, `content`

## Imports

The sandbox runs in a Docker container with full Python available. Any module installed in the container image can be imported:

```python
import re
import json
from collections import Counter

# Extract and count patterns
results = search("error", limit=50)
error_types = []
for r in results:
    matches = re.findall(r'Error: (\w+)', r['content'])
    error_types.extend(matches)

print(Counter(error_types).most_common(10))
```

The default image (`ghcr.io/ggozad/haiku.rag-slim`) includes the Python standard library. Custom images can add additional packages like `pandas` or `numpy`.

## Docker Sandbox

Code executes in an isolated Docker container with:

- **Read-only database**: The LanceDB database is mounted read-only
- **Memory limits**: Configurable memory limit (default 512MB)
- **Execution timeout**: Code times out after configurable limit (default 60s)
- **Output truncation**: Large outputs are truncated to prevent memory issues
- **Container reuse**: Within a single `rlm()` call, the container stays warm for multiple code executions

## Context Filter

The `filter` parameter restricts what documents the agent can access. Unlike tool parameters, the filter is applied automatically and cannot be bypassed by the LLM:

```python
# Agent can only see documents with "confidential" in the URI
result = await client.rlm(
    "Summarize all findings",
    filter="uri LIKE '%confidential%'"
)
```

This is useful for:

- Scoping to specific document sets
- Enforcing access control
- Limiting context for focused analysis

## Configuration

RLM settings can be configured in `haiku.rag.yaml`:

```yaml
rlm:
  model:
    provider: anthropic
    name: claude-sonnet-4-20250514
  code_timeout: 60.0      # Max seconds for code execution
  max_tool_calls: 20      # Max execute_code calls per question
  max_output_chars: 50000 # Truncate output after this many chars
  docker_image: "ghcr.io/ggozad/haiku.rag-slim:latest"  # Container image
  docker_memory_limit: "512m"  # Container memory limit
```

### Custom Docker Image

To add additional Python packages, create a custom Dockerfile:

```dockerfile
FROM ghcr.io/ggozad/haiku.rag-slim:latest
RUN pip install pandas numpy
```

Build and configure:

```bash
docker build -t my-rlm-image .
```

```yaml
rlm:
  docker_image: "my-rlm-image"
```
