# RLM Agent (Recursive Language Model)

The RLM agent enables complex analytical tasks by writing and executing Python code in a sandboxed environment. It solves problems that traditional RAG struggles with:

- **Aggregation**: "How many documents mention security vulnerabilities?"
- **Computation**: "What's the average revenue across all quarterly reports?"
- **Multi-document analysis**: "Compare the key findings between Report A and Report B"
- **Structured data extraction**: "Extract all dollar amounts and compute totals"

## How It Works

1. The agent receives a question
2. It writes Python code to explore the knowledge base
3. Code executes in a sandboxed Python interpreter with access to knowledge base functions
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

## Sandbox Capabilities

The agent's code runs in a sandboxed Python interpreter ([pydantic-monty](https://github.com/pydantic/monty)) with access to these knowledge base functions:

| Function | Description |
|----------|-------------|
| `search(query, limit)` | Hybrid search (vector + full-text) returning matching chunks with scores |
| `list_documents(limit, offset)` | List documents in the knowledge base |
| `get_document(id_or_title)` | Get full text content of a document |
| `get_chunk(chunk_id)` | Get a chunk with metadata (headings, page numbers, labels) for citations |
| `get_docling_document(document_id)` | Get the full DoclingDocument structure as a dict (texts, tables, pictures, pages) |
| `llm(prompt)` | Call an LLM for classification, summarization, or extraction |

When documents are pre-loaded via the `documents` parameter, they are injected as a `documents` variable accessible in the sandbox code.

### Python Features

The interpreter supports a subset of Python: variables, arithmetic, strings, f-strings, lists, dicts, tuples, sets, loops, conditionals, comprehensions, functions, async/await, try/except, and the `json` module.

Not supported: imports (other than `json`), class definitions, generators/yield, match statements, decorators, `with` statements. For pattern matching, the agent can use string methods or the `llm()` function.

### Security

Code executes in an isolated interpreter with:

- **No filesystem access**: Code cannot read or write files
- **No network access**: Code cannot make HTTP requests or open sockets
- **No imports**: Only the `json` module is available
- **Execution timeout**: Configurable limit (default 60s)
- **Output truncation**: Large outputs are truncated to prevent memory issues

## Context Filter

The `filter` parameter restricts what documents the agent can access. Unlike tool parameters, the filter is applied automatically and cannot be bypassed by the LLM:

```python
# Agent can only see documents with "confidential" in the URI
result = await client.rlm(
    "Summarize all findings",
    filter="uri LIKE '%confidential%'"
)
```

This is useful for scoping to specific document sets, enforcing access control, or limiting context for focused analysis.

## Configuration

RLM settings can be configured in `haiku.rag.yaml`:

```yaml
rlm:
  model:
    provider: anthropic
    name: claude-sonnet-4-20250514
  code_timeout: 60.0      # Max seconds for code execution
  max_output_chars: 50000 # Truncate output after this many chars
```
