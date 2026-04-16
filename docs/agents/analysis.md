# Analysis Agent

The analysis agent enables complex analytical tasks by writing and executing Python code in a sandboxed environment. It solves problems that traditional RAG struggles with:

- **Aggregation**: "How many documents mention security vulnerabilities?"
- **Computation**: "What's the average revenue across all quarterly reports?"
- **Multi-document analysis**: "Compare the key findings between Report A and Report B"
- **Structured data extraction**: "Extract all dollar amounts and compute totals"

## How It Works

1. The agent receives a question
2. It writes Python code to explore the knowledge base
3. Code executes in a sandboxed Python interpreter with access to search, LLM, and a virtual filesystem of documents
4. The agent iterates: run code, examine results, refine approach
5. Final answer is synthesized from the gathered data

## CLI Usage

```bash
# Basic usage
haiku-rag analyze "How many documents are in the database?"

# With document filter (restricts what the agent can access)
haiku-rag analyze "Summarize the key points" --filter "uri LIKE '%report%'"

# Pre-load specific documents
haiku-rag analyze "Compare these two reports" --document "Q1 Report" --document "Q2 Report"
```

## Python Usage

```python
from haiku.rag.client import HaikuRAG

async with HaikuRAG(path_to_db) as client:
    # Basic question
    result = await client.analyze("How many documents mention 'security'?")
    print(result.answer)    # The answer
    print(result.program)   # The final consolidated program

    # With filter (agent can only see filtered documents)
    result = await client.analyze(
        "What is the total revenue?",
        filter="title LIKE '%Financial%'"
    )

    # Pre-load specific documents
    result = await client.analyze(
        "Compare the conclusions",
        documents=["Report A", "Report B"]
    )
```

## Sandbox Capabilities

The agent's code runs in a sandboxed Python interpreter ([pydantic-monty](https://github.com/pydantic/monty)) with:

### Functions

| Function | Description |
|----------|-------------|
| `search(query, limit)` | Hybrid search (vector + full-text) with automatic context expansion. Returns `doc_item_refs` for cross-referencing with `items.jsonl` |
| `list_documents()` | List all documents in the knowledge base |
| `llm(prompt)` | Call an LLM for classification, summarization, or extraction |

### Document Filesystem

All documents are mounted as a virtual filesystem at `/documents/`. The agent uses standard Python `pathlib.Path` to browse and read files:

```
/documents/{document_id}/
    metadata.json    # {id, title, uri, created_at}
    content.txt      # Full document text
    items.jsonl      # Structured items: position, self_ref, label, text, page_numbers
```

- **`metadata.json`** — Loaded eagerly (small). Use `Path('/documents').iterdir()` to discover documents.
- **`content.txt`** — Lazy-loaded on first read. Full document text for regex or keyword search.
- **`items.jsonl`** — Lazy-loaded on first read. One JSON object per line with structured document elements. Tables are pre-rendered as markdown. Labels include `section_header`, `text`, `table`, `list_item`, `caption`, `formula`, `picture`, `code`, `footnote`, etc.

Search results include `doc_item_refs` (e.g. `["#/texts/5", "#/tables/0"]`) that match `self_ref` values in `items.jsonl`, enabling navigation from search hits to document structure.

When documents are pre-loaded via the `documents` parameter, they are also injected as a `documents` variable accessible in the sandbox code.

### Python Features

The interpreter supports a subset of Python: variables, arithmetic, strings, f-strings, lists, dicts, tuples, sets, loops, conditionals, comprehensions, functions, async/await, `filter()`, `getattr()`, try/except, file I/O via `pathlib.Path`, and the `json`, `re`, `math` modules.

Not supported: most imports (only `json`, `re`, `math`, `pathlib` are available), class definitions, generators/yield, match statements, decorators, `with` statements. For pattern matching, the agent can use `import re`, string methods, or the `llm()` function.

### Security

Code executes in an isolated interpreter with:

- **Virtual filesystem only**: The `/documents/` filesystem is sandboxed — no access to the real filesystem
- **No network access**: Code cannot make HTTP requests or open sockets
- **No imports**: Only `json`, `re`, `math`, and `pathlib` modules are available
- **Execution timeout**: Configurable limit (default 60s)
- **Output truncation**: Large outputs are truncated to prevent memory issues

## Context Filter

The `filter` parameter restricts what documents the agent can access. Unlike tool parameters, the filter is applied automatically and cannot be bypassed by the LLM — both the VFS and search results are scoped to the filter:

```python
# Agent can only see documents with "confidential" in the URI
result = await client.analyze(
    "Summarize all findings",
    filter="uri LIKE '%confidential%'"
)
```

This is useful for scoping to specific document sets, enforcing access control, or limiting context for focused analysis.

## Configuration

Analysis settings can be configured in `haiku.rag.yaml`:

```yaml
analysis:
  model:
    provider: anthropic
    name: claude-sonnet-4-20250514
  code_timeout: 60.0      # Max seconds for code execution
  max_output_chars: 50000 # Truncate output after this many chars
```
