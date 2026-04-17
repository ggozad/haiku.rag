---
name: rag-analysis
description: >
  Computational analysis of the knowledge base via code execution in a sandboxed Python interpreter.
  Use for questions requiring counting, aggregation, statistics, data traversal,
  comparison across documents, or any task best answered by writing Python code.
  Examples: "how many pages?", "compare table 3 across documents",
  "calculate average word count", "extract all email addresses".
---

# Analysis

You solve complex analytical questions by writing and executing Python code against the knowledge base.

## Tools

### execute_code
Execute Python code in a sandboxed interpreter. Each call runs in a fresh interpreter — variables do not persist between calls. Use `print()` to output results.

Inside the code, these functions are available (use `await`):
- `await search(query, limit=10)` → list of dicts with keys: chunk_id, content, document_id, document_title, document_uri, score, page_numbers, headings, doc_item_refs, labels
- `await list_documents()` → list of dicts with keys: id, title, uri, created_at
- `await llm(prompt)` → string response from an LLM (for classification, summarization, extraction)

Available modules: `json`, `re`, `math`, `pathlib`
Not supported: class definitions, generators/yield, match statements, decorators, `with` statements

### search
Search the knowledge base directly (outside code execution). Use for initial exploration before writing code.

### list_documents
List available documents. Use to discover what's in the knowledge base.

### cite
Register chunk IDs as citations. Call after your analysis with chunk_id values from search results that support your answer.

## Document Filesystem (inside execute_code)

All documents are mounted as a virtual filesystem at `/documents/`:

```
/documents/{document_id}/
    metadata.json    # {"id", "title", "uri", "created_at"}
    content.txt      # Full document text
    items.jsonl      # Structured items (one JSON object per line)
```

### metadata.json
Document metadata. Use `Path('/documents').iterdir()` to discover documents.

### content.txt
Full text content. Use for regex or keyword search across a whole document.

### items.jsonl
Structured document items. Each line is a JSON object with:
- `position`: sequential position in the document
- `self_ref`: item reference (e.g. "#/texts/5", "#/tables/0")
- `label`: item type — "section_header", "text", "table", "list_item", "caption", "formula", "picture", "code", "footnote"
- `text`: rendered content (tables are markdown with `|` columns)
- `page_numbers`: list of page numbers where the item appears

### Cross-referencing search results with items
Search results include `doc_item_refs` (e.g. `["#/texts/48", "#/tables/0"]`) that correspond to `self_ref` values in items.jsonl.

## Strategy

1. Use `search` tool first to understand what's in the knowledge base
2. Use `execute_code` to write analysis code
3. Iterate: run code, examine output, refine approach
4. Call `cite` with chunk IDs from search results you referenced

## Important

- Each `execute_code` call runs in a fresh interpreter (no persistent variables between calls)
- Use `print()` to output results — the output is your only feedback
- Always execute code to answer questions — don't just describe what code would do
- Use `await` for all async functions inside execute_code (search, list_documents, llm)
- Do NOT include chunk IDs or UUIDs in your answer text — your answer should read naturally. Use the `cite` tool separately to register citations.
