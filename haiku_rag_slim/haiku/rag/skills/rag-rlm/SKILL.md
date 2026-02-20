---
name: rag-rlm
description: Analyze documents using code execution in a Docker sandbox.
---

# RLM (Reflexion Language Model) Analysis

You have access to a computational analysis tool that can write and execute Python code against the knowledge base.

## When to use `analyze`

Use the `analyze` tool for questions that require:

- **Computation** — counting, aggregation, averages, statistics
- **Data traversal** — iterating over documents, comparing tables, extracting structured data
- **Code execution** — any question best answered by writing and running Python code
- **Complex reasoning** — multi-step analysis that goes beyond simple search or Q&A

Examples:
- "How many pages are in this document?"
- "Compare the results in table 3 across all documents"
- "Calculate the average word count per document"
- "Write code to extract all email addresses"

## Requirements

The analyze tool requires Docker to be running, as code execution happens in an isolated Docker sandbox.

## Parameters

- `question` (required) — The analytical question to answer
- `document` — Optional document ID or title to pre-load for analysis
- `filter` — Optional SQL WHERE clause to filter documents
