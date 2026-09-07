---
name: haiku-rag
description: Search, read and compute over the user's haiku.rag knowledge base
  through the haiku-rag MCP tools. Use whenever a request could be answered
  from the user's ingested documents, when asked to find, look up, check or
  cite something in their documents or knowledge base, or when the question is
  about the user's own material rather than general knowledge.
compatibility: Requires the haiku-rag MCP server to be registered in the client.
allowed-tools:
  - mcp__plugin_haiku-rag_haiku-rag__search_documents
  - mcp__plugin_haiku-rag_haiku-rag__search_documents_by_image
  - mcp__plugin_haiku-rag_haiku-rag__get_document
  - mcp__plugin_haiku-rag_haiku-rag__get_document_outline
  - mcp__plugin_haiku-rag_haiku-rag__get_document_section
  - mcp__plugin_haiku-rag_haiku-rag__list_documents
  - mcp__plugin_haiku-rag_haiku-rag__execute_code
---

# Working with the knowledge base

Check the knowledge base before answering from memory whenever the question
could be about the user's documents. Say so when it has nothing relevant.

## Find

`search_documents` is the first call. Results come best first with the document
title, section headings, the matched chunk's metadata when it has any, and the
passage in its section. Pictures in the results arrive as images: answer
figure questions from them. `filter` restricts which documents are searched,
`limit` how many results come back. If it misses, rephrase once or narrow with
a filter before concluding the material is not there. When the question is
about an image rather than words and the server offers
`search_documents_by_image`, it takes the image as the query.

## Read

Every search result shows its `Document ID` (and `Collection` when there are
several); pass them to the read tools. `get_document` returns a document's
whole text in reading order. For a long one, `get_document_outline` gives the
heading tree with page numbers and `get_document_section` the text of one
section, subsections included.

## Compute

`execute_code` runs a Python program on the server over the same documents.
Under `/documents/{id}/` each has `metadata.json`, `content.txt`, `items.jsonl`,
`chunks.jsonl` and `toc.json`, and the program can `await search(query)` and
`await list_documents()`. Write code when the answer is a count, an aggregate, a
comparison across many documents, a lookup by document or chunk metadata, or a
pattern over whole documents: whatever search cannot rank. Each call is one
program and variables do not carry over, so gather, compute and `print` a
compact result in the same program. `filter` and `sources` select the documents
it sees. For a known document's structure read its `toc.json` first; `search()`
ranks across every document. Map a title or URI to an id with one
`list_documents()` call rather than reading every `metadata.json`; the files
carry no `source`, so over several collections group by its rows. Answer and
cite from what it printed.

## Explore

`list_documents` shows what is stored: titles, URIs and metadata. It is how you
learn what a filter can match.

## Filters

A SQL WHERE clause over the document columns `id`, `uri`, `title`,
`created_at`, `updated_at`, `metadata`. `metadata` is a JSON string, so match
it with LIKE: `metadata LIKE '%"author": "Smith"%'`. Also `uri LIKE '%.pdf'`,
`title = 'Q3 report'`.

## Results and citations

Rank is the signal; scores are not comparable across queries and are never
confidence. Cite the document title or URI, the section heading and page
numbers when present, and the matched chunk's metadata when it carries locators
such as paragraph or footnote numbers. When results carry `source`, the server
covers several collections: name it, and pass `sources` to search a subset.
