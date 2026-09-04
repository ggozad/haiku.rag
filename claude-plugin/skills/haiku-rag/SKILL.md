---
name: haiku-rag
description: Search, read and question the user's haiku.rag knowledge base
  through the haiku-rag MCP tools. Use whenever a request could be answered
  from the user's ingested documents, when asked to find, look up, check or
  cite something in their documents or knowledge base, or when the question is
  about the user's own material rather than general knowledge.
allowed-tools:
  - mcp__plugin_haiku-rag_haiku-rag__search_documents
  - mcp__plugin_haiku-rag_haiku-rag__search_documents_by_image
  - mcp__plugin_haiku-rag_haiku-rag__get_document
  - mcp__plugin_haiku-rag_haiku-rag__get_document_outline
  - mcp__plugin_haiku-rag_haiku-rag__get_document_section
  - mcp__plugin_haiku-rag_haiku-rag__list_documents
  - mcp__plugin_haiku-rag_haiku-rag__ask_question
  - mcp__plugin_haiku-rag_haiku-rag__analyze
---

# Working with the knowledge base

Check the knowledge base before answering from memory whenever the question
could be about the user's documents. Say so when it has nothing relevant.

## Find

`search_documents` is the first call. Results come best first with the document
title, section headings, the matched chunk's metadata when it has any, and the
passage in its section. `filter` restricts which
documents are searched, `limit` how many results come back. If it misses,
rephrase once or narrow with a filter before concluding the material is not
there.

## Read

Every search result shows its `Document ID` (and `Collection` when there are
several); pass them to the read tools. `get_document` returns a document's
whole text in reading order. For a long one, `get_document_outline` gives the
heading tree with page numbers and `get_document_section` the text of one
section, subsections included.

## Answer or compute

`ask_question` runs the RAG agent on the server and returns an answer with
citations; use it when the user wants an answer rather than material.
`analyze` runs code in a sandbox over the documents; use it for counting,
aggregation, comparison across many documents or computation over tables. Both
cost a model call and are slower than a search.

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
numbers when present. When results carry `source`, the server covers several
collections: name it, and pass `sources` to search a subset.
