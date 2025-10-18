# Tutorial

This tutorial quickstart instructions for getting familiar with `haiku.rag`. This tutorial is indented for people who are familiar with command line and Python, but not different AI ecosystem tools.

The tutorial covers:

- RAG and embeddings basics
- Install `haiku.rag` Python package
- Set up environment variables for running `haiku.rag`
- Adding and retrieving items
- Inspecting the database

The tutorial uses OpenAI API service - no local installation needed and will work on computers with any amount of RAM and GPU. The OpenAI API is pay-as-you-go, so you need to top it up at least up top ~$5 when creating the API key.

## Introduction

Embeddings serve as the foundational bridge between unstructured text data and computational efficiency in AI systems, particularly within Retrieval-Augmented Generation (RAG) frameworks that enhance Large Language Models (LLMs). At their core, embeddings are dense vector representations of words, sentences, or documents, created by models like those from OpenAI, which encode semantic and contextual meaning into numerical forms. In a RAG pipeline, these embeddings enable the system to index and store vast amounts of information in a vector database, allowing for rapid similarity searches. When a user query is posed, its embedding is generated and compared against the stored vectors using metrics like cosine similarity to retrieve the most relevant documents or chunks of text. This retrieval step addresses a key limitation of standalone LLMs, which rely on pre-trained knowledge that can be outdated, incomplete, or prone to hallucinations, by providing external, grounded context to inform the model's output.

You technically can create vector embeddings yourself without relying on pre-built models like OpenAI's, but it's impractical for most users due to the complexity and resources involved in training or implementing from scratch. Generating embeddings requires a neural network architecture (e.g., transformer-based) trained on massive datasets to learn semantic relationships, which demands significant computational power (GPUs/TPUs), expertise in machine learning, and access to billions of text examples for fine-tuning.

`haiku.rag` is a Python library allowing you to set up your own embeddings database, pipeline to feed into it and query it with different LLM providers and related services.

## Setup

First, [get an OpenAI API key](https://platform.openai.com/api-keys).

Install `haiku.rag` Python package using [uv](https://docs.astral.sh/uv/getting-started/installation/) or your favourite Python package manager:

```shell
# Python 3.12+ needed
uv install haiku.rag
```

Configure your OpenAI API key and embeddings model.

- Haiku RAG supports [dotenv](https://pypi.org/project/python-dotenv/) environment files and environment varibles for configuration
- [See OpenAPI vector embeddings documentation](https://platform.openai.com/docs/guides/embeddings/embedding-models)

Create a file called `.env` and add:

```shell
#
# These settings are relevant for converting documents to embeddings
#

EMBEDDINGS_PROVIDER="openai"
# or text-embedding-3-large
EMBEDDINGS_MODEL="text-embedding-3-small"
EMBEDDINGS_VECTOR_DIM=1536
OPENAI_API_KEY="<your OpenAPI API key goes here>"

#
# These settings are relevant for question answering chats
#

# We tell Haiku.rag to use OpenAI remote AI for chats, instead of local ollama.
QA_PROVIDER="openai"
QA_MODEL="gpt-4o-mini"  # or gpt-4, gpt-3.5-turbo, etc.
```

For the list of available OpenAI embedding models and `EMBEDDINGS_VECTOR_DIM` options, ask ChatGPT for instructions.

## Adding the first documents

Now you can add some pieces of text in the database:

```shell
haiku-rag add "Python is the best programming language in the world, because it is flexible, with robust ecosystem, open source licensing and thousands of contributors"
haiku-rag add "JavaScript is a popular programming language, but has a lot of warts"
haiku-rag add "PHP is a bad programming language, because of spotted security history, horrible syntax and declining popularity"
```

What will happen

- The piece of text is send to OpenAI `/embeddings` API service
- OpenAI translates the free form text to RAG embedding vectors needed for the retrieval
- The vector values will be stored in a local database

Now you can view your [LanceDB](<(https://lancedb.com/) database, and the embeddings it is configured for:

```shell
haiku-rag info
```

You should get the back the information:

```
haiku.rag database info
  path: /Users/moo/Library/Application Support/haiku.rag/haiku.rag.lancedb
  haiku.rag version (db): 0.12.1
  embeddings: openai/text-embedding-3-small (dim: 1536)
  documents: 4
  versions (documents): 9
  versions (chunks): 10
──────────────────────────────────────────────────────────────────────────────────
Versions
  haiku.rag: 0.12.1
  lancedb: 0.25.2
  docling: 2.57.0
```

## Asking questions and retrieving information

Now we can use OpenAI LLMs to retrieve information from our embeddings database.

In this example, we connect to a remote OpenAI API.
Mak

Behind the scenes [pydantic-ai](https://ai.pydantic.dev/) query is created
using `OpenAIChatModel.request()`.

The easiest way to do this is `ask` CLI command:

```shell
haiku-rag ask "What is the best programming language in the world"
```

```
Question: What is the best programming language in the world

Answer:
According to the document, Python is considered the best programming language in the world due to its flexibility, robust ecosystem, open-source licensing, and thousands of contributors.
```

## Programmatic interaction in Python

You can interact with Haiku RAG from Python in a similar manner as you can from the command line. Here we use Haiku RAG with the interactive Python command prompt (REPL).

First we need to install `ipython`, as built-in Python REPL does not support async blocks.

```shell
uv pip install ipython
```

Run IPython:

```shell
ipython
```

Then copy paste in the snippet (you can use [%cpaste](https://ipythonbook.com/magic/cpaste.html) command):

```python
import sys
import logging
from haiku.rag.client import HaikuRAG

# Increase logging verbosity so we see what happens behind the scenes,
# and check that the logger works
logging.basicConfig(
  stream=sys.stdout,
  level=logging.DEBUG,
  format="%(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.debug("AGI here we come")

# Uses LanceDB database from Config.DEFAULT_DATA_DIR
async with HaikuRAG() as client:
    answer = await client.ask("What is the best programming language in the world?")
    print(answer)

```

You should see:

```
2025-10-18 17:05:49,611 - DEBUG - HTTP Response: POST https://api.openai.com/v1/chat/completions "200 OK" Headers({'date': 'Sat, 18 Oct 2025 14:05:49 GMT', 'content-type': 'application/json', 'transfer-encoding': 'chunked', 'connection': 'keep-alive', 'access-control-expose-headers': 'X-Request-ID', 'openai-organization': 'xxx', 'openai-processing-ms': '788', 'openai-project': 'xxx', 'openai-version': '2020-10-01', 'x-envoy-upstream-service-time': '1050', 'x-ratelimit-limit-requests': '10000', 'x-ratelimit-limit-tokens': '200000', 'x-ratelimit-remaining-requests': '9998', 'x-ratelimit-remaining-tokens': '199603', 'x-ratelimit-reset-requests': '14.981s', 'x-ratelimit-reset-tokens': '119ms', 'x-request-id': 'req_9651a3691a144dd388e97066ad67a49c', 'x-openai-proxy-wasm': 'v0.1', 'cf-cache-status': 'DYNAMIC', 'strict-transport-security': 'max-age=31536000; includeSubDomains; preload', 'x-content-type-options': 'nosniff', 'server': 'cloudflare', 'cf-ray': '990897b6f8d270d7-ARN', 'content-encoding': 'gzip', 'alt-svc': 'h3=":443"; ma=86400'})
2025-10-18 17:05:49,611 - DEBUG - request_id: req_9651a3691a144dd388e97066ad67a49c

According to the document, Python is considered the best programming language in the world due to its flexibility, robust ecosystem, open-source licensing, and support from thousands of contributors.
```

## Complex documents

Haiku RAG can also handle types beyond plain text..

Here we add research papers about Python from [arxiv](https://arxiv.org/search/?query=python&searchtype=all&source=header) using URL retriever.

```shell
# Better Python Programming for all: With the focus on Maintainability
haiku-rag add-src --meta collection="Interesting Python papers" "https://arxiv.org/pdf/2408.09134"

# Interoperability From OpenTelemetry to Kieker: Demonstrated as Export from the Astronomy Shop
haiku-rag add-src --meta collection="Interesting Python papers" "https://arxiv.org/pdf/2510.11179"
```

Then we can query this:

```shell
haiku-rag ask "Who wrote a paper about OpenTelemetry interoperability, and what was his take"
```

We should get something along the lines:

```
Answer:
David Georg Reichelt from Lancaster University wrote a paper titled "Interoperability From OpenTelemetry to Kieker: Demonstrated as Export from the Astronomy Shop." In his work, he indicates that there is a structural difference between Kieker’s synchronous traces and OpenTelemetry’s asynchronous traces, leading to  limited compatibility between the two systems. This highlights the challenges of interoperability in observability frameworks.
```

We can also add offline files, like PDFs. Here we add a local file to ensure OpenAI does not cheat - a file we know that should not very well known in Internet:

```shell
# This static file is supplied in haiku.rag repo
haiku-rag add-src "examples/samples/PyCon Finland 2025 Schedule.html"
```

And then:

```shell
haiku-rag ask "Who were presenting talks in Pycon Finland 2025? Can you give at least five different people."
```

```
The following people are presenting talks at PyCon Finland 2025:

 1 Jeremy Mayeres - Talk: The Limits of Imagination: An Open Source Journey
 2 Aroma Rodrigues - Talk: Python and Rust, a Perfect Pairing
 3 Andreas Jung - Talk: Guillotina Volto: A New Backend for Volto
 4 Daniel Vahla - Talk: Experiences with AI in Software Projects
 5 Andreas Jung (also presenting another talk) - Talk: Debugging Python
```

## Reseting the embeddings database

If you change your embeddings provider (OpenAI -> ollama) or its parameters, you need to delete the LanceDB database and add the documents again:

```shell
rm -rf "/Users/moo/Library/Application Support/haiku.rag/haiku.rag.lancedb"
```

## Configuration

See [Configuration page](./configuration.md) for more information about configurait

For the available environment variable config options see [config.py](https://github.com/ggozad/haiku.rag/blob/main/src/haiku/rag/config.py).
