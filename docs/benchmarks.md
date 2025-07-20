# Benchmarks

We use the [repliqa](https://huggingface.co/datasets/ServiceNow/repliqa) dataset for the evaluation of `haiku.rag`.

You can perform your own evaluations using as example the script found at
`tests/generate_benchmark_db.py`.

## Recall

In order to calculate recall, we load the `News Stories` from `repliqa_3` which is 1035 documents and index them in a sqlite db. Subsequently, we run a search over the `question` field for each row of the dataset and check whether we match the document that answers the question.


The recall obtained is ~0.73 for matching in the top result, raising to ~0.75 for the top 3 results.

| Embedding Model                       | Document in top 1 | Document in top 3 | Reranker               |
|---------------------------------------|-------------------|-------------------|------------------------|
| Ollama / `mxbai-embed-large`          | 0.77              | 0.89              | None                   |
| Ollama / `mxbai-embed-large`          | 0.81              | 0.91              | `mxbai-rerank-base-v2` |
| Ollama / `nomic-embed-text`           | 0.74              | 0.88              | None                   |
| OpenAI / `text-embeddings-3-small`    | 0.75              | 0.88              | None                   |
| OpenAI / `text-embeddings-3-small`    | 0.75              | 0.88              | None                   |
| OpenAI / `text-embeddings-3-small`    | 0.83              | 0.90              | Cohere / `rerank-v3.5` |

## Question/Answer evaluation

Again using the same dataset, we use a QA agent to answer the question. In addition we use an LLM judge (using the Ollama `qwen3`) to evaluate whether the answer is correct or not. The obtained accuracy is as follows:

| Embedding Model                    | QA Model                          | Accuracy  | Reranker               |
|------------------------------------|-----------------------------------|-----------|------------------------|
| Ollama / `mxbai-embed-large`       | Ollama / `qwen3`                  | 0.64      | None                   |
| Ollama / `mxbai-embed-large`       | Ollama / `qwen3`                  | 0.72      | `mxbai-rerank-base-v2` |
| Ollama / `mxbai-embed-large`       | Anthropic / `Claude Sonnet 3.7`   | 0.79      | None                   |
| OpenAI / `text-embeddings-3-small` | OpenAI / `gpt-4-turbo`            | 0.62      | None                   |
