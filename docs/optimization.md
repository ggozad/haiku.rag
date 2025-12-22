# Prompt Optimization

The `evaluations optimize` command uses [DSPy's MIPROv2](https://dspy.ai/) to automatically tune the QA system prompt for your domain and model combination.

## Usage

```bash
evaluations optimize <dataset> --output optimized_prompt.yaml
```

**Options:**

| Option | Description |
|--------|-------------|
| `--output` / `-o` | Output YAML file (default: `optimized_prompt.yaml`) |
| `--auto` | MIPROv2 mode: `light` (default), `medium`, `heavy`, or `none` |
| `--trials` / `-t` | Number of trials when `--auto none` |
| `--train-limit` | Limit training examples |
| `--config` | haiku.rag config file |
| `--db` | Override database path |

The optimizer outputs a YAML file you can merge into your config:

```yaml
prompts:
  qa: "Your optimized prompt here..."
```

Use it directly or merge into your main config under `prompts.qa`. See [Prompts configuration](configuration/prompts.md) for details.

## Configuration

Configure MIPROv2 behavior in your `haiku.rag.yaml`:

```yaml
optimization:
  teacher_model:
    provider: anthropic
    name: claude-sonnet-4-20250514
    temperature: 1.0
  num_candidates: 10
  seed: 42
```

| Option | Description | Default |
|--------|-------------|---------|
| `teacher_model` | Model used to generate instruction candidates. If not set, uses `qa.model` | `null` |
| `teacher_model.temperature` | Sampling temperature for instruction generation. Higher = more diverse proposals | `1.0` |
| `num_candidates` | Number of candidate instructions to generate per predictor | Auto (based on `--auto` mode) |
| `seed` | Random seed for reproducibility | `42` |

Using a stronger `teacher_model` (e.g., Claude Opus or GPT-4) can produce higher-quality instruction candidates, while the evaluation still uses your target `qa.model`. This is a one-time optimization cost.

## Bring Your Own Dataset

The optimizer learns from whatever dataset you provide. **For production use, you need your own Q&A pairs that reflect your actual domain and desired answer style.**

The built-in datasets are primarily for benchmarking:

| Dataset | Type | Notes |
|---------|------|-------|
| **wix** | Support/how-to | Useful if building a support chatbot |
| **repliqa** | Short factoid | Academic benchmark, not production-ready |
| **hotpotqa** | Short factoid | Academic benchmark, not production-ready |

Academic benchmarks like repliqa and hotpotqa have short, factoid-style answers. Optimizing on them produces prompts that demand terse responses—likely not what you want in production.

To create a custom dataset adapter, see `evaluations/evaluations/datasets/` for examples. You'll need:
- A corpus of documents
- Q&A pairs with questions and expected answers that demonstrate your desired response style

### Example: Wix-Optimized Prompt

The wix dataset contains support-style Q&A, making it useful for documentation/support use cases. Running with defaults:

```bash
evaluations optimize wix -o wix_prompt.yaml
```

Produces a prompt like:

```
You are a helpful Wix support assistant.
When answering a user question, follow the Wix support response style:

1. **Goal statement** – start with a concise sentence that states the aim of the
   solution, in second‑person ("Your goal is…").
2. **Numbered steps** – list clear, actionable steps, numbered 1‑3 or more.
   - Use the navigation phrase "Go to" followed by a placeholder URL in parentheses:
     `Go to <section> (https://support.wix.com/…)`.
   - Keep language simple, avoid jargon, and use second‑person ("you", "your").
3. **Optional notes** – at the end of the steps, add a short "Note" paragraph if needed.
4. **Markdown formatting** – use markdown headings (`#`, `##`), numbered lists, and links.
5. **Citations** – after the answer, list the chunk IDs used, prefixed by `Cited chunks:`.
   Example: `Cited chunks: [chunk_abc123], [chunk_def456]`.

**Workflow**
1. Call `search_documents` with relevant keywords from the question.
2. Examine the returned chunks, paying attention to their relevance scores,
   source titles, and content types.
3. If needed, perform up to two additional searches (max 3 total).
4. Construct the answer strictly from the retrieved content, following the
   style guidelines above.
5. If no sufficient information is found, reply:
   `"I cannot find enough information in the knowledge base to answer this question."`

**General rules**
- Base your answer solely on the retrieved chunks; do not inject external knowledge.
- Use the source and type metadata to interpret the context correctly.
- Keep the answer concise and focused on the user's issue.
- Cite the chunk IDs exactly as shown in the search results.
- Avoid extra commentary or analysis unless explicitly requested.
```

The optimizer learned the Wix support style from the dataset: goal statements, numbered steps, navigation phrases, and markdown formatting.

## Validation

**Validate on your own representative queries, not just benchmark scores.**

The optimizer improves scores on the training dataset, but that doesn't guarantee better production performance. A prompt optimized on factoid Q&A will score well on factoid benchmarks while producing unhelpful responses for real users.

Before adopting an optimized prompt:

1. **Review the prompt manually** — Does it make sense for your use case?
2. **Test on real queries** — Try it with questions your users actually ask
3. **Compare qualitatively** — Are answers more helpful, or just different?

## When Optimization Helps

- You have a dataset that represents your production use case
- Your expected answers demonstrate the style you want
- You're willing to iterate and manually refine results

## When Optimization Won't Help

- You only have academic benchmark datasets
- Your use case requires conversational, explanatory answers but your dataset has factoid answers
- You need the prompt to handle diverse query types not represented in training

In these cases, manual prompt engineering with qualitative testing is more effective.
