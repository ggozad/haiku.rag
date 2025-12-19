# Prompt Optimization

The `evaluations optimize` command uses [DSPy's MIPROv2](https://dspy.ai/) optimizer to automatically tune the QA system prompt for your specific domain and model combination.

## How It Works

```
MIPROv2 generates candidate prompts
    ↓
Metric function runs real QuestionAnswerAgent with candidate prompt
    ↓
Agent performs tool calling (search_documents), generates answer
    ↓
LLM judge scores answer against expected output
    ↓
Score returned to MIPROv2 for next iteration
```

The optimizer doesn't simulate anything—it runs the actual `QuestionAnswerAgent` with full tool calling against the real database, then uses the existing LLM judge to score answers.

## Usage

```bash
# Basic optimization (light mode)
evaluations optimize repliqa --output optimized_prompt.yaml

# Different auto modes (light/medium/heavy control effort vs quality)
evaluations optimize wix --auto medium -o wix_prompt.yaml

# Manual control over trials (use --auto none)
evaluations optimize repliqa --auto none --trials 50

# Limit training examples for faster iteration
evaluations optimize hotpotqa --train-limit 100
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

## Using Optimized Prompts

The optimizer outputs a YAML file you can merge into your config:

```yaml
# optimized_prompt.yaml
prompts:
  qa: "Your optimized prompt here..."
```

Use it directly:

```bash
haiku-rag ask "question" --config optimized_prompt.yaml
```

Or merge into your main config file under the `prompts.qa` key. See [Prompts configuration](configuration/prompts.md) for details.

## Dataset Selection

The dataset you optimize against significantly shapes the resulting prompt. Each dataset has characteristics that influence what the optimizer learns:

| Dataset | Answer Style | Best For |
|---------|-------------|----------|
| **repliqa** | Short factoid (entity, number, yes/no) | Extractive Q&A, fact lookup |
| **wix** | Support/how-to (explanatory) | Customer support, documentation |
| **hotpotqa** | Short factoid (multi-hop reasoning) | Complex queries requiring synthesis |

**Recommendation:** Choose a dataset that matches your production use case. If your users ask support-style questions, optimize on `wix`. If they ask factual lookups, use `repliqa` or `hotpotqa`.

## Important Considerations

### Benchmark vs Production Tradeoff

Academic benchmarks have "gold standard" answers that optimizers target. The LLM judge rewards responses matching those expected answers. This creates a fundamental tension:

- **Benchmark datasets** favor short, verifiable answers (easier to evaluate)
- **Production users** often want explanatory, helpful responses

An optimizer will learn to produce answers that score well on the benchmark—which may mean terse, factoid-style responses that feel unhelpful in practice.

**Example:** A prompt optimized on hotpotqa might learn rules like "provide single-word answers" because that's what scores highest. That same prompt would frustrate users asking "How do I configure authentication?"

### What to Expect

- Optimized prompts typically improve benchmark scores by 5-15%
- The improvement comes from learning dataset-specific patterns
- Prompts may become overly specific to the training dataset's answer format
- Multi-hop datasets (hotpotqa) tend to produce more rigid, factoid-focused prompts

### Recommended Workflow

1. **Start with wix** if your use case involves support/documentation queries
2. **Run optimization** with `--auto light` for initial results
3. **Manually review** the optimized prompt before adopting it
4. **Test on real queries** outside the benchmark to verify helpfulness
5. **Iterate** — you may need to manually edit the optimized prompt to restore flexibility

### When Optimization May Not Help

- Your use case requires conversational, explanatory answers
- You need the prompt to handle diverse query types
- The available datasets don't match your domain

In these cases, manual prompt engineering with qualitative testing may be more effective than automated optimization.

## Validating Results

After optimization, compare benchmark scores before and after:

```bash
# Baseline
evaluations run repliqa --limit 200

# With optimized prompt
evaluations run repliqa --limit 200 --config optimized_prompt.yaml
```

A score improvement on the benchmark doesn't guarantee production improvement. Always validate with representative real-world queries before adopting an optimized prompt.
