import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import dspy
import yaml
from rich.console import Console

from evaluations.config import DatasetSpec
from evaluations.evaluators import LLMJudge
from evaluations.optimizer.dspy_module import QAModule
from evaluations.optimizer.metric import QAMetric
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig, ModelConfig
from haiku.rag.qa.prompts import QA_SYSTEM_PROMPT

console = Console()


def build_training_examples(
    spec: DatasetSpec,
    limit: int | None = None,
) -> list[dspy.Example]:
    """
    Build DSPy training examples from the evaluation dataset.

    Args:
        spec: Dataset specification
        limit: Maximum number of examples to load

    Returns:
        List of DSPy examples with question and expected_answer fields
    """
    corpus = spec.qa_loader()
    if limit is not None:
        corpus = corpus.select(range(min(limit, len(corpus))))

    examples = []
    for doc in corpus:
        doc_mapping = cast(Mapping[str, Any], doc)
        question = doc_mapping["question"]
        answer = doc_mapping["answer"]

        # Skip unanswerable questions
        if answer == "The answer is not found in the document.":
            continue

        example = dspy.Example(
            question=question,
            expected_answer=answer,
        ).with_inputs("question")

        examples.append(example)

    return examples


def create_dspy_lm(model_config: ModelConfig, app_config: AppConfig) -> dspy.LM:
    """Create a DSPy LM from haiku.rag model configuration."""
    provider = model_config.provider
    model = model_config.name

    if provider == "ollama":
        base_url = app_config.providers.ollama.base_url
        return dspy.LM(
            model=f"ollama_chat/{model}",
            api_base=base_url,
        )
    elif provider == "openai":
        return dspy.LM(model=f"openai/{model}")
    elif provider == "anthropic":
        return dspy.LM(model=f"anthropic/{model}")
    else:
        return dspy.LM(model=f"{provider}/{model}")


AutoMode = Literal["light", "medium", "heavy"]


async def optimize_prompt(
    spec: DatasetSpec,
    config: AppConfig,
    db_path: Path | None = None,
    train_limit: int | None = None,
    num_trials: int = 20,
    auto_mode: AutoMode | None = "light",
) -> str:
    """
    Run MIPROv2 optimization on the QA system prompt.

    Args:
        spec: Dataset specification
        config: Application configuration
        db_path: Override database path
        train_limit: Limit training examples
        num_trials: Number of optimization trials
        auto_mode: MIPROv2 auto mode (light/medium/heavy)

    Returns:
        Optimized system prompt string
    """
    db = spec.db_path(db_path)

    if not db.exists():
        raise ValueError(
            f"Database not found: {db}. "
            f"Run 'evaluations run {spec.key}' first to populate the database."
        )

    # Build training examples
    console.print("Building training examples...", style="blue")
    examples = build_training_examples(spec, limit=train_limit)
    console.print(f"Loaded {len(examples)} training examples", style="green")

    if len(examples) < 10:
        raise ValueError(
            f"Not enough examples ({len(examples)}). Need at least 10 for optimization."
        )

    # Split into train/val (80/20)
    split_idx = int(len(examples) * 0.8)
    trainset = examples[:split_idx]
    valset = examples[split_idx:]
    console.print(f"Train: {len(trainset)}, Val: {len(valset)}", style="dim")

    # Configure DSPy with the model for prompt generation
    console.print("Configuring DSPy...", style="blue")
    dspy_lm = create_dspy_lm(config.qa.model, config)
    dspy.configure(lm=dspy_lm)

    # Get initial prompt
    initial_prompt = config.prompts.qa or QA_SYSTEM_PROMPT
    console.print(f"Initial prompt length: {len(initial_prompt)} chars", style="dim")

    # Create DSPy module with initial prompt
    qa_module = QAModule(instructions=initial_prompt)

    async with HaikuRAG(db, config=config) as rag:
        # Create judge and metric
        judge = LLMJudge(config=config)
        metric = QAMetric(rag=rag, config=config, judge=judge)

        def evaluation_metric(
            example: dspy.Example,
            prediction: dspy.Prediction,
            trace: object = None,
        ) -> float:
            """Metric wrapper that updates the prompt before evaluation."""
            # Get current instructions from the module being evaluated
            current_prompt = qa_module.get_instructions()
            metric.set_prompt(current_prompt)
            return metric(example, prediction, trace)

        # Configure MIPROv2 for instruction-only optimization
        # When auto mode is set, MIPROv2 determines num_trials automatically
        # When auto is None, we use explicit num_trials
        if auto_mode is not None:
            optimizer = dspy.MIPROv2(
                metric=evaluation_metric,
                auto=auto_mode,
                verbose=False,
            )
        else:
            optimizer = dspy.MIPROv2(
                metric=evaluation_metric,
                verbose=False,
            )

        if auto_mode is not None:
            optimized_module = optimizer.compile(
                qa_module,
                trainset=trainset,
                valset=valset,
                max_bootstrapped_demos=0,  # instruction-only, no few-shot
                max_labeled_demos=0,
            )
        else:
            optimized_module = optimizer.compile(
                qa_module,
                trainset=trainset,
                valset=valset,
                num_trials=num_trials,
                max_bootstrapped_demos=0,
                max_labeled_demos=0,
            )

        # Extract optimized prompt
        optimized_prompt = optimized_module.get_instructions()

    return optimized_prompt


def save_optimized_prompt(prompt: str, output_path: Path) -> None:
    """
    Save optimized prompt to YAML file compatible with PromptsConfig.

    Args:
        prompt: The optimized prompt text
        output_path: Path to save the YAML file
    """
    config = {
        "prompts": {
            "qa": prompt,
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, width=1000)

    console.print(f"Saved optimized prompt to: {output_path}", style="green")


def run_optimization(
    spec: DatasetSpec,
    config: AppConfig,
    db_path: Path | None = None,
    train_limit: int | None = None,
    num_trials: int = 20,
    auto_mode: AutoMode | None = "light",
) -> str:
    """Sync wrapper for optimize_prompt."""
    return asyncio.run(
        optimize_prompt(
            spec=spec,
            config=config,
            db_path=db_path,
            train_limit=train_limit,
            num_trials=num_trials,
            auto_mode=auto_mode,
        )
    )
