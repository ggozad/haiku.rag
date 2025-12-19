import asyncio
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from evaluations.benchmark import evaluate_dataset
from evaluations.datasets import DATASETS
from haiku.rag.config import AppConfig, find_config_file, load_yaml_config
from haiku.rag.logging import configure_cli_logging

load_dotenv()
configure_cli_logging()
console = Console()

app = typer.Typer(help="Run retrieval and QA benchmarks for configured datasets.")


def load_config(config_path: Path | None) -> AppConfig:
    """Load config from file or use defaults."""
    if config_path:
        if not config_path.exists():
            raise typer.BadParameter(f"Config file not found: {config_path}")
        console.print(f"Loading config from: {config_path}", style="dim")
        yaml_data = load_yaml_config(config_path)
        return AppConfig.model_validate(yaml_data)

    found_path = find_config_file(None)
    if found_path:
        console.print(f"Loading config from: {found_path}", style="dim")
        yaml_data = load_yaml_config(found_path)
        return AppConfig.model_validate(yaml_data)

    console.print("No config file found, using defaults", style="dim")
    return AppConfig()


def get_dataset_spec(dataset: str):
    """Get dataset spec or raise error."""
    spec = DATASETS.get(dataset.lower())
    if spec is None:
        valid_datasets = ", ".join(sorted(DATASETS))
        raise typer.BadParameter(
            f"Unknown dataset '{dataset}'. Choose from: {valid_datasets}"
        )
    return spec


@app.command()
def run(
    dataset: str = typer.Argument(..., help="Dataset key to evaluate."),
    config: Path | None = typer.Option(
        None, "--config", help="Path to haiku.rag YAML config file."
    ),
    db: Path | None = typer.Option(None, "--db", help="Override the database path."),
    skip_db: bool = typer.Option(
        False, "--skip-db", help="Skip updating the evaluation db."
    ),
    skip_retrieval: bool = typer.Option(
        False, "--skip-retrieval", help="Skip retrieval benchmark."
    ),
    skip_qa: bool = typer.Option(False, "--skip-qa", help="Skip QA benchmark."),
    limit: int | None = typer.Option(
        None, "--limit", help="Limit number of test cases for both retrieval and QA."
    ),
    name: str | None = typer.Option(None, "--name", help="Override evaluation name."),
    vacuum_interval: int = typer.Option(
        100, "--vacuum-interval", help="Vacuum every N documents during DB population."
    ),
) -> None:
    """Run retrieval and QA benchmarks on a dataset."""
    spec = get_dataset_spec(dataset)
    app_config = load_config(config)

    asyncio.run(
        evaluate_dataset(
            spec=spec,
            config=app_config,
            skip_db=skip_db,
            skip_retrieval=skip_retrieval,
            skip_qa=skip_qa,
            limit=limit,
            name=name,
            db_path=db,
            vacuum_interval=vacuum_interval,
        )
    )


@app.command()
def optimize(
    dataset: str = typer.Argument(..., help="Dataset key to use for optimization."),
    output: Path = typer.Option(
        Path("optimized_prompt.yaml"),
        "--output",
        "-o",
        help="Output YAML file for the optimized prompt.",
    ),
    config: Path | None = typer.Option(
        None, "--config", help="Path to haiku.rag YAML config file."
    ),
    db: Path | None = typer.Option(None, "--db", help="Override the database path."),
    trials: int = typer.Option(
        20, "--trials", "-t", help="Number of optimization trials."
    ),
    train_limit: int | None = typer.Option(
        None, "--train-limit", help="Limit training examples."
    ),
    auto_mode: str = typer.Option(
        "light",
        "--auto",
        help="MIPROv2 auto mode: light, medium, or heavy.",
    ),
) -> None:
    """Optimize the QA system prompt using DSPy MIPROv2.

    Uses the specified dataset to train/validate prompt optimization.
    Outputs an optimized prompt to a YAML file compatible with haiku.rag config.

    Example:
        evaluations optimize repliqa --output optimized_prompt.yaml --trials 20
    """
    from evaluations.optimizer import run_optimization, save_optimized_prompt

    spec = get_dataset_spec(dataset)

    if auto_mode not in ("light", "medium", "heavy"):
        raise typer.BadParameter(
            f"Invalid auto mode '{auto_mode}'. Choose from: light, medium, heavy"
        )

    app_config = load_config(config)

    console.print(f"Optimizing prompt for dataset: {dataset}", style="bold magenta")
    console.print(
        f"Model: {app_config.qa.model.provider}/{app_config.qa.model.name}", style="dim"
    )
    console.print(f"Trials: {trials}, Auto mode: {auto_mode}", style="dim")

    optimized_prompt = run_optimization(
        spec=spec,
        config=app_config,
        db_path=db,
        train_limit=train_limit,
        num_trials=trials,
        auto_mode=auto_mode,  # type: ignore[arg-type]
    )

    save_optimized_prompt(optimized_prompt, output)

    console.print("\n=== Optimization Complete ===", style="bold green")
    console.print(f"Optimized prompt saved to: {output}")
    console.print("\nTo use this prompt, merge into your haiku.rag.yaml or use:")
    console.print(f"  haiku-rag ask 'question' --config {output}")


if __name__ == "__main__":
    app()
