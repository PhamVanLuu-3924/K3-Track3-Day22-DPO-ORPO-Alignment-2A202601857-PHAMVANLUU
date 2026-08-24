from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import print

from .config import load_config
from .data import load_jsonl, split_by_prompt
from .evaluate import MetricValue, deterministic_quality_score, pairwise_accuracy, write_metrics
from .trainers import MockPreferenceTrainer, TrainingConfig

app = typer.Typer(help="Preference alignment lab CLI")


@app.command()
def validate(data: Path) -> None:
    examples = load_jsonl(data)
    print(f"[green]Loaded {len(examples)} preference examples[/green]")


@app.command()
def train(
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ],
) -> None:
    cfg = load_config(config)
    training_cfg = cfg["training"]
    evaluation_cfg = cfg.get("evaluation", {})
    examples = load_jsonl(cfg["paths"]["train_data"])
    train_examples, _ = split_by_prompt(
        examples,
        validation_ratio=float(evaluation_cfg.get("validation_ratio", 0.2)),
        seed=int(cfg.get("seed", 42)),
    )
    trainer = MockPreferenceTrainer(
        TrainingConfig(
            method=str(training_cfg["method"]),
            backend=str(training_cfg.get("backend", "mock_cpu")),
            beta=float(training_cfg.get("beta", 0.1)),
            lambda_orpo=float(training_cfg.get("lambda_orpo", 0.1)),
            max_length=int(training_cfg.get("max_length", 512)),
            batch_size=int(training_cfg.get("batch_size", 2)),
            steps=int(training_cfg.get("steps", 20)),
            learning_rate=float(training_cfg.get("learning_rate", 1.0)),
            output_dir=Path(cfg["paths"]["output_dir"]),
        )
    )
    out = trainer.train(train_examples)
    print(f"[green]Wrote mock training metrics to {out}[/green]")


@app.command()
def evaluate(
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ],
) -> None:
    cfg = load_config(config)
    examples = load_jsonl(cfg["paths"]["train_data"])
    evaluation_cfg = cfg.get("evaluation", {})
    scorer = str(evaluation_cfg.get("scorer", "deterministic_cpu"))
    if scorer != "deterministic_cpu":
        raise ValueError(f"unsupported evaluation scorer: {scorer}")
    validation_ratio = float(evaluation_cfg.get("validation_ratio", 0.2))
    seed = int(cfg.get("seed", 42))
    _, evaluation_examples = split_by_prompt(examples, validation_ratio=validation_ratio, seed=seed)

    chosen_scores = [deterministic_quality_score(example.chosen) for example in evaluation_examples]
    rejected_scores = [
        deterministic_quality_score(example.rejected) for example in evaluation_examples
    ]
    ties = sum(chosen == rejected for chosen, rejected in zip(chosen_scores, rejected_scores))
    metrics: dict[str, MetricValue] = {
        "evaluation_mode": scorer,
        "num_examples": len(evaluation_examples),
        "pairwise_accuracy": pairwise_accuracy(evaluation_examples, chosen_scores, rejected_scores),
        "ties": ties,
        "mean_chosen_score": sum(chosen_scores) / len(chosen_scores),
        "mean_rejected_score": sum(rejected_scores) / len(rejected_scores),
    }
    out = write_metrics(metrics, cfg["paths"]["output_dir"])
    print(f"[green]Wrote metrics to {out}[/green]")


if __name__ == "__main__":
    app()
