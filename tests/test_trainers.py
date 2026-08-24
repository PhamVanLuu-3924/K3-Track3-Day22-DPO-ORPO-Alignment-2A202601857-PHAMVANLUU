import json
from pathlib import Path

import pytest

from preference_lab.schemas import PreferenceExample
from preference_lab.trainers import MockPreferenceTrainer, TrainingConfig


def _examples() -> list[PreferenceExample]:
    return [
        PreferenceExample(
            prompt="Explain regularization.",
            chosen="Regularization helps reduce overfitting by penalizing model complexity.",
            rejected="Regularization makes every model train faster.",
        ),
        PreferenceExample(
            prompt="Explain backpropagation.",
            chosen="Backpropagation computes gradients which are used to update model weights.",
            rejected="Backpropagation randomly replaces the model weights.",
        ),
    ]


def test_mock_trainer_reduces_dpo_loss_and_writes_metrics(tmp_path: Path) -> None:
    trainer = MockPreferenceTrainer(
        TrainingConfig(method="dpo", steps=20, learning_rate=1.0, output_dir=tmp_path)
    )

    output = trainer.train(_examples())
    metrics = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / "training_metrics.json"
    assert metrics["training_mode"] == "mock_cpu"
    assert metrics["method"] == "dpo"
    assert metrics["num_train_examples"] == 2
    assert metrics["final_loss"] < metrics["initial_loss"]
    assert metrics["final_preference_margin"] > 0.0


def test_mock_trainer_is_deterministic(tmp_path: Path) -> None:
    config = TrainingConfig(method="dpo", output_dir=tmp_path)
    trainer = MockPreferenceTrainer(config)

    first = json.loads(trainer.train(_examples()).read_text(encoding="utf-8"))
    second = json.loads(trainer.train(_examples()).read_text(encoding="utf-8"))

    assert first == second


def test_mock_trainer_rejects_unsupported_method(tmp_path: Path) -> None:
    trainer = MockPreferenceTrainer(TrainingConfig(method="orpo", output_dir=tmp_path))

    with pytest.raises(ValueError, match="supports only method='dpo'"):
        trainer.train(_examples())


def test_mock_trainer_rejects_unsupported_backend(tmp_path: Path) -> None:
    trainer = MockPreferenceTrainer(
        TrainingConfig(method="dpo", backend="trl", output_dir=tmp_path)
    )

    with pytest.raises(ValueError, match="requires backend='mock_cpu'"):
        trainer.train(_examples())


def test_mock_trainer_rejects_empty_examples(tmp_path: Path) -> None:
    trainer = MockPreferenceTrainer(TrainingConfig(method="dpo", output_dir=tmp_path))

    with pytest.raises(ValueError, match="must not be empty"):
        trainer.train([])


@pytest.mark.parametrize(
    ("steps", "learning_rate", "message"),
    [
        (0, 1.0, "steps must be positive"),
        (1, 0.0, "learning_rate must be a positive finite number"),
        (1, float("inf"), "learning_rate must be a positive finite number"),
    ],
)
def test_mock_trainer_rejects_invalid_optimization_config(
    tmp_path: Path, steps: int, learning_rate: float, message: str
) -> None:
    trainer = MockPreferenceTrainer(
        TrainingConfig(
            method="dpo",
            steps=steps,
            learning_rate=learning_rate,
            output_dir=tmp_path,
        )
    )

    with pytest.raises(ValueError, match=message):
        trainer.train(_examples())
