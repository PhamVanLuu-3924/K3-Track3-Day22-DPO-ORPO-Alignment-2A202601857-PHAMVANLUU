from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .evaluate import deterministic_quality_score
from .losses import dpo_loss
from .schemas import PreferenceExample


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    backend: str = "mock_cpu"
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2
    steps: int = 20
    learning_rate: float = 1.0
    output_dir: Path = Path("outputs")


class PreferenceTrainer(ABC):
    """Interface for DPO/ORPO training implementations."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    @abstractmethod
    def train(self, examples: list[PreferenceExample]) -> Path:
        """Run training and return the path to its metrics artifact."""


class MockPreferenceTrainer(PreferenceTrainer):
    """CPU-only demonstration of optimizing a scalar DPO preference margin.

    This trainer does not update a language model. It starts from deterministic
    response scores, applies a shared policy preference margin, and optimizes that
    scalar with the analytical DPO gradient. The resulting metrics demonstrate the
    objective's direction and numerical behavior without claiming a real checkpoint.
    """

    def train(self, examples: list[PreferenceExample]) -> Path:
        if self.config.backend != "mock_cpu":
            raise ValueError("MockPreferenceTrainer requires backend='mock_cpu'")
        if self.config.method != "dpo":
            raise ValueError("MockPreferenceTrainer supports only method='dpo'")
        if not examples:
            raise ValueError("training examples must not be empty")
        if self.config.steps <= 0:
            raise ValueError("steps must be positive")
        if not math.isfinite(self.config.learning_rate) or self.config.learning_rate <= 0.0:
            raise ValueError("learning_rate must be a positive finite number")

        reference_chosen = np.array(
            [deterministic_quality_score(example.chosen) - 1.0 for example in examples]
        )
        reference_rejected = np.array(
            [deterministic_quality_score(example.rejected) - 1.0 for example in examples]
        )

        preference_margin = 0.0
        initial_loss = self._loss(reference_chosen, reference_rejected, preference_margin)
        for _ in range(self.config.steps):
            scaled_margin = self.config.beta * preference_margin
            inverse_exponential = math.exp(-scaled_margin)
            gradient = -self.config.beta * inverse_exponential / (1.0 + inverse_exponential)
            preference_margin -= self.config.learning_rate * gradient

        final_loss = self._loss(reference_chosen, reference_rejected, preference_margin)
        metrics: dict[str, float | int | str] = {
            "training_mode": self.config.backend,
            "method": self.config.method,
            "num_train_examples": len(examples),
            "steps": self.config.steps,
            "beta": self.config.beta,
            "learning_rate": self.config.learning_rate,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "final_preference_margin": preference_margin,
        }

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        output = self.config.output_dir / "training_metrics.json"
        output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        return output

    def _loss(
        self,
        reference_chosen: np.ndarray,
        reference_rejected: np.ndarray,
        preference_margin: float,
    ) -> float:
        policy_chosen = reference_chosen + preference_margin / 2.0
        policy_rejected = reference_rejected - preference_margin / 2.0
        return dpo_loss(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            beta=self.config.beta,
        )
