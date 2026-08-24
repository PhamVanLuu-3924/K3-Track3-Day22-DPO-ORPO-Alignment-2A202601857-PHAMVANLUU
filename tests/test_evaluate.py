import json
from pathlib import Path

import pytest

from preference_lab.evaluate import (
    deterministic_quality_score,
    pairwise_accuracy,
    write_metrics,
)
from preference_lab.schemas import PreferenceExample


def _examples(count: int) -> list[PreferenceExample]:
    return [
        PreferenceExample(prompt=f"p{index}", chosen="good", rejected="bad")
        for index in range(count)
    ]


def test_pairwise_accuracy() -> None:
    examples = _examples(3)
    assert pairwise_accuracy(examples, [2.0, 1.0, 1.0], [1.0, 2.0, 1.0]) == 0.5


def test_pairwise_accuracy_handles_empty_input() -> None:
    assert pairwise_accuracy([], [], []) == 0.0


def test_pairwise_accuracy_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="must have equal lengths"):
        pairwise_accuracy(_examples(1), [1.0], [])


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_pairwise_accuracy_rejects_non_finite_scores(score: float) -> None:
    with pytest.raises(ValueError, match="only finite values"):
        pairwise_accuracy(_examples(1), [score], [0.0])


def test_deterministic_quality_score_is_bounded_and_label_independent() -> None:
    short_score = deterministic_quality_score("Incorrect.")
    detailed_score = deterministic_quality_score(
        "Regularization helps reduce overfitting because it penalizes excessive "
        "model complexity while preserving useful patterns in the training data."
    )

    assert 0.0 <= short_score <= 1.0
    assert 0.0 <= detailed_score <= 1.0
    assert detailed_score > short_score
    assert deterministic_quality_score("   ") == 0.0


def test_write_metrics_supports_metadata(tmp_path: Path) -> None:
    metrics = {
        "evaluation_mode": "deterministic_cpu",
        "num_examples": 5,
        "pairwise_accuracy": 0.8,
    }

    output = write_metrics(metrics, tmp_path / "outputs")

    assert json.loads(output.read_text(encoding="utf-8")) == metrics
