from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .schemas import PreferenceExample

MetricValue = float | int | str

_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
_EXPLANATION_MARKERS = frozenset(
    {
        "because",
        "by",
        "during",
        "enabling",
        "helps",
        "means",
        "therefore",
        "while",
        "which",
    }
)


def deterministic_quality_score(text: str) -> float:
    """Return a deterministic, label-independent CPU quality heuristic.

    The score rewards informative length, lexical coverage, and explanatory
    connectives. It is useful for a reproducible lab smoke test, but it is not a
    substitute for human evaluation or model-based judging.
    """
    words = [match.group(0).casefold() for match in _WORD_PATTERN.finditer(text)]
    if not words:
        return 0.0

    word_count = len(words)
    informative_length = min(word_count / 40.0, 1.0)
    lexical_coverage = (len(set(words)) / word_count) * min(word_count / 20.0, 1.0)
    marker_count = sum(word in _EXPLANATION_MARKERS for word in words)
    explanation = min(marker_count / 2.0, 1.0)
    return float(0.6 * informative_length + 0.25 * lexical_coverage + 0.15 * explanation)


def pairwise_accuracy(
    examples: list[PreferenceExample],
    chosen_scores: list[float],
    rejected_scores: list[float],
) -> float:
    """Return pairwise accuracy, counting tied scores as half a win."""
    expected_length = len(examples)
    if len(chosen_scores) != expected_length or len(rejected_scores) != expected_length:
        raise ValueError("examples, chosen_scores, and rejected_scores must have equal lengths")
    if any(not math.isfinite(score) for score in (*chosen_scores, *rejected_scores)):
        raise ValueError("scores must contain only finite values")
    if not examples:
        return 0.0

    wins = sum(chosen > rejected for chosen, rejected in zip(chosen_scores, rejected_scores))
    ties = sum(chosen == rejected for chosen, rejected in zip(chosen_scores, rejected_scores))
    return (wins + 0.5 * ties) / expected_length


def write_metrics(metrics: dict[str, MetricValue], output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return out
