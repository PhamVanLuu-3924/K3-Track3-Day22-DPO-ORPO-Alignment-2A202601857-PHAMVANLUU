from __future__ import annotations

import json
import random
import re
from pathlib import Path

from pydantic import ValidationError

from .schemas import PreferenceExample, normalize_text

_PII_PATTERNS = {
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "phone number": re.compile(r"(?<!\w)(?:\+?\d[\d .()-]{7,}\d)(?!\w)"),
    "API key": re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|AKIA[A-Z0-9]{16})\b"),
}


def _find_pii(example: PreferenceExample) -> str | None:
    text = f"{example.prompt}\n{example.chosen}\n{example.rejected}"
    for label, pattern in _PII_PATTERNS.items():
        if pattern.search(text):
            return label
    return None


def load_jsonl(path: str | Path, *, guard_pii: bool = False) -> list[PreferenceExample]:
    """Load and validate preference examples from a JSONL file.

    Errors include the source path and line number. Duplicate prompts are rejected
    after Unicode, case, and whitespace normalization. PII checks are opt-in because
    simple pattern matching can produce false positives for technical datasets.
    """
    source = Path(path)
    examples: list[PreferenceExample] = []
    prompt_lines: dict[str, int] = {}

    with source.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source}:{line_number}: invalid JSON: {exc.msg} (column {exc.colno})"
                ) from exc

            try:
                example = PreferenceExample.model_validate(payload)
            except ValidationError as exc:
                raise ValueError(
                    f"{source}:{line_number}: invalid preference example: {exc}"
                ) from exc

            normalized_prompt = normalize_text(example.prompt)
            if normalized_prompt in prompt_lines:
                first_line = prompt_lines[normalized_prompt]
                raise ValueError(
                    f"{source}:{line_number}: duplicate prompt; first seen on line {first_line}"
                )

            if guard_pii and (pii_label := _find_pii(example)) is not None:
                raise ValueError(f"{source}:{line_number}: possible {pii_label} detected")

            prompt_lines[normalized_prompt] = line_number
            examples.append(example)
    return examples


def split_by_prompt(
    examples: list[PreferenceExample], validation_ratio: float = 0.2, seed: int = 42
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Deterministically split prompt groups to avoid train/validation leakage."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    if not examples:
        return [], []

    grouped: dict[str, list[PreferenceExample]] = {}
    for example in examples:
        grouped.setdefault(normalize_text(example.prompt), []).append(example)

    groups = list(grouped.values())
    random.Random(seed).shuffle(groups)

    if len(groups) == 1:
        return groups[0].copy(), []

    validation_group_count = round(len(groups) * validation_ratio)
    validation_group_count = min(max(1, validation_group_count), len(groups) - 1)

    validation_groups = groups[:validation_group_count]
    train_groups = groups[validation_group_count:]
    train = [example for group in train_groups for example in group]
    validation = [example for group in validation_groups for example in group]
    return train, validation
