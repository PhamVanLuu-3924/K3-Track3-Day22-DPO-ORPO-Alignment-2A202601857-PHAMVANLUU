import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from preference_lab.data import load_jsonl, split_by_prompt
from preference_lab.schemas import PreferenceExample, normalize_text


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    content = "\n".join(json.dumps(row) for row in rows)
    path.write_text(f"{content}\n", encoding="utf-8")


def _example(prompt: str, suffix: str = "") -> PreferenceExample:
    return PreferenceExample(
        prompt=prompt,
        chosen=f"A correct and sufficiently detailed response {suffix}",
        rejected=f"An incorrect and misleading response {suffix}",
    )


def test_load_sample_data() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")

    assert len(examples) == 24
    assert examples[0].prompt == 'Explain the concept of "self-attention" in Transformers.'
    assert all(example.chosen != example.rejected for example in examples)


def test_load_jsonl_reports_invalid_json_line(tmp_path: Path) -> None:
    data_file = tmp_path / "invalid.jsonl"
    data_file.write_text(
        '{"prompt":"p","chosen":"good","rejected":"bad"}\n{not-json}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r":2: invalid JSON"):
        load_jsonl(data_file)


def test_load_jsonl_reports_schema_error_line(tmp_path: Path) -> None:
    data_file = tmp_path / "invalid-schema.jsonl"
    _write_jsonl(data_file, [{"prompt": "p", "chosen": "", "rejected": "bad"}])

    with pytest.raises(ValueError, match=r":1: invalid preference example"):
        load_jsonl(data_file)


def test_load_jsonl_rejects_normalized_duplicate_prompts(tmp_path: Path) -> None:
    data_file = tmp_path / "duplicates.jsonl"
    _write_jsonl(
        data_file,
        [
            {"prompt": "Shared prompt", "chosen": "good one", "rejected": "bad one"},
            {"prompt": " shared   PROMPT ", "chosen": "good two", "rejected": "bad two"},
        ],
    )

    with pytest.raises(ValueError, match=r":2: duplicate prompt; first seen on line 1"):
        load_jsonl(data_file)


def test_pii_guard_is_opt_in(tmp_path: Path) -> None:
    data_file = tmp_path / "pii.jsonl"
    _write_jsonl(
        data_file,
        [
            {
                "prompt": "Email student@example.com with the result",
                "chosen": "Provide a careful response",
                "rejected": "Ignore privacy concerns",
            }
        ],
    )

    assert len(load_jsonl(data_file)) == 1
    with pytest.raises(ValueError, match=r":1: possible email address detected"):
        load_jsonl(data_file, guard_pii=True)


def test_schema_rejects_normalized_and_near_duplicate_responses() -> None:
    with pytest.raises(ValidationError, match="must differ after normalization"):
        PreferenceExample(prompt="p", chosen=" Same  answer ", rejected="same answer")

    with pytest.raises(ValidationError, match="near duplicates"):
        PreferenceExample(
            prompt="p",
            chosen="A sufficiently long answer with the same content.",
            rejected="a sufficiently long answer with the same content!",
        )


def test_split_is_deterministic_and_has_no_prompt_leakage() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")

    train, validation = split_by_prompt(examples, validation_ratio=0.2, seed=42)
    repeated_train, repeated_validation = split_by_prompt(examples, validation_ratio=0.2, seed=42)

    assert (train, validation) == (repeated_train, repeated_validation)
    assert (len(train), len(validation)) == (19, 5)
    assert len(train) + len(validation) == len(examples)
    train_prompts = {normalize_text(example.prompt) for example in train}
    validation_prompts = {normalize_text(example.prompt) for example in validation}
    assert train_prompts.isdisjoint(validation_prompts)


def test_split_keeps_duplicate_prompt_group_together() -> None:
    examples = [
        _example("Shared prompt", "one"),
        _example(" shared   PROMPT ", "two"),
        _example("Second prompt", "three"),
        _example("Third prompt", "four"),
    ]

    train, validation = split_by_prompt(examples, validation_ratio=0.34, seed=7)
    train_prompts = {normalize_text(example.prompt) for example in train}
    validation_prompts = {normalize_text(example.prompt) for example in validation}

    assert len(train) + len(validation) == len(examples)
    assert train_prompts.isdisjoint(validation_prompts)


@pytest.mark.parametrize("validation_ratio", [0.0, 1.0, -0.1, 1.1])
def test_split_rejects_invalid_ratio(validation_ratio: float) -> None:
    with pytest.raises(ValueError, match="validation_ratio must be between 0 and 1"):
        split_by_prompt([_example("prompt")], validation_ratio=validation_ratio)


def test_split_handles_empty_and_single_prompt_groups() -> None:
    assert split_by_prompt([], validation_ratio=0.2) == ([], [])

    examples = [_example("Shared", "one"), _example(" shared ", "two")]
    train, validation = split_by_prompt(examples, validation_ratio=0.2)
    assert train == examples
    assert validation == []
