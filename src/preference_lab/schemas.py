from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, Field, field_validator

_NEAR_DUPLICATE_THRESHOLD = 0.97


def normalize_text(value: str) -> str:
    """Normalize text for reliable equality and duplicate checks."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


class PreferenceExample(BaseModel):
    """One preference pair for DPO/ORPO-style alignment."""

    prompt: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", "chosen", "rejected")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("rejected")
    @classmethod
    def chosen_and_rejected_must_differ(cls, rejected: str, info: Any) -> str:
        chosen = info.data.get("chosen")
        if not isinstance(chosen, str):
            return rejected

        normalized_chosen = normalize_text(chosen)
        normalized_rejected = normalize_text(rejected)
        if normalized_chosen == normalized_rejected:
            raise ValueError("chosen and rejected must differ after normalization")

        similarity = SequenceMatcher(
            None, normalized_chosen, normalized_rejected, autojunk=False
        ).ratio()
        if similarity >= _NEAR_DUPLICATE_THRESHOLD:
            raise ValueError(
                f"chosen and rejected are near duplicates (similarity={similarity:.3f})"
            )
        return rejected
