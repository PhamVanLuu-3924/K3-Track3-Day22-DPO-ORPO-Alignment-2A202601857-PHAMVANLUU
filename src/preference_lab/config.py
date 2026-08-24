from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as f:
        loaded: object = yaml.safe_load(f)

    if not isinstance(loaded, dict):
        raise TypeError(f"{source}: config root must be a mapping")
    if any(not isinstance(key, str) for key in loaded):
        raise TypeError(f"{source}: config keys must be strings")
    return cast(dict[str, Any], loaded)
