from pathlib import Path

import pytest

from preference_lab.config import load_config


def test_load_config_returns_mapping() -> None:
    config = load_config("configs/local.yaml")

    assert config["seed"] == 42
    assert config["training"]["method"] == "dpo"


@pytest.mark.parametrize(
    "content",
    [
        "",
        "- first\n- second\n",
    ],
)
def test_load_config_rejects_non_mapping_root(tmp_path: Path, content: str) -> None:
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text(content, encoding="utf-8")

    with pytest.raises(TypeError, match="config root must be a mapping"):
        load_config(config_file)


def test_load_config_rejects_non_string_keys(tmp_path: Path) -> None:
    config_file = tmp_path / "invalid-key.yaml"
    config_file.write_text("1: value\n", encoding="utf-8")

    with pytest.raises(TypeError, match="config keys must be strings"):
        load_config(config_file)
