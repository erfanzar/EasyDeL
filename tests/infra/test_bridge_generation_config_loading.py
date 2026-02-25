import json

from easydel.infra.mixins.bridge import _load_generation_config


def test_load_generation_config_from_local_path(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    generation_config_file = model_dir / "generation_config.json"
    generation_config_file.write_text(json.dumps({"eos_token_id": [154820, 154827, 154829]}))

    generation_config = _load_generation_config(model_dir)

    assert generation_config is not None
    assert generation_config.eos_token_id == [154820, 154827, 154829]


def test_load_generation_config_from_subfolder(tmp_path):
    model_dir = tmp_path / "model"
    sub_dir = model_dir / "checkpoint-1"
    sub_dir.mkdir(parents=True)
    generation_config_file = sub_dir / "generation_config.json"
    generation_config_file.write_text(json.dumps({"eos_token_id": [10, 20]}))

    generation_config = _load_generation_config(model_dir, subfolder="checkpoint-1")

    assert generation_config is not None
    assert generation_config.eos_token_id == [10, 20]


def test_load_generation_config_returns_none_when_missing(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    generation_config = _load_generation_config(model_dir, local_files_only=True)

    assert generation_config is None
