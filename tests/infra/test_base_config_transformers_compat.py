import pytest

from easydel.infra.base_config import EasyDeLBaseConfig


def test_save_pretrained_works_with_transformers5_generation_api(tmp_path):
    config = EasyDeLBaseConfig()
    config.save_pretrained(tmp_path)
    assert (tmp_path / "config.json").is_file()


def test_save_pretrained_warns_on_generation_parameters(tmp_path):
    config = EasyDeLBaseConfig()
    config.temperature = 0.7

    with pytest.warns(UserWarning, match="Non-default generation parameters"):
        config.save_pretrained(tmp_path)
