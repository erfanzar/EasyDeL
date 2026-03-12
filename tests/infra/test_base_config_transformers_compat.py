import json

import jax.numpy as jnp
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


def test_save_pretrained_serializes_nested_dtype_like_fields(tmp_path):
    config = EasyDeLBaseConfig()
    config.custom_dtype = jnp.dtype(jnp.bfloat16)
    config.nested_dtype_values = {
        "kvdtype": jnp.dtype(jnp.float16),
        "values": [jnp.bfloat16],
    }

    config.save_pretrained(tmp_path)

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["custom_dtype"] == "bfloat16"
    assert saved["nested_dtype_values"]["kvdtype"] == "float16"
    assert saved["nested_dtype_values"]["values"] == ["bfloat16"]
