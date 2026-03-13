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


def test_to_json_string_normalizes_dtype_like_values_from_diff(monkeypatch):
    config = EasyDeLBaseConfig()

    monkeypatch.setattr(
        EasyDeLBaseConfig,
        "to_diff_dict",
        lambda self: {"leftover_dtype": jnp.dtype(jnp.bfloat16)},
    )

    saved = json.loads(config.to_json_string(use_diff=True))
    assert saved["leftover_dtype"] == "bfloat16"


def test_from_pretrained_rehydrates_dtype_strings(tmp_path):
    config = EasyDeLBaseConfig(
        attn_dtype="bfloat16",
        kvdtype="float16",
        attn_softmax_dtype="float32",
        mla_attn_dtype="bf16",
        mla_attn_softmax_dtype="fp32",
    )
    config.save_pretrained(tmp_path)

    reloaded = EasyDeLBaseConfig.from_pretrained(tmp_path)
    assert reloaded.attn_dtype == jnp.dtype(jnp.bfloat16)
    assert reloaded.kvdtype == jnp.dtype(jnp.float16)
    assert reloaded.attn_softmax_dtype == jnp.dtype(jnp.float32)
    assert reloaded.mla_attn_dtype == jnp.dtype(jnp.bfloat16)
    assert reloaded.mla_attn_softmax_dtype == jnp.dtype(jnp.float32)


def test_to_json_string_normalizes_dtype_inside_to_dict_objects(monkeypatch):
    class ToDictWithDtype:
        def to_dict(self):
            return {"inner_dtype": jnp.dtype(jnp.bfloat16)}

    config = EasyDeLBaseConfig()

    monkeypatch.setattr(
        EasyDeLBaseConfig,
        "to_diff_dict",
        lambda self: {"wrapped": ToDictWithDtype()},
    )

    saved = json.loads(config.to_json_string(use_diff=True))
    assert saved["wrapped"]["inner_dtype"] == "bfloat16"


def test_to_json_string_normalizes_torch_dtype(monkeypatch):
    torch = pytest.importorskip("torch")
    config = EasyDeLBaseConfig()

    monkeypatch.setattr(
        EasyDeLBaseConfig,
        "to_diff_dict",
        lambda self: {"dtype": torch.bfloat16},
    )

    saved = json.loads(config.to_json_string(use_diff=True))
    assert saved["dtype"] == "bfloat16"


def test_coerce_runtime_dtype_fields_accepts_torch_dtype_strings():
    config = EasyDeLBaseConfig()
    config.dtype = "torch.bfloat16"
    config.attn_dtype = "torch.float16"
    config._coerce_runtime_dtype_fields()
    assert config.dtype == jnp.dtype(jnp.bfloat16)
    assert config.attn_dtype == jnp.dtype(jnp.float16)


def test_coerce_runtime_dtype_fields_accepts_fp8_fp4_aliases():
    config = EasyDeLBaseConfig()
    config.dtype = "fp8"
    config.attn_dtype = "fp8_e4m3"
    config.kvdtype = "fp4"
    config._coerce_runtime_dtype_fields()
    assert config.dtype == jnp.dtype(jnp.float8_e5m2)
    assert config.attn_dtype == jnp.dtype(jnp.float8_e4m3)
    assert config.kvdtype == jnp.dtype(jnp.float4_e2m1fn)
