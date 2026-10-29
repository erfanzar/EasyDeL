"""Only measured v5p decode and W4A4 prefill families select Pallas."""

import types

import jax
import jax.numpy as jnp
import pytest
from easydel.layers.moe import _moe_module as mm
from jax.experimental.pallas import tpu


@pytest.mark.parametrize(
    "shape,bits,force,backend,chip,expected",
    [
        ((128, 80, 2560, 1280), 4, False, "tpu", "v5p", "pallas"),
        ((128, 80, 640, 2560), 4, False, "tpu", "v5p", "pallas"),
        ((128, 80, 2560, 1280), 4, True, "tpu", "v5p", "xla"),
        ((128, 80, 2560, 1280), 8, False, "tpu", "v5p", "xla"),
        ((128, 80, 2560, 1280), 16, False, "tpu", "v5p", "pallas"),
        ((128, 80, 640, 2560), 16, False, "tpu", "v5p", "pallas"),
        ((128, 80, 640, 2560), 16, True, "tpu", "v5p", "xla"),
        ((128, 256, 2560, 1280), 4, False, "tpu", "v5p", "xla"),
        ((128, 1280, 2560, 1280), 4, False, "tpu", "v5p", "pallas"),
        ((128, 10240, 640, 2560), 4, False, "tpu", "v5p", "pallas"),
        ((128, 81920, 2560, 1280), 4, False, "tpu", "v5p", "pallas"),
        ((128, 81920, 640, 2560), 4, False, "tpu", "v5p", "pallas"),
        ((128, 81920, 640, 2560), 4, True, "tpu", "v5p", "xla"),
        ((128, 81921, 640, 2560), 4, False, "tpu", "v5p", "xla"),
        ((128, 1279, 640, 2560), 4, False, "tpu", "v5p", "xla"),
        ((128, 1280, 640, 2560), 16, False, "tpu", "v5p", "xla"),
        ((128, 81920, 640, 2560), 8, False, "tpu", "v5p", "xla"),
        ((128, 80, 2560, 1280), 4, False, "cpu", "v5p", "xla"),
        ((128, 80, 2560, 1280), 4, False, "tpu", "v6e", "xla"),
        ((2, 16, 128, 128), 4, False, "tpu", "v5p", "xla"),
    ],
)
def test_selects_only_measured_decode_case(monkeypatch, shape, bits, force, backend, chip, expected):
    select = getattr(mm, "_channelwise_grouped_platform", None)
    assert callable(select), "missing narrow dispatch policy"
    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    monkeypatch.setattr(
        tpu, "get_tpu_info", lambda: types.SimpleNamespace(chip_version=types.SimpleNamespace(value=chip))
    )
    e, m, k, n = shape
    x = jax.ShapeDtypeStruct((m, k), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((e, k, n), jnp.int4 if bits == 4 else jnp.int8)
    assert select(x, w, bits, force_xla=force) == expected


@pytest.mark.parametrize("dtype", [jnp.int4, jnp.int8])
def test_weight_only_decode_uses_measured_streaming_path(monkeypatch, dtype):
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(
        tpu, "get_tpu_info", lambda: types.SimpleNamespace(chip_version=types.SimpleNamespace(value="v5p"))
    )
    x = jax.ShapeDtypeStruct((24, 2560), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((128, 2560, 1280), dtype)
    assert mm._channelwise_grouped_platform(x, w, 16) == "pallas"
    assert mm._channelwise_grouped_platform(x, w, 16, force_xla=True) == "xla"
