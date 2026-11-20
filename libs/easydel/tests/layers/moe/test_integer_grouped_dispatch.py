"""Integer-mode expert grouped-matmul platform selection.

Outside inference mode (training) the selection is the earlier one, unchanged: only the two measured v5p decode and
W4A4 prefill families select Pallas (the first two tests are the earlier tests, verbatim, and run outside inference
mode; ``test_training_selection_is_unchanged`` pins more shapes, GLM-5.3-Flash's int4 experts included). Under
inference mode the general rule applies: Pallas exactly when ejkernel's ``grouped_matmul_channelwise_pallas_tiles``
gives the call tiles (weight shape, codes' bit width, 128-alignment, 2 MiB RHS buffer and row bounds; no shape lists).
"""

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


def _v5p(monkeypatch, backend="tpu", chip="v5p"):
    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    monkeypatch.setattr(
        tpu, "get_tpu_info", lambda: types.SimpleNamespace(chip_version=types.SimpleNamespace(value=chip))
    )


def _args(shape, dtype):
    e, m, k, n = shape
    return jax.ShapeDtypeStruct((m, k), jnp.bfloat16), jax.ShapeDtypeStruct((e, k, n), dtype)


# What the earlier selection returns outside inference mode (training), on TPU v5p with bfloat16 rows.
@pytest.mark.parametrize(
    "shape,bits,dtype,expected",
    [
        ((128, 1, 2560, 1280), 4, jnp.int4, "pallas"),
        ((128, 128, 640, 2560), 4, jnp.int4, "pallas"),
        ((128, 129, 640, 2560), 4, jnp.int4, "xla"),
        ((128, 1279, 2560, 1280), 4, jnp.int4, "xla"),
        ((128, 1280, 2560, 1280), 4, jnp.int4, "pallas"),
        ((128, 81920, 640, 2560), 4, jnp.int4, "pallas"),
        ((128, 81921, 640, 2560), 4, jnp.int4, "xla"),
        ((128, 80, 2560, 1280), 16, jnp.int4, "pallas"),
        ((128, 128, 640, 2560), 16, jnp.int8, "pallas"),
        ((128, 129, 640, 2560), 16, jnp.int4, "xla"),
        ((128, 1280, 640, 2560), 16, jnp.int4, "xla"),
        ((128, 80, 2560, 1280), 8, jnp.int8, "xla"),
        ((128, 80, 2560, 1280), 8, jnp.int4, "xla"),
        # GLM-5.3-Flash per-chip experts (ep4 x tp4): XLA for every width in training.
        ((72, 256, 4096, 512), 8, jnp.int4, "xla"),
        ((72, 256, 512, 4096), 8, jnp.int4, "xla"),
        ((72, 2048, 4096, 512), 8, jnp.int4, "xla"),
        ((72, 256, 4096, 512), 8, jnp.int8, "xla"),
        ((72, 64, 4096, 512), 16, jnp.int4, "xla"),
        ((72, 64, 512, 4096), 16, jnp.int4, "xla"),
        ((72, 64, 4096, 512), 4, jnp.int4, "xla"),
        ((72, 2048, 512, 4096), 4, jnp.int4, "xla"),
        # DeepSeek-V4.1 per-device experts.
        ((96, 48, 5120, 2304), 16, jnp.int4, "xla"),
        ((96, 1950, 2304, 5120), 4, jnp.int4, "xla"),
    ],
)
def test_training_selection_is_unchanged(monkeypatch, shape, bits, dtype, expected):
    from easydel.utils.inference_mode import set_inference_mode

    _v5p(monkeypatch)
    x, w = _args(shape, dtype)
    with set_inference_mode(False):
        assert mm._channelwise_grouped_platform(x, w, bits) == expected
        assert mm._channelwise_grouped_platform(x, w, bits, force_xla=True) == "xla"


@pytest.mark.parametrize(
    "shape,bits,force,backend,chip,expected",
    [
        ((128, 80, 2560, 1280), 4, False, "tpu", "v5p", "pallas"),
        ((128, 80, 640, 2560), 16, False, "tpu", "v5p", "pallas"),
        ((128, 80, 2560, 1280), 4, True, "tpu", "v5p", "xla"),
        ((128, 256, 2560, 1280), 4, False, "tpu", "v5p", "xla"),
        ((128, 1280, 2560, 1280), 4, False, "tpu", "v5p", "pallas"),
        ((128, 81921, 640, 2560), 4, False, "tpu", "v5p", "xla"),
        ((128, 80, 2560, 1280), 8, False, "tpu", "v5p", "xla"),  # int8 slab 3.1 MiB > 2 MiB
        ((128, 1280, 640, 2560), 16, False, "tpu", "v5p", "pallas"),  # A16 full K up to 12,288 rows
        ((128, 12289, 640, 2560), 16, False, "tpu", "v5p", "xla"),
        ((128, 81920, 640, 2560), 8, False, "tpu", "v5p", "pallas"),  # int8 slab 1.6 MiB
        ((2, 16, 128, 128), 4, False, "tpu", "v5p", "pallas"),
        ((96, 48, 5120, 2304), 16, False, "tpu", "v5p", "pallas"),
        ((96, 1950, 2304, 5120), 16, False, "tpu", "v5p", "pallas"),
        ((72, 256, 160, 256), 16, False, "tpu", "v5p", "xla"),  # K not 128-aligned
        ((128, 80, 2560, 1280), 4, False, "cpu", "v5p", "xla"),
        ((128, 80, 2560, 1280), 4, False, "tpu", "v6e", "xla"),
    ],
)
def test_inference_selection_follows_the_general_rule(monkeypatch, shape, bits, force, backend, chip, expected):
    from easydel.utils.inference_mode import set_inference_mode

    _v5p(monkeypatch, backend, chip)
    x, w = _args(shape, jnp.int4 if bits == 4 else jnp.int8)
    with set_inference_mode():
        assert mm._channelwise_grouped_platform(x, w, bits, force_xla=force) == expected


@pytest.mark.parametrize(
    "shape,bits,expected",
    [
        ((72, 256, 4096, 512), 8, "pallas"),
        ((72, 256, 512, 4096), 8, "pallas"),
        ((72, 2048, 4096, 512), 8, "pallas"),
        ((72, 81920, 512, 4096), 8, "pallas"),
        ((72, 81921, 512, 4096), 8, "xla"),
        ((72, 256, 4096, 1024), 8, "pallas"),  # 2 MiB at four bits
        ((72, 256, 4096, 2048), 8, "xla"),
        ((96, 48, 5120, 2304), 8, "xla"),  # the slab exceeds the buffer even at four bits
        ((72, 256, 4096, 512), 16, "pallas"),
        ((72, 256, 512, 4096), 16, "pallas"),
    ],
)
def test_four_bit_glm_shapes_select_pallas_only_under_inference(monkeypatch, shape, bits, expected):
    from easydel.utils.inference_mode import set_inference_mode

    _v5p(monkeypatch)
    x, w = _args(shape, jnp.int4)
    with set_inference_mode():
        assert mm._channelwise_grouped_platform(x, w, bits) == expected
    with set_inference_mode(False):
        assert mm._channelwise_grouped_platform(x, w, bits) == "xla"
