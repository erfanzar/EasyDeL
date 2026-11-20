"""General byte-budget tile rule of the TPU Pallas channelwise grouped matmul (rule: pure shape; outputs: TPU only).

The tile and path rules of ``grouped_matmul_channelwise``'s Pallas entry read only the weight shape, the codes' bit
width (``jax.dtypes.itemsize_bits``), 128-alignment, the 2 MiB RHS buffer and row bounds, for any expert count. Pinned:

* every expert family the former measured lists named (``_MEASURED_DECODE_FAMILIES`` / ``_MEASURED_PREFILL_FAMILIES``)
  gets, at the row counts those lists covered and for int4 and int8 codes, exactly the A16 tiles the lists gave (the
  table was generated from the former policy), and the former W4A4 families keep their streaming tiles;
* A8 with int4 codes streams one whole slab under the same budget counted at four bits: the per-chip GLM-5.3-Flash
  shapes get the int8 slab's tiles at every row count, and int4 slabs up to twice the int8 element count qualify;
* the public helper easydel's selection asks gives the same tiles;
* W4A8 slab outputs equal the XLA entry's bitwise and W4A16 full-K outputs equal it to float32 summation order, at small
  and at the GLM-like per-chip shapes, decode and prefill rows; the int4 codes stay packed in HBM.
"""

import importlib
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules import grouped_matmul_channelwise, grouped_matmul_channelwise_pallas_tiles

impl = importlib.import_module("ejkernel.kernels._pallas.tpu.grouped_matmul_channelwise._interface")
_on_tpu = pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires real TPU")

# (rows, E, K, N, codes dtype, A16 tiles) of the former ``_weight_only_streaming_tiles`` for every family of
# ``_MEASURED_DECODE_FAMILIES`` (rows 1..128) and ``_MEASURED_PREFILL_FAMILIES`` (rows 129..12,288).
_FORMER_A16_TILES = (
    (1, 96, 2304, 5120, jnp.int4, (16, 2304, 1280)),
    (16, 96, 2304, 5120, jnp.int4, (16, 2304, 1280)),
    (48, 96, 2304, 5120, jnp.int4, (16, 2304, 1280)),
    (128, 96, 2304, 5120, jnp.int4, (16, 2304, 1280)),
    (1, 96, 2304, 5120, jnp.int8, (16, 2304, 640)),
    (16, 96, 2304, 5120, jnp.int8, (16, 2304, 640)),
    (48, 96, 2304, 5120, jnp.int8, (16, 2304, 640)),
    (128, 96, 2304, 5120, jnp.int8, (16, 2304, 640)),
    (1, 96, 5120, 2304, jnp.int4, (16, 5120, 768)),
    (16, 96, 5120, 2304, jnp.int4, (16, 5120, 768)),
    (48, 96, 5120, 2304, jnp.int4, (16, 5120, 768)),
    (128, 96, 5120, 2304, jnp.int4, (16, 5120, 768)),
    (1, 96, 5120, 2304, jnp.int8, (16, 5120, 384)),
    (16, 96, 5120, 2304, jnp.int8, (16, 5120, 384)),
    (48, 96, 5120, 2304, jnp.int8, (16, 5120, 384)),
    (128, 96, 5120, 2304, jnp.int8, (16, 5120, 384)),
    (1, 96, 5120, 4608, jnp.int4, (16, 5120, 768)),
    (16, 96, 5120, 4608, jnp.int4, (16, 5120, 768)),
    (48, 96, 5120, 4608, jnp.int4, (16, 5120, 768)),
    (128, 96, 5120, 4608, jnp.int4, (16, 5120, 768)),
    (1, 96, 5120, 4608, jnp.int8, (16, 5120, 384)),
    (16, 96, 5120, 4608, jnp.int8, (16, 5120, 384)),
    (48, 96, 5120, 4608, jnp.int8, (16, 5120, 384)),
    (128, 96, 5120, 4608, jnp.int8, (16, 5120, 384)),
    (1, 128, 640, 2560, jnp.int4, (16, 640, 2560)),
    (16, 128, 640, 2560, jnp.int4, (16, 640, 2560)),
    (48, 128, 640, 2560, jnp.int4, (16, 640, 2560)),
    (128, 128, 640, 2560, jnp.int4, (16, 640, 2560)),
    (1, 128, 640, 2560, jnp.int8, (16, 640, 2560)),
    (16, 128, 640, 2560, jnp.int8, (16, 640, 2560)),
    (48, 128, 640, 2560, jnp.int8, (16, 640, 2560)),
    (128, 128, 640, 2560, jnp.int8, (16, 640, 2560)),
    (1, 128, 2560, 1280, jnp.int4, (16, 2560, 1280)),
    (16, 128, 2560, 1280, jnp.int4, (16, 2560, 1280)),
    (48, 128, 2560, 1280, jnp.int4, (16, 2560, 1280)),
    (128, 128, 2560, 1280, jnp.int4, (16, 2560, 1280)),
    (1, 128, 2560, 1280, jnp.int8, (16, 2560, 640)),
    (16, 128, 2560, 1280, jnp.int8, (16, 2560, 640)),
    (48, 128, 2560, 1280, jnp.int8, (16, 2560, 640)),
    (128, 128, 2560, 1280, jnp.int8, (16, 2560, 640)),
    (129, 96, 2304, 5120, jnp.int4, (64, 2304, 1280)),
    (256, 96, 2304, 5120, jnp.int4, (64, 2304, 1280)),
    (1950, 96, 2304, 5120, jnp.int4, (64, 2304, 1280)),
    (12288, 96, 2304, 5120, jnp.int4, (64, 2304, 1280)),
    (129, 96, 2304, 5120, jnp.int8, (64, 2304, 640)),
    (256, 96, 2304, 5120, jnp.int8, (64, 2304, 640)),
    (1950, 96, 2304, 5120, jnp.int8, (64, 2304, 640)),
    (12288, 96, 2304, 5120, jnp.int8, (64, 2304, 640)),
    (129, 96, 5120, 2304, jnp.int4, (64, 5120, 768)),
    (256, 96, 5120, 2304, jnp.int4, (64, 5120, 768)),
    (1950, 96, 5120, 2304, jnp.int4, (64, 5120, 768)),
    (12288, 96, 5120, 2304, jnp.int4, (64, 5120, 768)),
    (129, 96, 5120, 2304, jnp.int8, (64, 5120, 384)),
    (256, 96, 5120, 2304, jnp.int8, (64, 5120, 384)),
    (1950, 96, 5120, 2304, jnp.int8, (64, 5120, 384)),
    (12288, 96, 5120, 2304, jnp.int8, (64, 5120, 384)),
)


@pytest.mark.parametrize("m,e,k,n,dtype,expected", _FORMER_A16_TILES)
def test_former_measured_a16_families_keep_their_tiles(m, e, k, n, dtype, expected):
    assert impl._weight_only_streaming_tiles(m, e, k, n, jnp.dtype(dtype)) == expected
    assert impl._pallas_path_tiles(m, 16, e, k, n, jnp.dtype(dtype)) == expected
    assert grouped_matmul_channelwise_pallas_tiles(m, (e, k, n), dtype, activation_bits=16) == expected


@pytest.mark.parametrize("k,n,expected", [(2560, 1280, (32, 2560, 640)), (640, 2560, (32, 640, 1280))])
def test_former_w4a4_families_keep_their_streaming_tiles(k, n, expected):
    for m in (1, 24, 80, 128, 1280, 10240, 81920):
        assert impl._pallas_path_tiles(m, 4, 128, k, n, jnp.dtype(jnp.int4)) == expected, m
    for m in (129, 1279, 81921):
        assert impl._pallas_path_tiles(m, 4, 128, k, n, jnp.dtype(jnp.int4)) is None, m


# The int8 slab row tiles of the per-chip GLM-5.3-Flash experts (``test_grouped_a8_slab``): (last row, tile_m).
_GLM_ROW_TILES = ((1024, 32), (3960, 64), (14400, 128), (28800, 256), (81920, 512))


@pytest.mark.parametrize("k,n", [(4096, 512), (512, 4096)], ids=["gate_up", "down"])
def test_int4_a8_slab_matches_the_int8_slab_tiles_on_glm_shapes(k, n):
    first = 1
    for last, tile_m in _GLM_ROW_TILES:
        for m in sorted({first, (first + last) // 2, last}):
            for dtype in (jnp.int4, jnp.int8):
                assert impl._a8_slab_tiles(m, 8, 72, k, n, jnp.dtype(dtype)) == (tile_m, k, n), (m, dtype)
                assert grouped_matmul_channelwise_pallas_tiles(m, (72, k, n), dtype, activation_bits=8) == (
                    tile_m,
                    k,
                    n,
                )
        first = last + 1
    for m in (0, 81921):
        assert impl._a8_slab_tiles(m, 8, 72, k, n, jnp.dtype(jnp.int4)) is None


@pytest.mark.parametrize(
    "m,e,k,n,int8_tiles,int4_tiles",
    [
        (256, 72, 4096, 1024, None, (32, 4096, 1024)),  # 4 MiB as int8, 2 MiB as int4
        (256, 72, 8192, 512, None, (32, 8192, 512)),
        (256, 72, 4096, 2048, None, None),  # 4 MiB even at four bits
        (256, 72, 4096, 640, None, (32, 4096, 640)),
        (2048, 32, 2048, 1536, None, (128, 2048, 1536)),  # 64 rows per expert: the 128-row prefill tile
        (256, 72, 4096, 512, (32, 4096, 512), (32, 4096, 512)),
        (256, 72, 160, 256, None, None),  # K not 128-aligned
        (256, 72, 256, 192, None, None),  # N not 128-aligned
    ],
)
def test_slab_budget_counts_code_bits(m, e, k, n, int8_tiles, int4_tiles):
    assert impl._a8_slab_tiles(m, 8, e, k, n, jnp.dtype(jnp.int8)) == int8_tiles
    assert impl._a8_slab_tiles(m, 8, e, k, n, jnp.dtype(jnp.int4)) == int4_tiles
    # A4 and A16 never take the A8 slab.
    assert impl._a8_slab_tiles(m, 4, e, k, n, jnp.dtype(jnp.int4)) is None
    assert impl._a8_slab_tiles(m, 16, e, k, n, jnp.dtype(jnp.int4)) is None


@pytest.mark.parametrize(
    "m,k,n,dtype,expected",
    [
        (256, 4096, 512, jnp.int4, (64, 4096, 512)),
        (256, 512, 4096, jnp.int4, (64, 512, 4096)),
        (128, 4096, 512, jnp.int4, (16, 4096, 512)),
        (5120, 512, 4096, jnp.int4, (64, 512, 4096)),
        (256, 4096, 512, jnp.int8, (64, 4096, 512)),
        (12289, 4096, 512, jnp.int4, None),
        (0, 4096, 512, jnp.int4, None),
        (256, 4096, 640, jnp.int4, (64, 4096, 640)),
        (256, 4096, 640, jnp.int8, (64, 4096, 128)),
        (256, 160, 256, jnp.int4, None),
    ],
)
def test_a16_full_k_rule(m, k, n, dtype, expected):
    assert impl._weight_only_full_k_tiles(m, 72, k, n, jnp.dtype(dtype)) == expected
    assert impl._pallas_path_tiles(m, 16, 72, k, n, jnp.dtype(dtype)) == expected


def _inputs(e, m, k, n, local, active, seed, skew=False):
    """Rows sorted by expert: ``local`` routed rows over ``active`` experts, then NaN dispatch-tail rows; int4 codes."""
    rng = np.random.default_rng(seed)
    sizes = np.zeros(e, np.int32)
    if local:
        experts = np.sort(rng.choice(e, size=active, replace=False))
        sizes[experts] = 1
        np.add.at(sizes, rng.choice(experts, size=local - active, replace=True), 1)
        if skew:
            sizes[:] = 0
            ids = rng.choice(experts, size=local)
            ids[: local // 3] = experts[0]
            sizes = np.bincount(ids, minlength=e).astype(np.int32)
    x = rng.normal(size=(m, k)).astype(np.float32)
    x[local:] = np.nan
    codes = rng.integers(-8, 8, size=(e, k, n), dtype=np.int8)
    scales = rng.uniform(0.001, 0.01, size=(e, 1, n)).astype(np.float32)
    return (
        jnp.asarray(x, jnp.bfloat16),
        jnp.asarray(codes).astype(jnp.int4),
        jnp.asarray(scales),
        jnp.asarray(sizes),
    )


def _call(platform, bits, dtype):
    return jax.jit(
        lambda a, b, c, d: grouped_matmul_channelwise(
            a, b, c, d, activation_bits=bits, platform=platform, preferred_element_type=dtype
        )
    )


_SHAPES = [
    (16, 40, 256, 384, 33, 9, False),
    (72, 256, 4096, 512, 60, 40, False),
    (72, 256, 512, 4096, 60, 40, False),
    (72, 2048, 4096, 512, 1600, 72, False),
    (72, 5120, 512, 4096, 4000, 72, True),
    (72, 20480, 4096, 512, 16000, 72, False),
]
_IDS = ["small", "glm_gate_decode", "glm_down_decode", "glm_gate_w2048", "glm_down_w5120_skew", "glm_gate_w20480"]


@_on_tpu
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("e,m,k,n,local,active,skew", _SHAPES, ids=_IDS)
def test_w4a8_slab_is_bitwise_the_xla_entry(e, m, k, n, local, active, skew, dtype):
    assert impl._a8_slab_tiles(m, 8, e, k, n, jnp.dtype(jnp.int4)) is not None
    args = _inputs(e, m, k, n, local, active, seed=m + k + n, skew=skew)
    actual = np.asarray(_call("pallas", 8, dtype)(*args), np.float32)
    expected = np.asarray(_call("xla", 8, dtype)(*args), np.float32)
    assert int(np.sum(actual != expected)) == 0
    assert np.isfinite(actual).all() and float(np.abs(expected).max()) > 0
    np.testing.assert_array_equal(actual[local:], 0)


@_on_tpu
def test_w4a8_slab_with_every_group_empty_writes_zeros():
    x, codes, scales, _ = _inputs(72, 256, 4096, 512, 0, 0, seed=3)
    scales = scales.at[0, 0, :3].set(jnp.nan)
    sizes = jnp.zeros((72,), jnp.int32)
    out = np.asarray(_call("pallas", 8, jnp.float32)(x, codes, scales, sizes))
    np.testing.assert_array_equal(out, 0)


@_on_tpu
@pytest.mark.parametrize("e,m,k,n,local,active,skew", _SHAPES[:5], ids=_IDS[:5])
def test_w4a16_full_k_matches_the_xla_entry(e, m, k, n, local, active, skew):
    assert impl._weight_only_full_k_tiles(m, e, k, n, jnp.dtype(jnp.int4)) is not None
    args = _inputs(e, m, k, n, local, active, seed=m + k + n + 1, skew=skew)
    actual = np.asarray(_call("pallas", 16, jnp.float32)(*args))
    expected = np.asarray(_call("xla", 16, jnp.float32)(*args))
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[local:], 0)
    # Both sum exact bf16 x int4 products in float32; only the summation order may differ.
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5 * float(np.abs(expected).max()))


@_on_tpu
def test_w4a8_slab_compiles_one_kernel_reading_packed_int4():
    args = (
        jax.ShapeDtypeStruct((256, 4096), jnp.bfloat16),
        jax.ShapeDtypeStruct((72, 4096, 512), jnp.int4),
        jax.ShapeDtypeStruct((72, 1, 512), jnp.float32),
        jax.ShapeDtypeStruct((72,), jnp.int32),
    )
    text = _call("pallas", 8, jnp.bfloat16).lower(*args).compile().as_text()
    kernels = [line for line in text.splitlines() if "tpu_custom_call" in line and "custom-call(" in line]
    assert len(kernels) == 1, kernels
    assert re.search(r"=\s*f32\[256,512\]", kernels[0]), kernels[0][:300]
    # No widened copy of the weights is materialized outside the kernel.
    assert not re.search(r"s8\[72,4096,512\]", text)
