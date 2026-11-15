"""A8 whole-slab streaming of int8 experts (TPU only).

A8 activations with int8 codes whose 128-aligned ``(K, N)`` expert slab fits 2 MiB stream one whole slab per expert
that has rows, and the Pallas entry applies both scales inside the kernel (``_a8_slab_tiles``), for any expert count:
at up to 1024 rows with 32-row tiles, and up to 81,920 rows with a row tile chosen from the rows per expert (halved for
slabs wider than 4096 in K or N). Pinned here: outputs bit-identical to the XLA entry (bfloat16 and float32) and equal
to an exact float64 reference at decode rows, at the per-chip expert shapes with decode-like routing (one to a few rows
on about half the experts, NaN dispatch-tail rows), at the 1024-row bound, at the tp16 per-chip shapes and at rows that
are not a multiple of the 32-row tile; bit-identical to the XLA entry at ep4 prefill windows with each prefill row
tile, including one group spanning many row tiles, and at prefill rows of other expert shapes (other expert counts,
widths and a halved row tile); all-empty groups (with NaN scales on the unrouted experts) write zeros; the tile policy
at its bounds; the per-chip GLM-5.3-Flash shapes keep the row tiles they were measured with at every row count; and
the compiled program runs one Mosaic kernel that writes the scaled float32 rows.
"""

import importlib
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules import grouped_matmul_channelwise

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires real TPU")

impl = importlib.import_module("ejkernel.kernels._pallas.tpu.grouped_matmul_channelwise._interface")


def _inputs(e, m, k, n, local, active, seed):
    """Rows sorted by expert: ``local`` routed rows over ``active`` experts, then NaN dispatch-tail rows."""
    rng = np.random.default_rng(seed)
    sizes = np.zeros(e, np.int32)
    if local:
        experts = np.sort(rng.choice(e, size=active, replace=False))
        sizes[experts] = 1
        np.add.at(sizes, rng.choice(experts, size=local - active, replace=True), 1)
    x = rng.normal(size=(m, k)).astype(np.float32)
    x[local:] = np.nan
    codes = rng.integers(-127, 128, size=(e, k, n), dtype=np.int8)
    scales = rng.uniform(0.001, 0.01, size=(e, 1, n)).astype(np.float32)
    return x, codes, scales, sizes


def _call(platform, dtype):
    return jax.jit(
        lambda a, b, c, d: grouped_matmul_channelwise(
            a, b, c, d, activation_bits=8, platform=platform, preferred_element_type=dtype
        )
    )


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "e,m,k,n,local,active",
    [
        (72, 256, 4096, 512, 61, 37),
        (72, 256, 512, 4096, 61, 37),
        (72, 1024, 4096, 512, 256, 60),
        (288, 256, 4096, 128, 256, 163),
        (288, 256, 128, 4096, 256, 163),
        (16, 40, 256, 384, 33, 9),
    ],
    ids=["gate_decode", "down_decode", "gate_1024_rows", "tp16_gate", "tp16_down", "rows_off_tile"],
)
def test_slab_matches_xla_and_exact_reference(e, m, k, n, local, active, dtype):
    from ejkernel.kernels._xla.quantized_matmul._integer_quantization import quantize_rows

    x, codes, scales, sizes = _inputs(e, m, k, n, local, active, seed=m + k + n)
    assert impl._a8_slab_tiles(m, 8, e, k, n, jnp.dtype(jnp.int8)) == (32, k, n)
    args = (jnp.asarray(x, jnp.bfloat16), jnp.asarray(codes), jnp.asarray(scales), jnp.asarray(sizes))
    actual = np.asarray(_call("pallas", dtype)(*args), np.float32)
    expected = np.asarray(_call("xla", dtype)(*args), np.float32)
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[local:], 0)
    # Exact reference: the int8 row codes both entries contract (|q|, |codes| <= 127 over K <= 4096 terms is exact in
    # float64), then (dot * row scale) * channel scale.
    valid = jnp.arange(m)[:, None] < local
    q, row_scale = jax.jit(lambda v: quantize_rows(jnp.where(valid, v, 0), 8))(args[0])
    q = np.asarray(q, np.float64)[:local]
    row_scale = np.asarray(row_scale, np.float64)[:local]
    groups = np.repeat(np.arange(e), sizes)
    reference = np.zeros((local, n))
    for g in np.unique(groups):
        sel = groups == g
        reference[sel] = (q[sel] @ codes[g].astype(np.float64)) * row_scale[sel] * scales[g].astype(np.float64)
    tol = 2**-7 if dtype == jnp.bfloat16 else 1e-6
    np.testing.assert_allclose(actual[:local], reference, rtol=tol, atol=tol * np.abs(reference).max())


@pytest.mark.parametrize(
    "e,m,k,n,local,skew",
    [
        (72, 2048, 4096, 512, 1600, False),
        (72, 5120, 512, 4096, 4000, True),
        (72, 20480, 4096, 512, 16000, False),
        (72, 40960, 512, 4096, 32000, True),
        (288, 2048, 4096, 128, 1600, False),
        (32, 8192, 2048, 768, 6500, True),
        (32, 8192, 768, 2048, 6500, False),
        (64, 32768, 256, 7168, 26000, True),
    ],
    ids=[
        "w2048_gate_tm64",
        "w5120_down_tm128_skew",
        "w20480_gate_tm256",
        "w40960_down_tm512_skew",
        "e288_gate_tm64",
        "e32_gate_tm256_skew",
        "e32_down_tm256",
        "e64_wide_n_tm256_halved_skew",
    ],
)
def test_prefill_slab_matches_xla(e, m, k, n, local, skew):
    """Prefill row buffers: the ep4 windows of the per-chip GLM-5.3-Flash experts at each row tile of the prefill
    rule, and other expert counts and widths, including a slab whose row tile is halved for its 7168-wide ``N``."""
    tiles = impl._a8_slab_tiles(m, 8, e, k, n, jnp.dtype(jnp.int8))
    assert tiles is not None and tiles[0] > 32 and tiles[1:] == (k, n), tiles
    rng = np.random.default_rng(m + k)
    ids = rng.integers(0, e, size=local)
    if skew:
        # A third of the routed rows on one expert: its group spans many row tiles.
        ids[: local // 3] = 5
    sizes = np.bincount(ids, minlength=e).astype(np.int32)
    x = rng.normal(size=(m, k)).astype(np.float32)
    x[local:] = np.nan
    args = (
        jnp.asarray(x, jnp.bfloat16),
        jnp.asarray(rng.integers(-127, 128, size=(e, k, n), dtype=np.int8)),
        jnp.asarray(rng.uniform(0.001, 0.01, size=(e, 1, n)), jnp.float32),
        jnp.asarray(sizes),
    )
    actual = _call("pallas", jnp.bfloat16)(*args)
    expected = _call("xla", jnp.bfloat16)(*args)
    assert bool(jnp.array_equal(actual, expected))
    assert bool(jnp.isfinite(actual).all())
    assert not bool(jnp.any(actual[local:] != 0))


def test_slab_with_every_group_empty_writes_zeros():
    x, codes, scales, sizes = _inputs(72, 256, 4096, 512, 0, 0, seed=7)
    assert not sizes.any() and np.isnan(x).all()
    # An unused NaN scale must not reach the output either.
    scales[0, 0, :3] = np.nan
    args = (jnp.asarray(x, jnp.bfloat16), jnp.asarray(codes), jnp.asarray(scales), jnp.asarray(sizes))
    for dtype in (jnp.bfloat16, jnp.float32):
        out = np.asarray(_call("pallas", dtype)(*args), np.float32)
        np.testing.assert_array_equal(out, 0)


@pytest.mark.parametrize(
    "m,bits,e,k,n,dtype,expected",
    [
        (256, 8, 72, 4096, 512, jnp.int8, (32, 4096, 512)),
        (1, 8, 72, 512, 4096, jnp.int8, (32, 512, 4096)),
        (1024, 8, 72, 4096, 512, jnp.int8, (32, 4096, 512)),
        (256, 8, 288, 4096, 128, jnp.int8, (32, 4096, 128)),
        (1025, 8, 72, 4096, 512, jnp.int8, (64, 4096, 512)),
        (2048, 8, 72, 4096, 512, jnp.int8, (64, 4096, 512)),
        (3072, 8, 72, 512, 4096, jnp.int8, (64, 512, 4096)),
        (5120, 8, 72, 4096, 512, jnp.int8, (128, 4096, 512)),
        (10240, 8, 72, 512, 4096, jnp.int8, (128, 512, 4096)),
        (20480, 8, 72, 4096, 512, jnp.int8, (256, 4096, 512)),
        (40960, 8, 72, 512, 4096, jnp.int8, (512, 512, 4096)),
        (81920, 8, 72, 4096, 512, jnp.int8, (512, 4096, 512)),
        (81921, 8, 72, 4096, 512, jnp.int8, None),
        (1025, 8, 288, 4096, 128, jnp.int8, (64, 4096, 128)),
        (5120, 8, 128, 640, 2560, jnp.int8, (64, 640, 2560)),
        (8192, 8, 32, 2048, 768, jnp.int8, (256, 2048, 768)),
        (32768, 8, 32, 768, 2048, jnp.int8, (512, 768, 2048)),
        (32768, 8, 64, 256, 7168, jnp.int8, (256, 256, 7168)),
        (32768, 8, 64, 7168, 256, jnp.int8, (256, 7168, 256)),
        (81920, 8, 8, 128, 16384, jnp.int8, (128, 128, 16384)),
        (81921, 8, 8, 128, 16384, jnp.int8, None),
        (8192, 8, 32, 2048, 1536, jnp.int8, None),
        (0, 8, 72, 4096, 512, jnp.int8, None),
        (256, 4, 72, 4096, 512, jnp.int4, None),
        (256, 16, 72, 4096, 512, jnp.int8, None),
        (256, 8, 72, 4096, 512, jnp.int4, None),
        (256, 8, 128, 2560, 1280, jnp.int8, None),
        (256, 8, 72, 4096, 640, jnp.int8, None),
        (256, 8, 72, 160, 256, jnp.int8, None),
        (256, 8, 72, 256, 192, jnp.int8, None),
    ],
)
def test_slab_tile_policy(m, bits, e, k, n, dtype, expected):
    assert impl._a8_slab_tiles(m, bits, e, k, n, jnp.dtype(dtype)) == expected


# The row tiles the per-chip GLM-5.3-Flash experts (ep4 x tp4) were measured and served with, as (last row, tile_m)
# ranges over the dispatch rows: 32 up to 1024 rows, then by rows per expert (72 experts) 64 up to 55 (3,960 rows),
# 128 up to 200 (14,400), 256 up to 400 (28,800) and 512 up to the 81,920-row cap; no slab tiles outside 1..81,920.
_GLM_ROW_TILES = ((1024, 32), (3960, 64), (14400, 128), (28800, 256), (81920, 512))


@pytest.mark.parametrize("k,n", [(4096, 512), (512, 4096)], ids=["gate_up", "down"])
def test_glm_expert_shapes_keep_their_measured_tiles_at_every_row_count(k, n):
    expected = {0: None, 81921: None, 100000: None}
    first = 1
    for last, tile_m in _GLM_ROW_TILES:
        expected.update({m: (tile_m, k, n) for m in range(first, last + 1)})
        first = last + 1
    for m, tiles in expected.items():
        assert impl._a8_slab_tiles(m, 8, 72, k, n, jnp.dtype(jnp.int8)) == tiles, m
    # The public helper easydel's dispatch selection asks gives the same tiles.
    from ejkernel.modules import grouped_matmul_channelwise_slab_tiles

    for m in (1, 1024, 1025, 3960, 3961, 14400, 14401, 28800, 28801, 81920, 81921):
        assert grouped_matmul_channelwise_slab_tiles(m, (72, k, n), jnp.int8, activation_bits=8) == expected.get(m)


def test_slab_compiles_one_kernel_writing_scaled_f32_rows():
    args = (
        jax.ShapeDtypeStruct((256, 4096), jnp.bfloat16),
        jax.ShapeDtypeStruct((72, 4096, 512), jnp.int8),
        jax.ShapeDtypeStruct((72, 1, 512), jnp.float32),
        jax.ShapeDtypeStruct((72,), jnp.int32),
    )
    text = _call("pallas", jnp.bfloat16).lower(*args).compile().as_text()
    kernels = [line for line in text.splitlines() if "tpu_custom_call" in line and "custom-call(" in line]
    assert len(kernels) == 1, kernels
    # The kernel writes the (row scale x channel scale)-scaled rows in float32, not the raw int32 dot.
    assert re.search(r"=\s*f32\[256,512\]", kernels[0]), kernels[0][:300]
