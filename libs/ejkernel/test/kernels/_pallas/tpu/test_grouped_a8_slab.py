"""A8 whole-slab streaming of int8 experts (TPU only).

A8 activations with int8 codes whose 128-aligned ``(K, N)`` expert slab fits 2 MiB stream one whole slab per expert
that has rows, and the Pallas entry applies both scales inside the kernel (``_a8_slab_tiles``): at up to 1024 rows with
32-row tiles, and on the measured prefill families (the per-chip experts of GLM-5.3-Flash served at ep4 x tp4) up to
81,920 rows with a row tile chosen from the rows per expert. Pinned here: outputs bit-identical to the XLA entry
(bfloat16 and float32) and equal to an exact float64 reference at decode rows, at the per-chip expert shapes with
decode-like routing (one to a few rows on about half the experts, NaN dispatch-tail rows), at the 1024-row bound, at the
tp16 per-chip shapes and at rows that are not a multiple of the 32-row tile; bit-identical to the XLA entry at ep4
prefill windows with each prefill row tile, including one group spanning many row tiles; all-empty groups (with NaN
scales on the unrouted experts) write zeros; the tile policy at its bounds; and the compiled program runs one Mosaic
kernel that writes the scaled float32 rows.
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
    "m,k,n,local,skew",
    [
        (2048, 4096, 512, 1600, False),
        (5120, 512, 4096, 4000, True),
        (20480, 4096, 512, 16000, False),
        (40960, 512, 4096, 32000, True),
    ],
    ids=["w2048_gate_tm64", "w5120_down_tm128_skew", "w20480_gate_tm256", "w40960_down_tm512_skew"],
)
def test_prefill_slab_matches_xla(m, k, n, local, skew):
    """ep4 prefill windows of the per-chip GLM-5.3-Flash experts, each row tile of the prefill rule."""
    e = 72
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
        (1025, 8, 288, 4096, 128, jnp.int8, None),
        (5120, 8, 128, 640, 2560, jnp.int8, None),
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
