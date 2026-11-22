# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pallas TPU sort-free selection (``topk`` modes ``"exact_mask"`` / ``"kth"``) against ``jax.lax.top_k``.

The reference is ``jax.lax.top_k(x, k)`` itself: its indices are scattered
into a mask in NumPy and its last value is the threshold. Every case forces the
Pallas config, so Mosaic compiles the kernel for real on the TPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.topk._pallas_select_fwd import select_block_rows, topk_select_tpu
from ejkernel.modules import topk
from ejkernel.modules.operations.configs import TopKConfig

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Mosaic lowering needs a TPU")

_PALLAS = TopKConfig(platform="pallas", backend="tpu")
_MIN = np.finfo(np.float32).min


def _reference(x, k):
    values, indices = jax.jit(lambda a: jax.lax.top_k(a, k))(x)
    mask = np.zeros(x.shape, dtype=bool)
    np.put_along_axis(mask, np.asarray(indices), True, axis=-1)
    return mask, values[..., -1]


def _bits(a):
    a = np.asarray(jnp.asarray(a).astype(jnp.float32))
    nan = np.isnan(a)
    return np.where(nan, 0, a).view(np.int32), nan


def _scores(kind, shape, seed):
    rng = np.random.default_rng(seed)
    if kind == "normal":
        return rng.standard_normal(shape).astype(np.float32)
    if kind == "ties":
        return rng.integers(0, 6, size=shape).astype(np.float32)
    if kind == "indexer":
        # Head-weighted ReLU sums with signed weights: negatives, exact zeros, ties, finfo.min fill and rows with
        # fewer than k candidates.
        x = rng.standard_normal(shape).astype(np.float32) * 0.05
        x = np.where(rng.random(shape) < 0.3, 0.0, x)
        x = np.where(rng.random(shape) < 0.5, np.round(x, 2), x)
        valid = rng.random(shape) < 0.6
        valid[..., : max(shape[-2] // 8, 1), :] = rng.random((*shape[:-2], max(shape[-2] // 8, 1), shape[-1])) < 0.02
        x = np.where(valid, x, _MIN)
        x[..., 0, :5] = -0.0
        return x
    x = rng.integers(-3, 4, size=shape).astype(np.float32)
    for value, frac in ((np.inf, 0.05), (-np.inf, 0.3), (np.nan, 0.03), (0.0, 0.1), (-0.0, 0.1)):
        x[rng.random(shape) < frac] = value
    x.view(np.int32)[rng.random(shape) < 0.02] = np.int32(-4194304)  # negative-sign NaN
    x.view(np.int32)[rng.random(shape) < 0.01] = np.int32(-1)  # all-ones NaN: the minimum key
    return x


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize(
    ("kind", "shape", "k"),
    [
        ("normal", (256, 8192), 2048),
        ("ties", (128, 4096), 1500),
        ("specials", (64, 1024), 300),
        ("indexer", (2, 256, 1024), 128),
        ("indexer", (1, 300, 1000), 100),  # unaligned rows and width
        ("normal", (13, 1000), 999),
        ("ties", (16, 512), 1),
        ("specials", (8, 256), 256),  # k == width
        ("normal", (16, 131072), 2048),  # long-context width
    ],
)
def test_bit_identical_to_lax_top_k(dtype, kind, shape, k):
    x = jnp.asarray(_scores(kind, shape, seed=k + shape[-1])).astype(dtype)
    want_mask, want_kth = _reference(x, k)

    got = topk(x, k, mode="exact_mask", cfg=_PALLAS)
    assert got.dtype == jnp.bool_ and got.shape == x.shape
    np.testing.assert_array_equal(np.asarray(got), want_mask)
    assert (np.asarray(got).sum(-1) == k).all()

    kth = topk(x, k, mode="kth", cfg=_PALLAS)
    assert kth.dtype == x.dtype and kth.shape == x.shape[:-1]
    got_bits, got_nan = _bits(kth)
    want_bits, want_nan = _bits(want_kth)
    np.testing.assert_array_equal(got_nan, want_nan)
    np.testing.assert_array_equal(got_bits, want_bits)  # -0.0 and NaN payloads included


@pytest.mark.parametrize("kind", ["normal", "ties", "specials"])
def test_per_row_dynamic_k(kind):
    shape = (64, 2048)
    x = jnp.asarray(_scores(kind, shape, seed=3))
    ks = jnp.asarray(np.random.default_rng(4).integers(0, shape[-1] + 1, size=shape[0]), jnp.int32)
    ks = ks.at[:4].set(jnp.asarray([0, 1, shape[-1] - 1, shape[-1]], jnp.int32))
    values, order = jax.jit(lambda a: jax.lax.top_k(a, shape[-1]))(x)
    order, values = np.asarray(order), values
    want = np.zeros(shape, dtype=bool)
    for r, k in enumerate(np.asarray(ks)):
        want[r, order[r, :k]] = True
    got = jax.jit(lambda a, kk: topk(a, kk, mode="exact_mask", cfg=_PALLAS))(x, ks)
    np.testing.assert_array_equal(np.asarray(got), want)
    kth = jax.jit(lambda a, kk: topk(a, kk, mode="kth", cfg=_PALLAS))(x, ks)
    want_kth = jnp.take_along_axis(values, jnp.maximum(ks, 1)[:, None] - 1, axis=-1)[:, 0]
    got_bits, got_nan = _bits(kth)
    want_bits, want_nan = _bits(want_kth)
    np.testing.assert_array_equal(got_nan, want_nan)
    np.testing.assert_array_equal(got_bits, want_bits)


def test_matches_xla_reference_and_heuristic_routes_to_pallas():
    from ejkernel.modules.operations.topk import TopK
    from ejkernel.ops import Invocation

    x = jnp.asarray(_scores("indexer", (1, 512, 8192), seed=9))
    xla = TopKConfig(platform="xla", backend="any")
    for mode in ("exact_mask", "kth"):
        inv = Invocation(
            op_id="topk", args=(jax.ShapeDtypeStruct((512, 8192), jnp.float32), 2048), kwargs={"mode": mode}
        )
        assert TopK().heuristic_cfg(inv).platform == "pallas"
        got, want = topk(x, 2048, mode=mode, cfg=_PALLAS), topk(x, 2048, mode=mode, cfg=xla)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize(
    ("shape", "dtype", "mode", "platform"),
    [
        ((8192, 65536), jnp.bfloat16, "kth", "pallas"),
        ((8192, 131072), jnp.float32, "exact_mask", "pallas"),  # fp32 mask still wins above the width cap
        ((8192, 131072), jnp.bfloat16, "exact_mask", "xla"),
        ((8192, 131072), jnp.float32, "kth", "xla"),
        ((32, 8192), jnp.float32, "exact_mask", "xla"),  # too few rows
        ((8192, 512), jnp.float32, "kth", "xla"),  # too narrow
    ],
)
def test_heuristic_routes_selection_by_measured_regime(shape, dtype, mode, platform):
    from ejkernel.modules.operations.topk import TopK
    from ejkernel.ops import Invocation

    inv = Invocation(op_id="topk", args=(jax.ShapeDtypeStruct(shape, dtype), 2048), kwargs={"mode": mode})
    assert TopK().heuristic_cfg(inv).platform == platform


def test_non_trailing_axis_and_jit_lowering():
    x = jnp.asarray(_scores("normal", (4, 1024, 3), seed=11))
    got = topk(x, 100, axis=1, mode="exact_mask", cfg=_PALLAS)
    want, _ = _reference(jnp.moveaxis(x, 1, -1), 100)
    np.testing.assert_array_equal(np.asarray(got), np.moveaxis(want, -1, 1))
    compiled = jax.jit(lambda a: topk(a, 256, mode="kth", cfg=_PALLAS)).lower(jnp.zeros((256, 8192))).compile()
    assert "topk_select_threshold" in compiled.as_text()


def test_detached_under_differentiation():
    x = jnp.asarray(_scores("normal", (64, 1024), seed=12))

    def loss(s):
        return jnp.sum(jnp.where(topk(s, 64, mode="exact_mask", cfg=_PALLAS), s, 0.0)) + jnp.sum(
            topk(s, 64, mode="kth", cfg=_PALLAS)
        )

    grad = jax.jit(jax.grad(loss))(x)
    want, _ = _reference(x, 64)
    np.testing.assert_array_equal(np.asarray(grad), want.astype(np.float32))


def test_invalid_arguments():
    with pytest.raises(ValueError, match="mode='kth'"):
        topk(jnp.zeros((64, 1024), jnp.float32), 0, mode="kth", cfg=_PALLAS)
    with pytest.raises(ValueError, match="mode='kth'"):
        topk(jnp.zeros((64, 1024), jnp.float32), 1025, mode="kth", cfg=_PALLAS)
    with pytest.raises(ValueError, match="mode must be one of"):
        topk_select_tpu(jnp.zeros((8, 128), jnp.float32), jnp.ones((8,), jnp.int32), mode="values")
    with pytest.raises(ValueError, match="2-D float"):
        topk_select_tpu(jnp.zeros((8, 128), jnp.int32), jnp.ones((8,), jnp.int32), mode="kth")
    assert select_block_rows(131072, jnp.float32) == 16
    assert select_block_rows(1024, jnp.float64) is None
    # Integer and 64-bit operands fall back to the XLA reference instead of failing.
    ints = jnp.asarray(np.random.default_rng(0).integers(-5, 5, size=(64, 1024)), jnp.int32)
    np.testing.assert_array_equal(
        np.asarray(topk(ints, 10, mode="exact_mask", cfg=_PALLAS)), _reference(ints.astype(jnp.float32), 10)[0]
    )
