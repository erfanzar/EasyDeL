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

"""Parity and contract tests for the fused exact top-k.

The reference is ``jax.lax.top_k`` itself, which is independent of this
operation's implementation, so a pass here is not the kernel agreeing with
itself. Shapes mirror the three real call sites: a MoE router (narrow axis,
tiny k), the DSA indexer (large k), and sampling top-k filtering (vocab-scale
axis, per-row dynamic k).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.modules import topk
from ejkernel.modules.operations.configs import TopKConfig


def _rand(shape, seed=0, dtype=jnp.float32):
    return jnp.asarray(np.random.default_rng(seed).normal(size=shape), dtype)


@pytest.mark.parametrize(
    ("shape", "k"),
    [
        ((8, 256), 6),  # MoE router
        ((2, 64, 512), 32),  # DSA indexer-ish
        ((4, 8192), 16),  # wide axis, small k -> pallas regime on TPU
        ((3, 129), 1),  # non-round width, k=1
        ((1, 17), 17),  # k == width
    ],
)
def test_values_mode_matches_lax_top_k(shape, k):
    """``mode='values'`` must equal ``jax.lax.top_k`` in values AND indices."""
    x = _rand(shape, seed=hash((shape, k)) % 2**31)
    values, indices = topk(x, k=k)
    ref_values, ref_indices = jax.lax.top_k(x, k)

    assert values.shape == ref_values.shape
    assert np.array_equal(np.asarray(values), np.asarray(ref_values))
    assert np.array_equal(np.asarray(indices), np.asarray(ref_indices))


def test_ties_break_on_lower_index_like_lax():
    """With duplicates, both must pick the same (lowest) indices."""
    x = jnp.asarray([[1.0, 3.0, 3.0, 3.0, 2.0, 3.0]], jnp.float32)
    values, indices = topk(x, k=3)
    ref_values, ref_indices = jax.lax.top_k(x, 3)
    assert np.array_equal(np.asarray(values), np.asarray(ref_values))
    assert np.array_equal(np.asarray(indices), np.asarray(ref_indices))


def test_axis_other_than_last():
    """A non-trailing reduction axis must give the same answer as moving it."""
    x = _rand((5, 40, 3), seed=7)
    values, indices = topk(x, k=4, axis=1)
    ref_values, ref_indices = jax.lax.top_k(jnp.moveaxis(x, 1, -1), 4)
    assert np.array_equal(np.asarray(values), np.asarray(jnp.moveaxis(ref_values, -1, 1)))
    assert np.array_equal(np.asarray(indices), np.asarray(jnp.moveaxis(ref_indices, -1, 1)))


@pytest.mark.parametrize("width", [256, 4096])
def test_mask_mode_keeps_exactly_k_per_row(width):
    """Per-row dynamic k -- the sampler's contract, which sorting cannot serve."""
    rows = 6
    x = _rand((rows, width), seed=3)
    ks = jnp.asarray([0, 1, 2, 5, 13, width], jnp.int32)[:rows]

    keep = topk(x, ks, mode="mask")
    assert keep.shape == x.shape
    assert np.array_equal(np.asarray(keep.sum(-1)), np.asarray(ks))

    # every kept element must be >= every dropped element in its row
    xs = np.asarray(x)
    km = np.asarray(keep)
    for r in range(rows):
        if km[r].any() and (~km[r]).any():
            assert xs[r][km[r]].min() >= xs[r][~km[r]].max()


def test_mask_mode_keeps_full_tie_groups():
    """Ties at the threshold are kept together, so a row may exceed k.

    Deliberate: an exact-k cut would have to invent an order over equal logits.
    """
    x = jnp.asarray([[5.0, 1.0, 5.0, 5.0, 0.0]], jnp.float32)
    keep = topk(x, jnp.asarray([2], jnp.int32), mode="mask")
    assert int(np.asarray(keep).sum()) == 3
    assert np.array_equal(np.asarray(keep)[0], np.array([True, False, True, True, False]))


def test_filter_mode_replaces_dropped_entries():
    """``filter`` returns the input with dropped entries set to ``mask_fill``."""
    x = jnp.asarray([[3.0, 1.0, 2.0, 0.0]], jnp.float32)
    out = topk(x, jnp.asarray([2], jnp.int32), mode="filter", mask_fill=-1e9)
    assert np.array_equal(np.asarray(out)[0], np.array([3.0, -1e9, 2.0, -1e9], np.float32))


def test_values_mode_rejects_traced_k():
    """A sorted top-k needs a static output width; say so instead of guessing."""
    x = _rand((2, 32))
    with pytest.raises(ValueError, match="static int k"):
        topk(x, jnp.asarray([3, 4], jnp.int32), mode="values")


def test_unknown_mode_raises():
    x = _rand((2, 32))
    with pytest.raises(ValueError, match="unknown topk mode"):
        topk(x, 3, mode="sorted")


def test_registry_has_both_platforms():
    """The XLA reference is mandatory; the TPU Pallas path is the accelerator."""
    assert kernel_registry.get("topk", platform=Platform.XLA, backend=Backend.ANY) is not None
    assert kernel_registry.get("topk", platform=Platform.PALLAS, backend=Backend.TPU) is not None


def test_registry_signatures_are_compatible():
    kernel_registry.validate_signatures("topk")


def test_gradient_matches_lax_top_k():
    """Gradient must flow only to the selected elements, exactly like lax.

    MoE router scores are trained through their top-k values, so a kernel
    without a transpose rule would break training rather than just be slow.
    """

    def ours(a):
        v, _ = topk(a, k=4)
        return (v * jnp.arange(1.0, 5.0)).sum()

    def ref(a):
        v, _ = jax.lax.top_k(a, 4)
        return (v * jnp.arange(1.0, 5.0)).sum()

    x = _rand((3, 64), seed=11)
    g_ours = jax.grad(ours)(x)
    g_ref = jax.grad(ref)(x)
    assert np.array_equal(np.asarray(g_ours), np.asarray(g_ref))
    # gradient is sparse: exactly k non-zeros per row
    assert np.array_equal((np.asarray(g_ours) != 0).sum(-1), np.full(3, 4))


_PALLAS = TopKConfig(platform="pallas", backend="tpu")
_tpu_only = pytest.mark.skipif(jax.default_backend() != "tpu", reason="pallas top-k path is TPU-only")


def _special_rows(shape, seed):
    """Heavy ties, signed zeros, infinities and both NaN signs, at random spots."""
    rng = np.random.default_rng(seed)
    x = rng.integers(-3, 4, size=shape).astype(np.float32)
    for value, frac in ((np.inf, 0.05), (-np.inf, 0.2), (np.nan, 0.03), (0.0, 0.1), (-0.0, 0.1)):
        x[rng.random(shape) < frac] = value
    bits = x.view(np.int32)
    negative_nan = rng.random(shape) < 0.02
    bits[negative_nan] = np.int32(-4194304)  # 0xFFC00000: a NaN with the sign bit set
    return jnp.asarray(x)


@_tpu_only
@pytest.mark.parametrize(
    ("name", "shape", "k"),
    [
        ("normal", (256, 16384), 2048),
        ("normal-small-k", (256, 4096), 32),
        ("specials", (64, 1024), 300),
        ("specials-wide", (64, 4096), 1500),  # k > 1024: bounded output merge
        ("ties-wide", (64, 4096), 1500),
        ("all-equal", (16, 512), 77),
        ("unaligned", (13, 1000), 999),
        ("k=1", (8, 128), 1),
        ("k=width", (8, 256), 256),
        ("leading-dims", (2, 3, 7, 640), 200),
    ],
)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_pallas_threshold_path_is_bit_identical_to_lax(name, shape, k, dtype):
    """Forced Pallas path: same values bits and same indices as ``lax.top_k``."""
    if name.startswith("ties"):
        x = jnp.asarray(np.random.default_rng(k).integers(0, 6, size=shape).astype(np.float32)).astype(dtype)
    elif name.startswith("specials") or name == "all-equal":
        x = _special_rows(shape, seed=k) if name.startswith("specials") else jnp.zeros(shape, jnp.float32)
        x = x.astype(dtype)
    else:
        x = _rand(shape, seed=k, dtype=dtype)
    values, indices = topk(x, k=k, cfg=_PALLAS)
    ref_values, ref_indices = jax.jit(lambda a: jax.lax.top_k(a, k))(x)
    assert values.dtype == ref_values.dtype
    assert np.array_equal(np.asarray(indices), np.asarray(ref_indices))
    got = np.asarray(values.astype(jnp.float32))
    want = np.asarray(ref_values.astype(jnp.float32))
    assert np.array_equal(np.isnan(got), np.isnan(want))
    assert np.array_equal(
        np.where(np.isnan(got), 0, got).view(np.int32), np.where(np.isnan(want), 0, want).view(np.int32)
    )


@_tpu_only
def test_pallas_gradient_matches_lax_top_k():
    """Value cotangents land on exactly the positions ``lax.top_k`` sends them to."""
    weights = jnp.linspace(-1.0, 2.0, 64)

    def ours(a):
        v, _ = topk(a, k=64, cfg=_PALLAS)
        return (v * weights).sum()

    def ref(a):
        v, _ = jax.lax.top_k(a, 64)
        return (v * weights).sum()

    x = _rand((256, 2048), seed=13)
    assert np.array_equal(np.asarray(jax.jit(jax.grad(ours))(x)), np.asarray(jax.jit(jax.grad(ref))(x)))


@_tpu_only
@pytest.mark.parametrize(
    ("shape", "k", "platform"),
    [
        ((8192, 16384), 2048, "pallas"),  # DSA indexer
        ((1024, 2048), 256, "pallas"),
        ((16384, 128), 8, "xla"),  # MoE router: narrow, tiny k
        ((8, 131072), 50, "xla"),  # sampling: few rows
        ((8192, 16384), 8, "xla"),  # tiny k
        ((8192, 512), 128, "xla"),  # 512 candidates: below the Pallas width floor
    ],
)
def test_heuristic_routes_by_measured_regime(shape, k, platform):
    from ejkernel.modules.operations.topk import TopK
    from ejkernel.ops import Invocation

    operand = jax.ShapeDtypeStruct(shape, jnp.float32)
    inv = Invocation(op_id="topk", args=(operand, k), kwargs={"mode": "values", "axis": -1})
    assert TopK().heuristic_cfg(inv).platform == platform


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
