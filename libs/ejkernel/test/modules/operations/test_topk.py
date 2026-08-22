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


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="pallas top-k path is TPU-only")
def test_pallas_gradient_matches_lax_top_k():
    """Same, forced through the Pallas superset path on a wide axis."""

    def ours(a):
        v, _ = topk(a, k=8)
        return (v**2).sum()

    def ref(a):
        v, _ = jax.lax.top_k(a, 8)
        return (v**2).sum()

    x = _rand((4, 8192), seed=13)
    assert np.allclose(np.asarray(jax.grad(ours)(x)), np.asarray(jax.grad(ref)(x)), atol=0, rtol=0)


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="pallas top-k path is TPU-only")
@pytest.mark.parametrize("k", [1, 8, 32])
def test_pallas_values_match_lax_on_wide_axis(k):
    """The Pallas superset must be exact, not approximate, at every k it takes."""
    x = _rand((8, 16384), seed=k)
    v, i = topk(x, k=k)
    rv, ri = jax.lax.top_k(x, k)
    assert np.array_equal(np.asarray(v), np.asarray(rv))
    assert np.array_equal(np.asarray(i), np.asarray(ri))


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
