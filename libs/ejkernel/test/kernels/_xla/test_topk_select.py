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

"""XLA reference of the sort-free ``topk`` modes ``"exact_mask"`` and ``"kth"``.

The reference is ``jax.lax.top_k`` run over the full row width (its index order
is the descending total order with lower-index tie-break), from which the first
``k`` picks of every row are read in NumPy. It never dispatches to ``topk``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._xla.topk._interface import topk


def _reference(x, ks):
    """``(pick mask, k-th value)`` per row from a full-width ``lax.top_k``; ``ks`` is ``x.shape[:-1]``."""
    values, order = jax.lax.top_k(x, x.shape[-1])
    values, order = np.asarray(values), np.asarray(order)
    ks = np.broadcast_to(np.asarray(ks), x.shape[:-1])
    mask = np.zeros(x.shape, dtype=bool)
    kth = np.zeros(x.shape[:-1], dtype=values.dtype)
    for row in np.ndindex(x.shape[:-1]):
        k = int(ks[row])
        mask[row][order[row][:k]] = True
        kth[row] = values[row][max(k, 1) - 1]
    return mask, kth


def _bits(a):
    a = np.asarray(jnp.asarray(a).astype(jnp.float32))
    return np.where(np.isnan(a), 0, a).view(np.int32), np.isnan(a)


def _specials(shape, seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(-3, 4, size=shape).astype(np.float32)  # heavy ties
    for value, frac in ((np.inf, 0.05), (-np.inf, 0.3), (np.nan, 0.03), (0.0, 0.1), (-0.0, 0.1)):
        x[rng.random(shape) < frac] = value
    x.view(np.int32)[rng.random(shape) < 0.02] = np.int32(-4194304)  # negative-sign NaN
    return x


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("kind", ["normal", "specials"])
@pytest.mark.parametrize(("shape", "k"), [((3, 64), 8), ((2, 5, 300), 37), ((4, 4096), 512), ((17,), 1), ((2, 9), 9)])
def test_static_k_matches_lax_top_k(dtype, kind, shape, k):
    rng = np.random.default_rng(k)
    x = rng.standard_normal(shape).astype(np.float32) if kind == "normal" else _specials(shape, k)
    x = jnp.asarray(x).astype(dtype)
    want_mask, want_kth = _reference(x, k)
    got = np.asarray(topk(x, k, mode="exact_mask"))
    np.testing.assert_array_equal(got, want_mask)
    assert (got.sum(-1) == k).all()
    kth = topk(x, k, mode="kth")
    assert kth.dtype == x.dtype and kth.shape == x.shape[:-1]
    got_bits, got_nan = _bits(kth)
    want_bits, want_nan = _bits(want_kth)
    np.testing.assert_array_equal(got_nan, want_nan)
    np.testing.assert_array_equal(got_bits, want_bits)  # bit-identical, -0.0 included


def test_ties_take_lowest_index_and_minus_inf_fills():
    x = jnp.asarray([[5.0, 1.0, 5.0, 5.0, 0.0], [-jnp.inf, 2.0, -jnp.inf, -jnp.inf, 1.0]], jnp.float32)
    got = np.asarray(topk(x, 2, mode="exact_mask"))
    assert got[0].tolist() == [True, False, True, False, False]
    got = np.asarray(topk(x, 4, mode="exact_mask"))
    assert got[1].tolist() == [True, True, True, False, True]
    assert np.asarray(topk(x, 4, mode="kth")).tolist() == [1.0, -np.inf]


def test_signed_zero_is_below_positive_zero():
    x = jnp.asarray([[-0.0, 0.0, -0.0, 0.0]], jnp.float32)
    np.testing.assert_array_equal(np.asarray(topk(x, 3, mode="exact_mask")), [[True, True, False, True]])
    kth = np.asarray(topk(x, 3, mode="kth"))
    assert kth.view(np.int32)[0] == np.float32(-0.0).view(np.int32)


@pytest.mark.parametrize("kind", ["normal", "specials"])
def test_per_row_dynamic_k(kind):
    shape = (6, 513)
    rng = np.random.default_rng(5)
    x = rng.standard_normal(shape).astype(np.float32) if kind == "normal" else _specials(shape, 5)
    x = jnp.asarray(x)
    ks = jnp.asarray([0, 1, 2, 100, 512, 513], jnp.int32)
    want_mask, want_kth = _reference(x, np.asarray(ks))
    got = np.asarray(jax.jit(lambda a, kk: topk(a, kk, mode="exact_mask"))(x, ks))
    np.testing.assert_array_equal(got, want_mask)
    np.testing.assert_array_equal(got.sum(-1), np.asarray(ks))
    # k = 0 has no k-th value: a traced k is clamped to 1.
    got_bits, got_nan = _bits(jax.jit(lambda a, kk: topk(a, kk, mode="kth"))(x, ks))
    want_bits, want_nan = _bits(want_kth)
    np.testing.assert_array_equal(got_nan, want_nan)
    np.testing.assert_array_equal(got_bits, want_bits)


def test_non_trailing_axis_and_k_beyond_width():
    x = jnp.asarray(np.random.default_rng(7).standard_normal((5, 40, 3)), jnp.float32)
    got = np.asarray(topk(x, 4, axis=1, mode="exact_mask"))
    want, want_kth = _reference(jnp.moveaxis(x, 1, -1), 4)
    np.testing.assert_array_equal(got, np.moveaxis(want, -1, 1))
    np.testing.assert_array_equal(np.asarray(topk(x, 4, axis=1, mode="kth")), want_kth)
    assert np.asarray(topk(x, 41, mode="exact_mask")).all()
    assert not np.asarray(topk(x, 0, mode="exact_mask")).any()


@pytest.mark.parametrize("k", [0, 4])
def test_kth_rejects_static_k_outside_width(k):
    with pytest.raises(ValueError, match="mode='kth'"):
        topk(jnp.zeros((2, 3), jnp.float32), k, mode="kth")


def test_unknown_mode_lists_the_new_modes():
    with pytest.raises(ValueError, match="exact_mask"):
        topk(jnp.zeros((2, 3), jnp.float32), 1, mode="sorted")
