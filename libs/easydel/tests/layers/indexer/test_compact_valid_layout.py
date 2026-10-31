# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""``compact_valid_layout`` (valid-first token order) and ``permute_tokens`` (gather-only token permutation).

Pool indexers group consecutive *valid* tokens, so attention kernels that read contiguous key ranges consume the keys
in each row's valid-first order. The layout must match a NumPy reference for every padding pattern, be the identity
exactly when every row is already valid-first, and its permutation must differentiate as a gather (no scatter-add).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from easydel.layers.indexer import compact_valid_layout, permute_tokens

SEQ = 37


def _rows(*patterns: str) -> np.ndarray:
    rng = np.random.default_rng(7)
    rows = []
    for name in patterns:
        row = np.ones(SEQ, bool)
        if name == "right":
            row[SEQ - 9 :] = False
        elif name == "left":
            row[:11] = False
        elif name == "holes":
            row = rng.random(SEQ) < 0.6
            row[0] = False
        elif name == "empty":
            row[:] = False
        rows.append(row)
    return np.stack(rows)


def _reference(valid: np.ndarray):
    order = np.stack([np.concatenate([np.flatnonzero(v), np.flatnonzero(~v)]) for v in valid])
    inverse = np.argsort(order, axis=1)
    rank = np.where(valid, np.cumsum(valid, axis=1) - 1, -1)
    return order, inverse, rank


def test_layout_matches_numpy_reference_for_every_padding_pattern():
    valid = _rows("all", "right", "left", "holes", "empty")
    layout = jax.jit(compact_valid_layout)(jnp.asarray(valid))
    order, inverse, rank = _reference(valid)
    np.testing.assert_array_equal(np.asarray(layout.order), order)
    np.testing.assert_array_equal(np.asarray(layout.inverse), inverse)
    np.testing.assert_array_equal(np.asarray(layout.rank), rank)
    assert not bool(layout.identity)


def test_layout_is_the_identity_exactly_for_valid_first_rows():
    layout = jax.jit(compact_valid_layout)(jnp.asarray(_rows("all", "right", "empty")))
    assert bool(layout.identity)
    np.testing.assert_array_equal(np.asarray(layout.order), np.broadcast_to(np.arange(SEQ), (3, SEQ)))
    assert not bool(jax.jit(compact_valid_layout)(jnp.asarray(_rows("all", "left"))).identity)


def test_gather_reads_tokens_in_valid_first_order():
    rng = np.random.default_rng(3)
    for patterns in (("all", "right"), ("left", "holes", "right")):
        valid = _rows(*patterns)
        x = jnp.asarray(rng.standard_normal((len(patterns), SEQ, 3, 4)), jnp.float32)
        got = jax.jit(lambda v, a: compact_valid_layout(v).gather(a))(jnp.asarray(valid), x)
        order, _, _ = _reference(valid)
        want = np.take_along_axis(np.asarray(x), order[:, :, None, None], axis=1)
        np.testing.assert_array_equal(np.asarray(got), want)


def test_permutation_derivatives_match_the_plain_gather():
    valid = _rows("left", "holes")
    layout = compact_valid_layout(jnp.asarray(valid))
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.standard_normal((2, SEQ, 6)), jnp.float32)
    w = jnp.asarray(rng.standard_normal((2, SEQ, 6)), jnp.float32)
    t = jnp.asarray(rng.standard_normal((2, SEQ, 6)), jnp.float32)

    def plain(a):
        return jnp.take_along_axis(a, layout.order[:, :, None], axis=1)

    def ours(a):
        return permute_tokens(a, layout.order, layout.inverse)

    def loss(fn, a):
        return jnp.sum(jnp.sin(fn(a)) * w)

    np.testing.assert_allclose(
        np.asarray(jax.grad(lambda a: loss(ours, a))(x)), np.asarray(jax.grad(lambda a: loss(plain, a))(x)), rtol=1e-6
    )
    _, tangent = jax.jvp(ours, (x,), (t,))
    np.testing.assert_array_equal(np.asarray(tangent), np.asarray(plain(t)))


def test_permutation_reverse_mode_gathers_instead_of_scattering():
    layout = compact_valid_layout(jnp.asarray(_rows("left", "holes")))
    x = jnp.ones((2, SEQ, 8), jnp.float32)

    def grad_text(fn):
        return jax.jit(jax.grad(lambda a: jnp.sum(jnp.sin(fn(a))))).lower(x).as_text()

    assert "scatter" in grad_text(lambda a: jnp.take_along_axis(a, layout.order[:, :, None], axis=1))
    assert "scatter" not in grad_text(lambda a: permute_tokens(a, layout.order, layout.inverse))
