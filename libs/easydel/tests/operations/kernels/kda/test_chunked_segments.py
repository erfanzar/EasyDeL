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

"""Chunked per-channel KDA (training path) with packed ``segment_ids``.

Packed training used to run the token-by-token recurrence. The chunked kernel now resets
the state at segment boundaries (anywhere inside a chunk) and treats negative ids as
padding. Outputs, final state and gradients are pinned against the sequential recurrent
reference in segment mode.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.operations.kernels.kda import _chunk_kda_per_channel_fwd, _recurrent_kda_per_channel_fwd

ATOL = 5e-4


def _inputs(B, H, L, K, V, seed, strong_decay):
    ks = jax.random.split(jax.random.PRNGKey(seed), 5)
    q = jax.random.normal(ks[0], (B, H, L, K), dtype=jnp.float32)
    k = jax.random.normal(ks[1], (B, H, L, K), dtype=jnp.float32)
    v = jax.random.normal(ks[2], (B, H, L, V), dtype=jnp.float32)
    b = jax.nn.sigmoid(jax.random.normal(ks[3], (B, H, L), dtype=jnp.float32))
    n = jax.random.normal(ks[4], (B, H, L, K), dtype=jnp.float32)
    g = -5.0 * jax.nn.sigmoid(n) if strong_decay else -jax.nn.softplus(n)
    return q, k, v, b, g


def _segments(lengths_per_row, L):
    rows = []
    for lengths in lengths_per_row:
        ids = []
        for i, n in enumerate(lengths):
            ids += [i] * n
        ids += [-1] * (L - len(ids))  # trailing padding
        rows.append(ids)
    return jnp.asarray(rows, jnp.int32)


CASES = [
    # boundaries inside chunks, a 1-token segment, a segment spanning several chunks
    ([[5, 11, 1, 20, 3], [40]], False),
    ([[16, 16, 8], [7, 9, 24]], True),
    ([[3, 30, 4], [1, 1, 1, 1, 36]], True),
]


@pytest.mark.parametrize(("lengths", "strong_decay"), CASES)
def test_segments_match_recurrent_reference(lengths, strong_decay):
    B, H, L, K, V = 2, 2, 48, 16, 16
    seg = _segments(lengths, L)
    q, k, v, b, g = _inputs(B, H, L, K, V, seed=len(lengths[0]), strong_decay=strong_decay)
    out, state = jax.jit(lambda *a: _chunk_kda_per_channel_fwd(*a, chunk_size=16, segment_ids=seg))(q, k, v, b, g)
    ref_out, ref_state = _recurrent_kda_per_channel_fwd(q, k, v, b, g, segment_ids=seg)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref_out), atol=ATOL)
    np.testing.assert_allclose(np.asarray(state), np.asarray(ref_state), atol=ATOL)


@pytest.mark.parametrize(
    ("ids", "L"),
    [
        # leading padding, mid-sequence padding, and padding sharing a chunk with a later segment
        ([[-1] * 5 + [0] * 20 + [1] * 23, [0] * 10 + [-1] * 9 + [1] * 29], 48),
        # L not a multiple of chunk_size: exercises the chunk-tail edge run padding in the forward
        ([[0] * 13 + [1] * 20 + [-1] * 4, [0] * 37], 37),
    ],
)
def test_padding_positions_and_ragged_tail_match_recurrent_reference(ids, L):
    B, H, K, V = 2, 2, 16, 16
    seg = jnp.asarray(ids, jnp.int32)
    assert seg.shape == (B, L)
    q, k, v, b, g = _inputs(B, H, L, K, V, seed=L, strong_decay=True)
    out, state = jax.jit(lambda *a: _chunk_kda_per_channel_fwd(*a, chunk_size=16, segment_ids=seg))(q, k, v, b, g)
    ref_out, ref_state = _recurrent_kda_per_channel_fwd(q, k, v, b, g, segment_ids=seg)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref_out), atol=ATOL)
    np.testing.assert_allclose(np.asarray(state), np.asarray(ref_state), atol=ATOL)


def test_initial_state_is_ignored_under_segment_ids():
    B, H, L, K, V = 2, 2, 48, 16, 16
    seg = _segments([[20, 28], [48]], L)
    q, k, v, b, g = _inputs(B, H, L, K, V, seed=3, strong_decay=True)
    s0 = jax.random.normal(jax.random.PRNGKey(4), (B, H, K, V))
    with_s0 = _chunk_kda_per_channel_fwd(q, k, v, b, g, chunk_size=16, segment_ids=seg, initial_state=s0)
    without = _chunk_kda_per_channel_fwd(q, k, v, b, g, chunk_size=16, segment_ids=seg)
    ref = _recurrent_kda_per_channel_fwd(q, k, v, b, g, segment_ids=seg, initial_state=s0)
    for got, want, ref_ in zip(with_s0, without, ref, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=ATOL)
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref_), atol=ATOL)


def test_segment_gradients_match_recurrent_reference():
    B, H, L, K, V = 2, 2, 40, 16, 16
    seg = _segments([[9, 14, 17], [25, 6]], L)
    q, k, v, b, g = _inputs(B, H, L, K, V, seed=5, strong_decay=True)
    ct = jax.random.normal(jax.random.PRNGKey(9), (B, H, L, V))
    cts = jax.random.normal(jax.random.PRNGKey(10), (B, H, K, V))

    def loss(fn):
        def f(q, k, v, b, g):
            o, s = fn(q, k, v, b, g)
            return jnp.sum(o * ct) + jnp.sum(s * cts)

        return jax.jit(jax.grad(f, argnums=(0, 1, 2, 3, 4)))

    chunked = loss(lambda *a: _chunk_kda_per_channel_fwd(*a, chunk_size=16, segment_ids=seg))(q, k, v, b, g)
    ref = loss(lambda *a: _recurrent_kda_per_channel_fwd(*a, segment_ids=seg))(q, k, v, b, g)
    for got, want in zip(chunked, ref, strict=True):
        want = np.asarray(want)
        rel = np.abs(np.asarray(got) - want).max() / max(np.abs(want).max(), 1e-30)
        assert rel < 1e-3, rel


def test_no_segments_gradients_match_recurrent_reference():
    """Unpacked training path (batched triangular solve) keeps reference gradients."""
    B, H, L, K, V = 2, 2, 50, 16, 16
    q, k, v, b, g = _inputs(B, H, L, K, V, seed=11, strong_decay=True)
    s0 = jax.random.normal(jax.random.PRNGKey(12), (B, H, K, V)) * 0.1
    ct = jax.random.normal(jax.random.PRNGKey(13), (B, H, L, V))

    def grads(fn):
        return jax.jit(jax.grad(lambda *a: jnp.sum(fn(*a)[0] * ct) + jnp.sum(fn(*a)[1]), argnums=(0, 1, 2, 3, 4, 5)))

    got = grads(lambda q, k, v, b, g, s: _chunk_kda_per_channel_fwd(q, k, v, b, g, chunk_size=16, initial_state=s))(
        q, k, v, b, g, s0
    )
    want = grads(lambda q, k, v, b, g, s: _recurrent_kda_per_channel_fwd(q, k, v, b, g, initial_state=s))(
        q, k, v, b, g, s0
    )
    for a, w in zip(got, want, strict=True):
        w = np.asarray(w)
        rel = np.abs(np.asarray(a) - w).max() / max(np.abs(w).max(), 1e-30)
        assert rel < 1e-3, rel
