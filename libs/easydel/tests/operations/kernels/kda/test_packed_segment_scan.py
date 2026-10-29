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

"""Parity tests for the segment-aligned serving KDA prefill.

``_packed_segment_kda_per_channel`` replaces the per-row bucket-padded chunk scan on
GLM-5.3's eSurge prefill path. It is pinned against the sequential per-channel recurrent
reference per row (ground truth) and against the row scan it replaces, including empty
rows, lengths that are not chunk multiples, the tight ``max_tokens`` budget used in
serving, and GLM-strength decay.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.operations.kernels.kda import (
    _chunked_scan_per_channel_rows,
    _packed_segment_kda_per_channel,
    _recurrent_kda_per_channel_fwd,
)

ATOL = 5e-4


def _inputs(R, H, T, K, V, lens, seed, strong_decay=False):
    ks = jax.random.split(jax.random.PRNGKey(seed), 6)
    q = jax.random.normal(ks[0], (R, H, T, K), dtype=jnp.float32)
    k = jax.random.normal(ks[1], (R, H, T, K), dtype=jnp.float32)
    v = jax.random.normal(ks[2], (R, H, T, V), dtype=jnp.float32)
    b = jax.nn.sigmoid(jax.random.normal(ks[3], (R, H, T), dtype=jnp.float32))
    n = jax.random.normal(ks[4], (R, H, T, K), dtype=jnp.float32)
    g = -5.0 * jax.nn.sigmoid(n) if strong_decay else -jax.nn.softplus(n)
    init = jax.random.normal(ks[5], (R, H, K, V), dtype=jnp.float32) * 0.1
    valid = (jnp.arange(T)[None, :] < jnp.asarray(lens)[:, None]).astype(jnp.float32)
    return q, k, v, b, g, init, valid


def _run(fn, q, k, v, b, g, init, valid, **kw):
    return fn(query=q, key=k, value=v, beta=b, decay=g, initial_state=init, token_valid=valid, **kw)


@pytest.mark.parametrize(
    ("lens", "strong_decay"),
    [
        ([48, 30, 17, 1], False),  # ragged, not chunk multiples, one decode-like row
        ([0, 40, 0, 8], False),  # empty rows must keep their initial state
        ([48, 48, 48, 48], False),  # every row full
        ([96, 0, 0, 0], True),  # one long row, GLM-strength decay (overflow guard)
        ([33, 16, 15, 32], True),
    ],
)
def test_matches_per_row_recurrent_reference(lens, strong_decay):
    R, H, K, V = 4, 2, 16, 16
    T = max(max(lens), 1)
    q, k, v, b, g, init, valid = _inputs(R, H, T, K, V, lens, seed=sum(lens), strong_decay=strong_decay)
    out, state = jax.jit(lambda *a: _run(_packed_segment_kda_per_channel, *a, chunk_size=16, max_tokens=sum(lens)))(
        q, k, v, b, g, init, valid
    )
    out, state = np.asarray(out), np.asarray(state)

    for r, length in enumerate(lens):
        if length == 0:
            np.testing.assert_array_equal(state[r], np.asarray(init[r]))
            assert float(np.abs(out[r]).max()) == 0.0
            continue
        ref_out, ref_state = _recurrent_kda_per_channel_fwd(
            query=q[r : r + 1, :, :length],
            key=k[r : r + 1, :, :length],
            value=v[r : r + 1, :, :length],
            beta=b[r : r + 1, :, :length],
            decay=g[r : r + 1, :, :length],
            initial_state=init[r : r + 1],
            use_qk_l2norm=True,
        )
        np.testing.assert_allclose(out[r, :, :length], np.asarray(ref_out[0]), atol=ATOL)
        np.testing.assert_allclose(state[r], np.asarray(ref_state[0]), atol=ATOL)
        if length < T:
            assert float(np.abs(out[r, :, length:]).max()) == 0.0


def test_matches_row_scan_with_serving_budget():
    """Serving shape in miniature: rows are segments of one T-token stream (max_tokens=T)."""
    R, H, T, K, V = 8, 2, 128, 32, 32
    lens = [40, 1, 0, 23, 64, 0, 0, 0]  # sums to exactly T
    assert sum(lens) == T
    args = _inputs(R, H, T, K, V, lens, seed=7, strong_decay=True)
    q, k, v, b, g, init, valid = args
    m4 = valid[:, None, :, None]
    ref_out, ref_state = _run(
        _chunked_scan_per_channel_rows, q * m4, k * m4, v * m4, b * valid[:, None], g * m4, init, valid, chunk_size=16
    )
    out, state = jax.jit(lambda *a: _run(_packed_segment_kda_per_channel, *a, chunk_size=16, max_tokens=T))(*args)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref_out), atol=ATOL)
    np.testing.assert_allclose(np.asarray(state), np.asarray(ref_state), atol=ATOL)


def test_no_initial_state_and_nonfinite_padding():
    """initial_state=None starts from zeros; garbage in padded slots must not leak."""
    R, H, T, K, V = 3, 2, 40, 16, 16
    lens = [40, 25, 0]
    q, k, v, b, g, _, valid = _inputs(R, H, T, K, V, lens, seed=3)
    pad = valid[:, None, :, None] == 0
    q_bad = jnp.where(pad, jnp.nan, q)
    v_bad = jnp.where(pad, jnp.inf, v)
    out, state = _run(_packed_segment_kda_per_channel, q_bad, k, v_bad, b, g, None, valid, chunk_size=16, max_tokens=65)
    assert np.isfinite(np.asarray(out)).all() and np.isfinite(np.asarray(state)).all()
    np.testing.assert_array_equal(np.asarray(state[2]), 0.0)
    for r in (0, 1):
        length = lens[r]
        ref_out, ref_state = _recurrent_kda_per_channel_fwd(
            query=q[r : r + 1, :, :length],
            key=k[r : r + 1, :, :length],
            value=v[r : r + 1, :, :length],
            beta=b[r : r + 1, :, :length],
            decay=g[r : r + 1, :, :length],
            initial_state=None,
            use_qk_l2norm=True,
        )
        np.testing.assert_allclose(np.asarray(out[r, :, :length]), np.asarray(ref_out[0]), atol=ATOL)
        np.testing.assert_allclose(np.asarray(state[r]), np.asarray(ref_state[0]), atol=ATOL)
