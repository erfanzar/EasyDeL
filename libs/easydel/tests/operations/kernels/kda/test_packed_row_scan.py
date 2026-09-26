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

"""Parity tests for the serving packed-row KDA scan.

``_chunked_scan_per_channel_rows`` is the eSurge serving path for GLM-5.3's
KDA layers: it processes a packed multi-row batch chunk-by-chunk under
``lax.scan`` with per-row validity masks. These tests pin it against the
sequential recurrent reference (the ground truth) for ragged rows — including
the zero-pad tail branch the training-path tests never hit.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from easydel.operations.kernels.kda import (
    _chunked_scan_per_channel_rows,
    _recurrent_kda_per_channel_fwd,
)

ATOL = 5e-4


def _recurrent_row(q, k, v, beta, decay, state):
    """Sequential per-channel reference for one row (f64-free, matches kernel math)."""
    for t in range(q.shape[0]):
        g_t = decay[t]
        state = state * jnp.exp(g_t)[:, :, None]
        kv_mem = jnp.sum(state * k[t][:, :, None], axis=-2)
        delta = (v[t] - kv_mem) * beta[t][:, None]
        state = state + k[t][:, :, None] * delta[:, :, None]
    return state


@pytest.mark.parametrize("seq", [5, 16, 17, 64, 100])
def test_scan_matches_recurrent_full_rows(seq):
    """Every row is a full-length independent sequence (packed prefill)."""
    R, H, K, V = 3, 2, 16, 16
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (R, H, seq, K), dtype=jnp.float32)
    k = jax.random.normal(key, (R, H, seq, K), dtype=jnp.float32)
    v = jax.random.normal(key, (R, H, seq, V), dtype=jnp.float32)
    b = jax.nn.sigmoid(jax.random.normal(key, (R, H, seq), dtype=jnp.float32))
    g = -jax.nn.softplus(jax.random.normal(key, (R, H, seq, K), dtype=jnp.float32))
    init = jax.random.normal(key, (R, H, K, V), dtype=jnp.float32) * 0.1

    scan_out, scan_state = _chunked_scan_per_channel_rows(
        query=q, key=k, value=v, beta=b, decay=g, initial_state=init,
        token_valid=jnp.ones((R, seq)), chunk_size=16,
    )
    ref_out, ref_state = _recurrent_kda_per_channel_fwd(
        query=q, key=k, value=v, beta=b, decay=g, initial_state=init, use_qk_l2norm=True
    )
    np.testing.assert_allclose(np.asarray(scan_out), np.asarray(ref_out), atol=ATOL)
    np.testing.assert_allclose(np.asarray(scan_state), np.asarray(ref_state), atol=ATOL)


def test_scan_ragged_rows_match_per_row_recurrent():
    """Ragged rows: each row only owns its first ``lens[r]`` tokens; padding
    must leave the carried state untouched and zero the padded outputs."""
    R, H, T, K, V = 4, 2, 48, 16, 16
    key = jax.random.PRNGKey(1)
    q = jax.random.normal(key, (R, H, T, K), dtype=jnp.float32)
    k = jax.random.normal(key, (R, H, T, K), dtype=jnp.float32)
    v = jax.random.normal(key, (R, H, T, V), dtype=jnp.float32)
    b = jax.nn.sigmoid(jax.random.normal(key, (R, H, T), dtype=jnp.float32))
    g = -jax.nn.softplus(jax.random.normal(key, (R, H, T, K), dtype=jnp.float32))
    init = jax.random.normal(key, (R, H, K, V), dtype=jnp.float32) * 0.1

    lens = jnp.array([48, 30, 17, 1])
    valid = (jnp.arange(T)[None, :] < lens[:, None]).astype(jnp.float32)

    def mask4(x):
        return x * valid[:, None, :, None]

    scan_out, scan_state = _chunked_scan_per_channel_rows(
        query=mask4(q), key=mask4(k), value=mask4(v),
        beta=b * valid[:, None], decay=mask4(g),
        initial_state=init, token_valid=valid, chunk_size=16,
    )

    for r in range(R):
        length = int(lens[r])
        ref_out, ref_state = _recurrent_kda_per_channel_fwd(
            query=q[r : r + 1, :, :length], key=k[r : r + 1, :, :length], value=v[r : r + 1, :, :length],
            beta=b[r : r + 1, :, :length], decay=g[r : r + 1, :, :length],
            initial_state=init[r : r + 1], use_qk_l2norm=True,
        )
        np.testing.assert_allclose(
            np.asarray(scan_out[r, :, :length]), np.asarray(ref_out[0]), atol=ATOL
        )
        np.testing.assert_allclose(np.asarray(scan_state[r]), np.asarray(ref_state[0]), atol=ATOL)
        # padded tail outputs must be exactly zero
        if length < T:
            assert float(jnp.abs(scan_out[r, :, length:]).max()) == 0.0
