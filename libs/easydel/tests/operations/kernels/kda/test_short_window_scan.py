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

"""Short-window per-channel KDA (speculative verify windows) in eSurge serving.

``_short_window_kda_per_channel`` advances rows of at most a few tokens with the per-token decode recurrence
instead of the chunked prefill kernel. Pinned against the sequential per-channel recurrent reference (ground
truth), against step-by-step decode with the cache's storage dtype between steps (the plain-decode contract), and
for empty rows / padding.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.operations.kernels.kda import (
    _recurrent_kda_per_channel_fwd,
    _short_window_kda_per_channel,
    _single_step_kda_per_channel_fwd_bthd,
)

ATOL = 5e-4


def _inputs(R, H, T, K, V, seed):
    ks = jax.random.split(jax.random.PRNGKey(seed), 6)
    q = jax.random.normal(ks[0], (R, H, T, K), dtype=jnp.float32)
    k = jax.random.normal(ks[1], (R, H, T, K), dtype=jnp.float32)
    v = jax.random.normal(ks[2], (R, H, T, V), dtype=jnp.float32)
    b = jax.nn.sigmoid(jax.random.normal(ks[3], (R, H, T), dtype=jnp.float32))
    g = -5.0 * jax.nn.sigmoid(jax.random.normal(ks[4], (R, H, T, K), dtype=jnp.float32))  # GLM gate range
    init = jax.random.normal(ks[5], (R, H, K, V), dtype=jnp.float32) * 0.1
    return q, k, v, b, g, init


@pytest.mark.parametrize("lens", [[2, 2, 2, 2], [3, 1, 0, 2], [8, 5, 0, 7], [0, 0, 4, 0]])
def test_matches_per_row_recurrent_reference(lens):
    R, H, T, K, V = 4, 2, 8, 16, 16
    q, k, v, b, g, init = _inputs(R, H, T, K, V, seed=sum(lens) + 11)
    lens_a = jnp.asarray(lens, jnp.int32)
    out, state = jax.jit(lambda *a: _short_window_kda_per_channel(*a, max_steps=8))(q, k, v, b, g, init, lens_a)
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


@pytest.mark.parametrize("state_dtype", [jnp.float32, jnp.bfloat16])
def test_matches_stepwise_decode_with_storage_cast(state_dtype):
    """Each verify token must see plain decode's arithmetic: single step, state stored in the cache dtype."""
    R, H, T, K, V = 3, 2, 4, 16, 16
    lens = [4, 2, 3]
    q, k, v, b, g, init = _inputs(R, H, T, K, V, seed=5)
    init = init.astype(state_dtype)
    out, state = jax.jit(lambda *a: _short_window_kda_per_channel(*a, max_steps=8))(
        q, k, v, b, g, init, jnp.asarray(lens, jnp.int32)
    )
    step = jax.jit(
        lambda qq, kk, vv, bb, gg, st: _single_step_kda_per_channel_fwd_bthd(
            query=qq, key=kk, value=vv, beta=bb, decay=gg, recurrent_state=st, use_qk_l2norm=True
        )
    )
    for r, length in enumerate(lens):
        st = init[r : r + 1]
        for t in range(length):
            o, s = step(
                q[r : r + 1, :, t : t + 1].transpose(0, 2, 1, 3),
                k[r : r + 1, :, t : t + 1].transpose(0, 2, 1, 3),
                v[r : r + 1, :, t : t + 1].transpose(0, 2, 1, 3),
                b[r : r + 1, :, t : t + 1].transpose(0, 2, 1),
                g[r : r + 1, :, t : t + 1].transpose(0, 2, 1, 3),
                st,
            )
            st = s.astype(state_dtype)
            np.testing.assert_allclose(
                np.asarray(out[r, :, t]), np.asarray(o[0, 0].astype(jnp.float32)), rtol=1e-5, atol=1e-5
            )
        np.testing.assert_allclose(np.asarray(state[r]), np.asarray(st[0].astype(jnp.float32)), rtol=1e-5, atol=1e-5)


def test_max_steps_caps_processing_and_padding_is_ignored():
    """Positions at or past ``max_steps`` / a row's length never touch state or output (garbage there is inert)."""
    R, H, T, K, V = 2, 2, 6, 16, 16
    q, k, v, b, g, init = _inputs(R, H, T, K, V, seed=9)
    pad = jnp.arange(T)[None, None, :, None] >= 3
    q_bad = jnp.where(pad, jnp.nan, q)
    v_bad = jnp.where(pad, jnp.inf, v)
    lens = jnp.asarray([3, 2], jnp.int32)
    out, state = jax.jit(lambda *a: _short_window_kda_per_channel(*a, max_steps=3))(q_bad, k, v_bad, b, g, init, lens)
    assert np.isfinite(np.asarray(out)).all() and np.isfinite(np.asarray(state)).all()
    ref, ref_state = jax.jit(lambda *a: _short_window_kda_per_channel(*a, max_steps=3))(q, k, v, b, g, init, lens)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))
    np.testing.assert_array_equal(np.asarray(state), np.asarray(ref_state))
