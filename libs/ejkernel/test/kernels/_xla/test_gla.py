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

"""Tests for XLA Gated Linear Attention (GLA) kernel."""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._xla.gla import recurrent_gla


def _make(batch=1, seq_len=16, heads=2, qk_dim=16, v_dim=16, dtype=jnp.float32, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    q = jax.random.normal(keys[0], (batch, seq_len, heads, qk_dim), dtype=dtype)
    k = jax.random.normal(keys[1], (batch, seq_len, heads, qk_dim), dtype=dtype)
    v = jax.random.normal(keys[2], (batch, seq_len, heads, v_dim), dtype=dtype)
    g = jax.nn.sigmoid(jax.random.normal(keys[3], (batch, seq_len, heads, qk_dim), dtype=dtype))
    return q, k, v, g


class TestGLAShapesAndFiniteness:
    def test_basic_shapes(self):
        q, k, v, g = _make()
        out, state = recurrent_gla(q, k, v, g=g)
        assert out.shape == (1, 16, 2, 16)
        assert state.shape == (1, 2, 16, 16)

    def test_finite_output(self):
        q, k, v, g = _make(seed=1)
        out, state = recurrent_gla(q, k, v, g=g)
        assert jnp.all(jnp.isfinite(out))
        assert jnp.all(jnp.isfinite(state))

    def test_no_gate(self):
        q, k, v, _ = _make(seed=2)
        out, _state = recurrent_gla(q, k, v, g=None)
        assert out.shape == v.shape
        assert jnp.all(jnp.isfinite(out))

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    def test_dtype(self, dtype):
        q, k, v, g = _make(dtype=dtype, seed=3)
        out, _state = recurrent_gla(q, k, v, g=g)
        assert jnp.all(jnp.isfinite(out))

    def test_with_initial_state(self):
        q, k, v, g = _make(heads=2, qk_dim=16, v_dim=16, seed=4)
        init = jnp.zeros((1, 2, 16, 16), dtype=jnp.float32)
        out, state = recurrent_gla(q, k, v, g=g, initial_state=init)
        assert jnp.all(jnp.isfinite(out))
        assert state.shape == init.shape

    def test_batch_size(self):
        q, k, v, g = _make(batch=4, seed=5)
        out, state = recurrent_gla(q, k, v, g=g)
        assert out.shape[0] == 4
        assert state.shape[0] == 4

    def test_single_token(self):
        q, k, v, g = _make(seq_len=1, seed=6)
        init = jnp.zeros((1, 2, 16, 16), dtype=jnp.float32)
        out, _state = recurrent_gla(q, k, v, g=g, initial_state=init)
        assert out.shape == (1, 1, 2, 16)
        assert jnp.all(jnp.isfinite(out))

    def test_reverse(self):
        q, k, v, g = _make(seed=7)
        out_fwd, _ = recurrent_gla(q, k, v, g=g, reverse=False)
        out_rev, _ = recurrent_gla(q, k, v, g=g, reverse=True)
        assert out_fwd.shape == out_rev.shape
        assert not jnp.allclose(out_fwd, out_rev, atol=1e-3)

    def test_continuation(self):
        q, k, v, g = _make(seq_len=32, seed=8)
        out_full, _ = recurrent_gla(q, k, v, g=g)
        _, state16 = recurrent_gla(q[:, :16], k[:, :16], v[:, :16], g=g[:, :16])
        out_cont, _ = recurrent_gla(q[:, 16:], k[:, 16:], v[:, 16:], g=g[:, 16:], initial_state=state16)
        assert jnp.allclose(out_full[:, 16:], out_cont, atol=1e-4)
