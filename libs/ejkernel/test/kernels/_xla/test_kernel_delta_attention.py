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

"""Tests for XLA Kernel Delta Attention (KDA)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._xla import kda, kda_decay, kernel_delta_attention
from ejkernel.kernels._xla.kernel_delta_attention._xla_impl_fwd import _recurrent_kda_fwd
from ejkernel.modules.operations import KernelDeltaAttention, KernelDeltaAttentionConfig


class TestKernelDeltaAttention:
    def test_forward_shape(self):
        batch, seq_len, num_heads, head_dim, value_dim = 2, 17, 4, 8, 8
        q = jnp.ones((batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        k = jnp.ones((batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        v = jnp.ones((batch, seq_len, num_heads, value_dim), dtype=jnp.float32)
        beta = jnp.ones((batch, seq_len, num_heads), dtype=jnp.float32) * 0.5
        decay = jnp.zeros((batch, seq_len, num_heads), dtype=jnp.float32)

        out, state = kernel_delta_attention(q, k, v, beta, decay, chunk_size=8)
        assert out.shape == (batch, seq_len, num_heads, value_dim)
        assert state.shape == (batch, num_heads, head_dim, value_dim)

    def test_alias_matches(self):
        key = jax.random.PRNGKey(0)
        batch, seq_len, num_heads, head_dim, value_dim = 1, 8, 2, 4, 4
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        q = jax.random.normal(k1, (batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        k = jax.random.normal(k2, (batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        v = jax.random.normal(k3, (batch, seq_len, num_heads, value_dim), dtype=jnp.float32)
        beta = jax.nn.sigmoid(jax.random.normal(k4, (batch, seq_len, num_heads), dtype=jnp.float32))
        decay = jax.random.normal(k5, (batch, seq_len, num_heads), dtype=jnp.float32) * -0.01

        out1, state1 = kernel_delta_attention(q, k, v, beta, decay, chunk_size=4)
        out2, state2 = kda(q, k, v, beta, decay, chunk_size=4)

        assert jnp.allclose(out1, out2, atol=1e-5, rtol=1e-5)
        assert jnp.allclose(state1, state2, atol=1e-5, rtol=1e-5)

    def test_kda_decay_sign(self):
        batch, seq_len, num_heads = 2, 5, 4
        gate = jnp.zeros((batch, seq_len, num_heads), dtype=jnp.float32)
        A_log = jnp.zeros((num_heads,), dtype=jnp.float32)
        dt_bias = jnp.zeros((num_heads,), dtype=jnp.float32)
        decay = kda_decay(gate, A_log, dt_bias)
        assert decay.shape == (batch, seq_len, num_heads)
        # A is negative and softplus is positive -> decay should be <= 0.
        assert jnp.all(decay <= 0)

    def test_chunked_matches_recurrent_small(self):
        key = jax.random.PRNGKey(42)
        batch, seq_len, num_heads, head_dim, value_dim = 2, 32, 4, 8, 8
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        q = jax.random.normal(k1, (batch, seq_len, num_heads, head_dim), dtype=jnp.float32) * 0.1
        k = jax.random.normal(k2, (batch, seq_len, num_heads, head_dim), dtype=jnp.float32) * 0.1
        v = jax.random.normal(k3, (batch, seq_len, num_heads, value_dim), dtype=jnp.float32) * 0.1
        beta = jax.nn.sigmoid(jax.random.normal(k4, (batch, seq_len, num_heads), dtype=jnp.float32))
        decay = jax.random.normal(k5, (batch, seq_len, num_heads), dtype=jnp.float32) * -0.01

        out_chunk, state_chunk = kernel_delta_attention(q, k, v, beta, decay, chunk_size=16, use_chunked=True)
        out_rec, state_rec = kernel_delta_attention(q, k, v, beta, decay, use_chunked=False)

        assert jnp.allclose(out_chunk, out_rec, atol=5e-4, rtol=5e-4)
        assert jnp.allclose(state_chunk, state_rec, atol=5e-4, rtol=5e-4)

    def test_single_step_matches_recurrent(self):
        key = jax.random.PRNGKey(7)
        batch, num_heads, head_dim, value_dim = 2, 3, 8, 8
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        q = jax.random.normal(k1, (batch, 1, num_heads, head_dim), dtype=jnp.float32) * 0.1
        k = jax.random.normal(k2, (batch, 1, num_heads, head_dim), dtype=jnp.float32) * 0.1
        v = jax.random.normal(k3, (batch, 1, num_heads, value_dim), dtype=jnp.float32) * 0.1
        beta = jax.nn.sigmoid(jax.random.normal(k4, (batch, 1, num_heads), dtype=jnp.float32))
        decay = jax.random.normal(k5, (batch, 1, num_heads), dtype=jnp.float32) * -0.01
        state0 = jax.random.normal(k6, (batch, num_heads, head_dim, value_dim), dtype=jnp.float32) * 0.01

        # kernel_delta_attention uses the single-step path when seq_len==1 and initial_state is provided.
        out_ss, state_ss = kernel_delta_attention(q, k, v, beta, decay, initial_state=state0)

        q_bhld = q.transpose(0, 2, 1, 3)
        k_bhld = k.transpose(0, 2, 1, 3)
        v_bhlv = v.transpose(0, 2, 1, 3)
        beta_bhl = beta.transpose(0, 2, 1)
        decay_bhl = decay.transpose(0, 2, 1)

        out_rec_bh, state_rec = _recurrent_kda_fwd(
            query=q_bhld,
            key=k_bhld,
            value=v_bhlv,
            beta=beta_bhl,
            decay=decay_bhl,
            softmax_scale=head_dim**-0.5,
            initial_state=state0,
            use_qk_l2norm=True,
        )
        out_rec = out_rec_bh.transpose(0, 2, 1, 3)

        assert jnp.allclose(out_ss, out_rec, atol=1e-5, rtol=1e-5)
        assert jnp.allclose(state_ss, state_rec, atol=1e-5, rtol=1e-5)

    def test_grad_shapes(self):
        key = jax.random.PRNGKey(123)
        batch, seq_len, num_heads, head_dim, value_dim = 1, 16, 2, 8, 8
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        q = jax.random.normal(k1, (batch, seq_len, num_heads, head_dim), dtype=jnp.float32) * 0.1
        k = jax.random.normal(k2, (batch, seq_len, num_heads, head_dim), dtype=jnp.float32) * 0.1
        v = jax.random.normal(k3, (batch, seq_len, num_heads, value_dim), dtype=jnp.float32) * 0.1
        beta = jax.nn.sigmoid(jax.random.normal(k4, (batch, seq_len, num_heads), dtype=jnp.float32))
        decay = jax.random.normal(k5, (batch, seq_len, num_heads), dtype=jnp.float32) * -0.01

        def loss_fn(q, k, v, beta, decay):
            out, _ = kernel_delta_attention(q, k, v, beta, decay, chunk_size=8)
            return jnp.sum(out)

        dq, dk, dv, dbeta, ddecay = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4))(q, k, v, beta, decay)
        assert dq.shape == q.shape
        assert dk.shape == k.shape
        assert dv.shape == v.shape
        assert dbeta.shape == beta.shape
        assert ddecay.shape == decay.shape


class TestKernelDeltaAttentionOperation:
    def test_forward_shape(self):
        batch, seq_len, num_heads, head_dim, value_dim = 2, 17, 4, 8, 8
        q = jnp.ones((batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        k = jnp.ones((batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        v = jnp.ones((batch, seq_len, num_heads, value_dim), dtype=jnp.float32)
        beta = jnp.ones((batch, seq_len, num_heads), dtype=jnp.float32) * 0.5
        decay = jnp.zeros((batch, seq_len, num_heads), dtype=jnp.float32)

        op = KernelDeltaAttention()
        cfg = KernelDeltaAttentionConfig(platform="xla", backend="any")
        out = op.run(q, k, v, beta, decay, chunk_size=8, cfg=cfg)
        assert out.shape == (batch, seq_len, num_heads, value_dim)

    def test_return_state(self):
        batch, seq_len, num_heads, head_dim, value_dim = 2, 17, 4, 8, 8
        q = jnp.ones((batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        k = jnp.ones((batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        v = jnp.ones((batch, seq_len, num_heads, value_dim), dtype=jnp.float32)
        beta = jnp.ones((batch, seq_len, num_heads), dtype=jnp.float32) * 0.5
        decay = jnp.zeros((batch, seq_len, num_heads), dtype=jnp.float32)

        op = KernelDeltaAttention()
        cfg = KernelDeltaAttentionConfig(platform="xla", backend="any")
        out, state = op.run(q, k, v, beta, decay, chunk_size=8, return_state=True, cfg=cfg)
        assert out.shape == (batch, seq_len, num_heads, value_dim)
        assert state.shape == (batch, num_heads, head_dim, value_dim)


if __name__ == "__main__":
    pytest.main([__file__])
