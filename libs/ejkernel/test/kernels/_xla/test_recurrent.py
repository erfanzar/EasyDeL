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


"""Tests for XLA recurrent attention implementation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ejkernel.kernels._xla import lightning_attn, recurrent, recurrent_gla
from ejkernel.kernels._xla.recurrent._xla_impl_fwd import _recurrent_attention_fwd


class TestRecurrentAttention:
    """Test suite for recurrent attention kernel."""

    def test_forward_shape(self):
        """Test output shapes are correct."""
        batch, seq_len, num_heads, head_dim = 2, 10, 4, 16
        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        output, final_state = recurrent(q, k, v)

        assert output.shape == (batch, seq_len, num_heads, head_dim)
        assert final_state.shape == (batch, num_heads, head_dim, head_dim)

    def test_gradient_shapes(self):
        """Test gradient shapes match input shapes."""
        batch, seq_len, num_heads, head_dim = 1, 5, 2, 8
        q = jnp.ones((batch, seq_len, num_heads, head_dim)) * 0.5
        k = jnp.ones((batch, seq_len, num_heads, head_dim)) * 0.5
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        def loss_fn(q, k, v):
            o, _ = recurrent(q, k, v)
            return jnp.sum(o)

        dq, dk, dv = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert dq.shape == q.shape
        assert dk.shape == k.shape
        assert dv.shape == v.shape

    def test_orthogonal_vectors(self):
        """Test with orthogonal query/key vectors."""

        q = jnp.array([[[[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0, 0.0]]]])
        k = q
        v = jnp.array([[[[2.0, 0.0, 0.0, 0.0]], [[0.0, 3.0, 0.0, 0.0]], [[0.0, 0.0, 4.0, 0.0]]]])

        output, _ = recurrent(q, k, v, softmax_scale=1.0)

        expected = jnp.array([[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0]])
        assert jnp.allclose(output[0, :, 0, :], expected, atol=1e-5)

    def test_with_initial_state(self):
        """Test with non-zero initial state."""
        batch, seq_len, num_heads, head_dim = 1, 3, 1, 4
        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        initial_state = jnp.ones((batch, num_heads, head_dim, head_dim)) * 0.5

        output, final_state = recurrent(q, k, v, initial_state=initial_state)

        assert output.shape == (batch, seq_len, num_heads, head_dim)
        assert final_state.shape == (batch, num_heads, head_dim, head_dim)

        assert not jnp.allclose(final_state, initial_state)

    def test_reverse_mode(self):
        """Test reverse processing."""
        batch, seq_len, num_heads, head_dim = 1, 5, 2, 8
        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        output_fwd, _ = recurrent(q, k, v, reverse=False)
        output_rev, _ = recurrent(q, k, v, reverse=True)

        assert output_fwd.shape == output_rev.shape

        assert not jnp.allclose(output_fwd, output_rev)

    def test_varlen_backward_matches_batched_and_vjp(self):
        """Test packed varlen backward under jit + finite-diff + VJP."""
        batch, seq_len, num_heads, head_dim = 2, 6, 2, 8
        key = jax.random.PRNGKey(0)
        kq, kk, kv, ks = jax.random.split(key, 4)

        q = jax.random.normal(kq, (batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        k = jax.random.normal(kk, (batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        v = jax.random.normal(kv, (batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        h0 = jax.random.normal(ks, (batch, num_heads, head_dim, head_dim), dtype=jnp.float32)

        q_p = q.reshape(1, batch * seq_len, num_heads, head_dim)
        k_p = k.reshape(1, batch * seq_len, num_heads, head_dim)
        v_p = v.reshape(1, batch * seq_len, num_heads, head_dim)
        cu = jnp.arange(0, (batch + 1) * seq_len, seq_len, dtype=jnp.int32)

        def loss_varlen(q_, k_, v_, h0_):
            out, st = recurrent(q_, k_, v_, initial_state=h0_, cu_seqlens=cu)
            return jnp.mean(out) + 0.1 * jnp.mean(st)

        def loss_batched_reference(q_, k_, v_, h0_):
            g_ = jnp.zeros_like(q_)
            gk_ = jnp.zeros_like(q_)
            gv_ = jnp.zeros_like(v_)
            g_gamma_ = jnp.zeros((q_.shape[-2],), dtype=q_.dtype)
            out, _, st = _recurrent_attention_fwd(
                q_,
                k_,
                v_,
                g_,
                g_gamma_,
                gk_,
                gv_,
                None,
                h0_,
                False,
            )
            return jnp.mean(out) + 0.1 * jnp.mean(st)

        grads_varlen = jax.jit(jax.grad(loss_varlen, argnums=(0, 1, 2, 3)))(q_p, k_p, v_p, h0)
        grads_batched = jax.jit(jax.grad(loss_batched_reference, argnums=(0, 1, 2, 3)))(q, k, v, h0)
        grads_varlen = jax.tree_util.tree_map(jax.block_until_ready, grads_varlen)
        grads_batched = jax.tree_util.tree_map(jax.block_until_ready, grads_batched)

        dq_varlen = grads_varlen[0].reshape(batch, seq_len, num_heads, head_dim)
        dk_varlen = grads_varlen[1].reshape(batch, seq_len, num_heads, head_dim)
        dv_varlen = grads_varlen[2].reshape(batch, seq_len, num_heads, head_dim)
        dh0_varlen = grads_varlen[3]

        np.testing.assert_allclose(
            np.asarray(dq_varlen, dtype=np.float32),
            np.asarray(grads_batched[0], dtype=np.float32),
            rtol=2e-4,
            atol=2e-4,
        )
        np.testing.assert_allclose(
            np.asarray(dk_varlen, dtype=np.float32),
            np.asarray(grads_batched[1], dtype=np.float32),
            rtol=2e-4,
            atol=2e-4,
        )
        np.testing.assert_allclose(
            np.asarray(dv_varlen, dtype=np.float32),
            np.asarray(grads_batched[2], dtype=np.float32),
            rtol=2e-4,
            atol=2e-4,
        )
        np.testing.assert_allclose(
            np.asarray(dh0_varlen, dtype=np.float32),
            np.asarray(grads_batched[3], dtype=np.float32),
            rtol=2e-4,
            atol=2e-4,
        )

        # Spot-check one element against finite differences for numerical sanity.
        idx = (0, seq_len // 2, 0, 0)
        eps = 1e-3
        delta = jnp.zeros_like(k_p).at[idx].set(eps)
        loss_jit = jax.jit(loss_varlen)
        fd = (loss_jit(q_p, k_p + delta, v_p, h0) - loss_jit(q_p, k_p - delta, v_p, h0)) / (2.0 * eps)
        np.testing.assert_allclose(
            np.asarray(grads_varlen[1][idx], dtype=np.float32),
            np.asarray(fd, dtype=np.float32),
            rtol=5e-2,
            atol=5e-3,
        )

        def fwd_varlen(q_, k_, v_, h0_):
            return recurrent(q_, k_, v_, initial_state=h0_, cu_seqlens=cu)

        @jax.jit
        def vjp_eval(q_, k_, v_, h0_):
            (out, st), vjp_fn = jax.vjp(fwd_varlen, q_, k_, v_, h0_)
            return vjp_fn((jnp.ones_like(out), jnp.ones_like(st)))

        vjp_grads = vjp_eval(q_p, k_p, v_p, h0)
        for grad in vjp_grads:
            assert jnp.all(jnp.isfinite(grad))


class TestGLA:
    """Test suite for Gated Linear Attention."""

    def test_forward_shape(self):
        """Test GLA output shapes."""
        batch, seq_len, num_heads, head_dim = 2, 10, 4, 16
        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))
        g = jnp.zeros((batch, seq_len, num_heads, head_dim))

        output, final_state = recurrent_gla(q, k, v, g=g)

        assert output.shape == (batch, seq_len, num_heads, head_dim)
        assert final_state.shape == (batch, num_heads, head_dim, head_dim)

    def test_gradient_shapes(self):
        """Test GLA gradient shapes."""
        batch, seq_len, num_heads, head_dim = 1, 5, 2, 8
        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))
        g = jnp.zeros((batch, seq_len, num_heads, head_dim))

        def loss_fn(q, k, v):
            o, _ = recurrent_gla(q, k, v, g=g)
            return jnp.sum(o)

        dq, dk, dv = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert dq.shape == q.shape
        assert dk.shape == k.shape
        assert dv.shape == v.shape


class TestLightningAttention:
    """Test suite for Lightning Attention."""

    def test_forward_shape(self):
        """Test Lightning attention output shapes."""
        batch, seq_len, num_heads, head_dim = 2, 10, 4, 16
        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        output, final_state = lightning_attn(q, k, v, layer_idx=5, num_layers=24)

        assert output.shape == (batch, seq_len, num_heads, head_dim)
        assert final_state.shape == (batch, num_heads, head_dim, head_dim)

    def test_layer_dependent_decay(self):
        """Test that different layers produce different outputs."""
        batch, seq_len, num_heads, head_dim = 1, 5, 4, 8
        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        output_layer0, _ = lightning_attn(q, k, v, layer_idx=0, num_layers=12)
        output_layer11, _ = lightning_attn(q, k, v, layer_idx=11, num_layers=12)

        assert not jnp.allclose(output_layer0, output_layer11)

    def test_gradient_shapes(self):
        """Test Lightning attention gradient shapes."""
        batch, seq_len, num_heads, head_dim = 1, 5, 2, 8
        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        def loss_fn(q, k, v):
            o, _ = lightning_attn(q, k, v, layer_idx=3, num_layers=12)
            return jnp.sum(o)

        dq, dk, dv = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert dq.shape == q.shape
        assert dk.shape == k.shape
        assert dv.shape == v.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
