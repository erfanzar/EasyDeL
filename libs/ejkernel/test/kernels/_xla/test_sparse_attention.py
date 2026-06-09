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


"""Tests for XLA native sparse attention implementation."""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._xla import apply_native_sparse_attention


class TestNativeSparseAttention:
    """Test suite for native sparse attention kernel."""

    def test_forward_shape(self):
        """Test output shapes are correct."""
        batch, seq_len, num_heads, head_dim = 2, 128, 8, 64
        block_size = 32

        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        block_counts = 2
        block_indices = jnp.tile(jnp.arange(2)[None, None, None, :], (batch, seq_len, num_heads, 1))

        output = apply_native_sparse_attention(q, k, v, block_indices, block_counts, block_size)
        assert output.shape == (batch, seq_len, num_heads, head_dim)
        assert jnp.isfinite(output).all()

    def test_gradient_shapes(self):
        """Test gradient shapes match input shapes and are finite."""
        key_q, key_k, key_v = jax.random.split(jax.random.PRNGKey(0), 3)
        batch, seq_len, num_heads, head_dim = 1, 64, 2, 16
        block_size = 32

        q = jax.random.normal(key_q, (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(key_k, (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(key_v, (batch, seq_len, num_heads, head_dim))

        block_counts = 2
        block_indices = jnp.tile(jnp.arange(2)[None, None, None, :], (batch, seq_len, num_heads, 1))

        def loss_fn(q, k, v):
            out = apply_native_sparse_attention(q, k, v, block_indices, block_counts, block_size)
            return jnp.sum(out)

        dq, dk, dv = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        assert dq.shape == q.shape and dk.shape == k.shape and dv.shape == v.shape
        assert jnp.isfinite(dq).all()
        assert jnp.isfinite(dk).all()
        assert jnp.isfinite(dv).all()

    def test_local_attention_pattern(self):
        """Test local attention pattern (each token attends to its own block)."""
        batch, seq_len, num_heads, head_dim = 1, 64, 1, 8
        block_size = 16

        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        block_indices = jnp.broadcast_to(
            (jnp.arange(seq_len) // block_size)[None, :, None, None], (batch, seq_len, num_heads, 1)
        ).astype(jnp.int32)
        block_counts = 1

        output = apply_native_sparse_attention(q, k, v, block_indices, block_counts, block_size)
        assert output.shape == (batch, seq_len, num_heads, head_dim)

        assert jnp.allclose(output, v, atol=1e-5)

    def test_custom_scale(self):
        """Test with custom attention softmax_scale parameter."""
        batch, seq_len, num_heads, head_dim = 1, 64, 2, 16
        block_size = 32

        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        block_indices = jnp.tile(jnp.arange(1)[None, None, None, :], (batch, seq_len, num_heads, 1))
        block_counts = 1

        output_custom = apply_native_sparse_attention(
            q, k, v, block_indices, block_counts, block_size, softmax_scale=0.5
        )
        assert output_custom.shape == (batch, seq_len, num_heads, head_dim)
        assert jnp.isfinite(output_custom).all()

    def test_variable_block_counts(self):
        """Test with different block counts per token."""
        batch, seq_len, num_heads, head_dim = 1, 64, 2, 16
        block_size = 32

        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        block_counts = jnp.ones((batch, seq_len, num_heads), dtype=jnp.int32)
        block_counts = block_counts.at[:, seq_len // 2 :, :].set(2)
        block_indices = jnp.tile(jnp.arange(2)[None, None, None, :], (batch, seq_len, num_heads, 1))

        output = apply_native_sparse_attention(q, k, v, block_indices, block_counts, block_size)
        assert output.shape == (batch, seq_len, num_heads, head_dim)
        assert jnp.isfinite(output).all()

    def test_gqa_grouping(self):
        """Test GQA case: num_q_heads > num_kv_heads."""
        batch, seq_len, num_q_heads, num_kv_heads, head_dim = 1, 64, 4, 2, 16
        block_size = 32

        S = 2
        block_indices = jnp.tile(jnp.arange(S)[None, None, None, :], (batch, seq_len, num_kv_heads, 1))
        block_counts = S

        q = jnp.ones((batch, seq_len, num_q_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_kv_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_kv_heads, head_dim))

        output = apply_native_sparse_attention(q, k, v, block_indices, block_counts, block_size)
        assert output.shape == (batch, seq_len, num_q_heads, head_dim)
        assert jnp.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
