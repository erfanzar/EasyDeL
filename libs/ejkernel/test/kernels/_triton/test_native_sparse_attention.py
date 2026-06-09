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


"""Comprehensive tests for native_sparse_attention Triton implementation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ejkernel.kernels._triton.native_sparse_attention import (
    apply_native_sparse_attention,
    native_sparse_attention,
)
from ejkernel.kernels._xla.native_sparse_attention import native_sparse_attention as xla_native_sparse_attention
from ejkernel.utils import generate_block_indices

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Triton tests require GPU backend")


def _to_token_block_indices(block_indices, seq_len: int, block_size: int):
    """
    Convert block-level indices [B, num_query_blocks, H_kv, S]
    to per-token indices [B, seq_len, H_kv, S], if needed.
    If already per-token, returns as-is.
    """
    if block_indices.shape[1] == seq_len:
        return block_indices
    num_query_blocks = seq_len // block_size
    assert block_indices.shape[1] == num_query_blocks, (
        f"block_indices has unexpected shape {block_indices.shape}, expected second dim {num_query_blocks} or {seq_len}"
    )

    qb_map = jnp.arange(seq_len) // block_size

    return jnp.take(block_indices, qb_map, axis=1)


class TestNativeSparseAttentionForward:
    """Test forward pass of native sparse attention."""

    @pytest.mark.parametrize(
        "batch_size,seq_len,num_q_heads,num_kv_heads,head_dim,block_size,num_blocks",
        [
            (1, 128, 16, 1, 64, 64, 2),
            (2, 256, 32, 2, 64, 64, 4),
            (1, 256, 64, 4, 64, 64, 4),
            (2, 512, 64, 2, 128, 64, 8),
            (1, 256, 128, 8, 64, 64, 4),
        ],
    )
    def test_forward_shapes(self, batch_size, seq_len, num_q_heads, num_kv_heads, head_dim, block_size, num_blocks):
        """Test that forward pass produces correct output shapes."""
        key_q = jax.random.PRNGKey(0)
        key_k = jax.random.PRNGKey(1)
        key_v = jax.random.PRNGKey(2)

        q = jax.random.normal(key_q, (batch_size, seq_len, num_q_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(key_k, (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(key_v, (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)

        num_query_blocks = seq_len // block_size
        block_indices_blocks = generate_block_indices(batch_size, num_query_blocks, num_kv_heads, num_blocks, block_size)
        block_indices = _to_token_block_indices(block_indices_blocks, seq_len, block_size)

        softmax_scale = float(1.0 / jnp.sqrt(head_dim))
        output = apply_native_sparse_attention(q, k, v, block_indices, num_blocks, block_size, softmax_scale)

        assert output.shape == (batch_size, seq_len, num_q_heads, head_dim)
        assert output.dtype == jnp.float16
        assert not jnp.any(jnp.isnan(output)), "NaN found in output"
        assert not jnp.any(jnp.isinf(output)), "Inf found in output"

    def test_forward_no_nans_or_infs(self):
        """Test that forward pass doesn't produce NaNs or Infs."""
        batch_size, seq_len, num_q_heads, num_kv_heads, head_dim = 2, 256, 32, 2, 64
        block_size = 64
        num_blocks = 4

        q = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, num_q_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(jax.random.PRNGKey(2), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)

        num_query_blocks = seq_len // block_size
        block_indices_blocks = generate_block_indices(batch_size, num_query_blocks, num_kv_heads, num_blocks, block_size)
        block_indices = _to_token_block_indices(block_indices_blocks, seq_len, block_size)

        softmax_scale = float(1.0 / jnp.sqrt(head_dim))
        output = apply_native_sparse_attention(q, k, v, block_indices, num_blocks, block_size, softmax_scale)

        assert not jnp.any(jnp.isnan(output)), "Output contains NaN values"
        assert not jnp.any(jnp.isinf(output)), "Output contains Inf values"

    def test_forward_gqa_assertion(self):
        """Test that GQA group size assertion works correctly."""
        batch_size, seq_len, num_q_heads, num_kv_heads, head_dim = 1, 128, 15, 1, 64
        block_size = 64

        q = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, num_q_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(jax.random.PRNGKey(2), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)

        with pytest.raises(AssertionError, match="Group size must be a multiple of 16"):
            native_sparse_attention(q, k, v, block_size=block_size)


class TestNativeSparseAttentionBackward:
    """Test backward pass of native sparse attention."""

    def test_backward_gradients_exist(self):
        """Test that backward pass computes gradients for all inputs."""
        batch_size, seq_len, num_q_heads, num_kv_heads, head_dim = 1, 128, 16, 1, 64
        block_size = 64
        num_blocks = 2

        q = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, num_q_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(jax.random.PRNGKey(2), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)

        num_query_blocks = seq_len // block_size
        block_indices_blocks = generate_block_indices(batch_size, num_query_blocks, num_kv_heads, num_blocks, block_size)
        block_indices = _to_token_block_indices(block_indices_blocks, seq_len, block_size)
        softmax_scale = float(1.0 / jnp.sqrt(head_dim))

        def loss_fn(q, k, v):
            output = apply_native_sparse_attention(q, k, v, block_indices, num_blocks, block_size, softmax_scale)
            return jnp.sum(output**2)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert grads[0].shape == q.shape
        assert grads[1].shape == k.shape
        assert grads[2].shape == v.shape

        assert jnp.abs(grads[0]).max() > 0
        assert jnp.abs(grads[1]).max() > 0
        assert jnp.abs(grads[2]).max() > 0

        assert not jnp.any(jnp.isnan(grads[0]))
        assert not jnp.any(jnp.isnan(grads[1]))
        assert not jnp.any(jnp.isnan(grads[2]))

    @pytest.mark.parametrize(
        "batch_size,seq_len,num_q_heads,num_kv_heads,head_dim",
        [
            (1, 128, 16, 1, 64),
            (2, 256, 32, 2, 64),
            (1, 256, 64, 4, 128),
        ],
    )
    def test_backward_no_nans(self, batch_size, seq_len, num_q_heads, num_kv_heads, head_dim):
        """Test that backward pass doesn't produce NaNs."""
        block_size = 64
        num_blocks = 4

        q = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, num_q_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(jax.random.PRNGKey(2), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)

        num_query_blocks = seq_len // block_size
        block_indices_blocks = generate_block_indices(batch_size, num_query_blocks, num_kv_heads, num_blocks, block_size)
        block_indices = _to_token_block_indices(block_indices_blocks, seq_len, block_size)
        softmax_scale = float(1.0 / jnp.sqrt(head_dim))

        def loss_fn(q, k, v):
            output = apply_native_sparse_attention(q, k, v, block_indices, num_blocks, block_size, softmax_scale)
            return jnp.sum(output**2)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert not jnp.any(jnp.isnan(grads[0]))
        assert not jnp.any(jnp.isnan(grads[1]))
        assert not jnp.any(jnp.isnan(grads[2]))


class TestNativeSparseAttentionWithCompression:
    """Test native sparse attention with compression gate."""

    def test_with_compression_gate(self):
        """Test NSA with compression gate produces valid output."""
        batch_size, seq_len, num_q_heads, num_kv_heads, head_dim = 2, 256, 32, 2, 64
        block_size = 64
        num_blocks = 4

        q = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, num_q_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(jax.random.PRNGKey(2), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)
        g_cmp = jax.random.uniform(jax.random.PRNGKey(3), (batch_size, seq_len, num_q_heads), dtype=jnp.float16)

        output = native_sparse_attention(
            query=q,
            key=k,
            value=v,
            g_cmp=g_cmp,
            block_counts=num_blocks,
            block_size=block_size,
        )

        assert output.shape == (batch_size, seq_len, num_q_heads, head_dim)
        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))

    def test_with_both_gates(self):
        """Test NSA with both compression and selection gates."""
        batch_size, seq_len, num_q_heads, num_kv_heads, head_dim = 1, 256, 16, 1, 64
        block_size = 64
        num_blocks = 4

        q = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, num_q_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(jax.random.PRNGKey(2), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)
        g_cmp = jax.random.uniform(jax.random.PRNGKey(3), (batch_size, seq_len, num_q_heads), dtype=jnp.float16)
        g_slc = jax.random.uniform(jax.random.PRNGKey(4), (batch_size, seq_len, num_q_heads), dtype=jnp.float16)

        output = native_sparse_attention(
            query=q,
            key=k,
            value=v,
            g_cmp=g_cmp,
            g_slc=g_slc,
            block_counts=num_blocks,
            block_size=block_size,
        )

        assert output.shape == (batch_size, seq_len, num_q_heads, head_dim)
        assert not jnp.any(jnp.isnan(output))

    def test_compression_backward(self):
        """Test that compression gate version has valid gradients with precomputed indices."""
        batch_size, seq_len, num_q_heads, num_kv_heads, head_dim = 1, 128, 16, 1, 64
        block_size = 64
        num_blocks = 2

        q = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, num_q_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(jax.random.PRNGKey(2), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)
        g_slc = jax.random.uniform(jax.random.PRNGKey(3), (batch_size, seq_len, num_q_heads), dtype=jnp.float16)

        num_query_blocks = seq_len // block_size
        block_indices_blocks = generate_block_indices(batch_size, num_query_blocks, num_kv_heads, num_blocks, block_size)
        block_indices = _to_token_block_indices(block_indices_blocks, seq_len, block_size)

        def loss_fn(q, k, v):
            output = native_sparse_attention(
                q, k, v, g_slc=g_slc, block_indices=block_indices, block_counts=num_blocks, block_size=block_size
            )
            return jnp.sum(output**2)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert not jnp.any(jnp.isnan(grads[0]))
        assert not jnp.any(jnp.isnan(grads[1]))
        assert not jnp.any(jnp.isnan(grads[2]))


def test_native_sparse_attention_matches_xla_smoke():
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)

    batch_size, seq_len, num_q_heads, num_kv_heads, head_dim = 1, 128, 16, 1, 64
    block_size = 64
    block_counts = 2

    q = jax.random.normal(kq, (batch_size, seq_len, num_q_heads, head_dim), dtype=jnp.float16)
    k = jax.random.normal(kk, (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)
    v = jax.random.normal(kv, (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)

    block_indices = jnp.broadcast_to(
        jnp.arange(block_counts, dtype=jnp.int32)[None, None, None, :],
        (batch_size, seq_len, num_kv_heads, block_counts),
    )

    out_triton = native_sparse_attention(
        query=q,
        key=k,
        value=v,
        g_cmp=None,
        g_slc=None,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
    )
    out_xla = xla_native_sparse_attention(
        query=q,
        key=k,
        value=v,
        g_cmp=None,
        g_slc=None,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
    )

    out_triton = jax.block_until_ready(out_triton)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(
        np.asarray(out_triton, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
