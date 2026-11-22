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


"""Tests for XLA page attention implementation."""

import jax.numpy as jnp
import pytest
from ejkernel.kernels._xla import page_attention


class TestPageAttention:
    """Test suite for page attention kernel (inference only)."""

    def test_forward_shape(self):
        """Test output shapes are correct."""
        num_seqs, num_heads, head_dim = 2, 8, 64
        num_blocks, num_kv_heads, block_size = 8, 4, 16

        query = jnp.ones((num_seqs, num_heads, head_dim))
        key_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))
        value_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))
        context_lens = jnp.array([48, 32])
        block_tables = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]])

        output = page_attention(query, key_cache, value_cache, context_lens, block_tables)

        assert output.shape == (num_seqs, num_heads, head_dim)

    def test_zero_context_returns_zero_not_mean_of_cache(self):
        """Empty rows must not attend to unused pages, including mixed-length batches."""
        query = jnp.zeros((2, 4, 2), dtype=jnp.float32)
        key_cache = jnp.zeros((2, 2, 2, 2), dtype=jnp.float32)
        value_cache = jnp.array([[[[2.0, 3.0], [4.0, 5.0]], [[7.0, 8.0], [9.0, 10.0]]]] * 2)
        context_lens = jnp.array([0, 1], dtype=jnp.int32)
        block_tables = jnp.array([[0, 1], [0, 1]], dtype=jnp.int32)
        output = page_attention(query, key_cache, value_cache, context_lens, block_tables)
        assert jnp.all(jnp.isfinite(output))
        assert jnp.array_equal(output[0], jnp.zeros((4, 2)))
        assert jnp.array_equal(output[1], value_cache[0, :, 0, :].repeat(2, axis=0))

    def test_fp16_softmax_accumulates_in_float32(self):
        """70k equal fp16 scores: a float16 softmax sum overflows and zeroes the output (r10 S04-3)."""
        block_size, num_blocks, head_dim = 1024, 70, 8
        context = 70000
        query = jnp.zeros((1, 1, head_dim), dtype=jnp.float16)
        key_cache = jnp.zeros((num_blocks, 1, block_size, head_dim), dtype=jnp.float16)
        value_cache = jnp.ones((num_blocks, 1, block_size, head_dim), dtype=jnp.float16)
        context_lens = jnp.array([context], dtype=jnp.int32)
        block_tables = jnp.arange(num_blocks, dtype=jnp.int32)[None, :]
        output = page_attention(query, key_cache, value_cache, context_lens, block_tables)
        assert output.dtype == jnp.float16
        assert jnp.allclose(output.astype(jnp.float32), 1.0, atol=1e-2), output

    def test_gqa_support(self):
        """Test Grouped Query Attention (GQA) with different num_heads and num_kv_heads."""
        num_seqs = 2
        num_heads = 8
        num_kv_heads = 2
        head_dim = 64
        num_blocks, block_size = 4, 16

        query = jnp.ones((num_seqs, num_heads, head_dim))
        key_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))
        value_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))
        context_lens = jnp.array([32, 16])
        block_tables = jnp.array([[0, 1], [2, 3]])

        output = page_attention(query, key_cache, value_cache, context_lens, block_tables)

        assert output.shape == (num_seqs, num_heads, head_dim)

    def test_different_context_lengths(self):
        """Test with different context lengths per sequence."""
        num_seqs, num_heads, head_dim = 3, 4, 32
        num_blocks, num_kv_heads, block_size = 6, 4, 8

        query = jnp.ones((num_seqs, num_heads, head_dim))
        key_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))
        value_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))

        context_lens = jnp.array([16, 8, 24])
        block_tables = jnp.array([[0, 1], [2, 3], [4, 5]])

        output = page_attention(query, key_cache, value_cache, context_lens, block_tables)

        assert output.shape == (num_seqs, num_heads, head_dim)

    def test_custom_scale(self):
        """Test with custom attention softmax_scale parameter (inference only)."""
        num_seqs, num_heads, head_dim = 2, 4, 16
        num_blocks, num_kv_heads, block_size = 4, 4, 8

        query = jnp.ones((num_seqs, num_heads, head_dim))
        key_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))
        value_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))
        context_lens = jnp.array([16, 8])
        block_tables = jnp.array([[0, 1], [2, 3]])

        output_custom = page_attention(query, key_cache, value_cache, context_lens, block_tables, attn_scale=0.5)
        assert output_custom.shape == (num_seqs, num_heads, head_dim)

    def test_block_table_indexing(self):
        """Test that block tables correctly index into cache."""
        num_seqs, num_heads, head_dim = 1, 2, 8
        num_blocks, num_kv_heads, block_size = 4, 2, 4

        query = jnp.ones((num_seqs, num_heads, head_dim))

        key_cache = jnp.arange(num_blocks * num_kv_heads * block_size * head_dim, dtype=jnp.float32).reshape(
            num_blocks, num_kv_heads, block_size, head_dim
        )
        value_cache = key_cache * 2

        context_lens = jnp.array([8])

        block_tables = jnp.array([[0, 1]])

        output = page_attention(query, key_cache, value_cache, context_lens, block_tables)

        assert output.shape == (num_seqs, num_heads, head_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
