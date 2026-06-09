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


import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._pallas.tpu.page_attention import page_attention
from ejkernel.kernels._xla.page_attention import page_attention as page_attention_xla


def _has_tpu():
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU tests require TPU backend")


class TestPageAttentionTPU:
    """Test suite for TPU Pallas page attention implementation."""

    def test_basic_forward_shape_and_finite(self):
        """Test basic forward pass with simple configuration."""
        num_seqs = 4
        num_q_heads = 8
        num_kv_heads = 2
        head_dim = 128
        block_size = 128
        num_blocks = 32

        key = jax.random.PRNGKey(42)
        key, k1, k2, k3 = jax.random.split(key, 4)

        query = jax.random.normal(k1, (num_seqs, num_q_heads, head_dim), dtype=jnp.float32)
        key_cache = jax.random.normal(k2, (num_kv_heads, num_blocks, block_size, head_dim), dtype=jnp.float32)
        value_cache = jax.random.normal(k3, (num_kv_heads, num_blocks, block_size, head_dim), dtype=jnp.float32)

        context_lens = jnp.array([256, 256, 128, 384], dtype=jnp.int32)
        block_tables = jnp.array(
            [
                [0, 1, 2, 3, 0, 0, 0, 0],
                [4, 5, 6, 0, 0, 0, 0, 0],
                [7, 8, 0, 0, 0, 0, 0, 0],
                [9, 10, 11, 12, 13, 0, 0, 0],
            ],
            dtype=jnp.int32,
        )

        output = page_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            context_lens=context_lens,
            block_tables=block_tables,
            pages_per_compute_block=2,
        )

        assert output.shape == (num_seqs, num_q_heads, head_dim)
        assert jnp.isfinite(output).all()
        assert output.dtype == query.dtype

    def test_single_sequence_single_block(self):
        """Test with a single sequence fitting in one block."""
        num_seqs = 4
        num_q_heads = 8
        num_kv_heads = 2
        head_dim = 128
        block_size = 128
        num_blocks = 8

        key = jax.random.PRNGKey(123)
        key, k1, k2, k3 = jax.random.split(key, 4)

        query = jax.random.normal(k1, (num_seqs, num_q_heads, head_dim), dtype=jnp.float32)
        key_cache = jax.random.normal(k2, (num_kv_heads, num_blocks, block_size, head_dim), dtype=jnp.float32)
        value_cache = jax.random.normal(k3, (num_kv_heads, num_blocks, block_size, head_dim), dtype=jnp.float32)

        context_lens = jnp.array([128, 128, 256, 128], dtype=jnp.int32)
        block_tables = jnp.array([[0, 0], [1, 0], [2, 3], [4, 0]], dtype=jnp.int32)

        output = page_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            context_lens=context_lens,
            block_tables=block_tables,
            pages_per_compute_block=2,
        )

        assert output.shape == (num_seqs, num_q_heads, head_dim)
        assert jnp.isfinite(output).all()

    def test_multiple_blocks_per_sequence(self):
        """Test sequences spanning multiple blocks."""
        num_seqs = 2
        num_q_heads = 8
        num_kv_heads = 8
        head_dim = 128
        block_size = 128
        num_blocks = 16

        key = jax.random.PRNGKey(456)
        key, k1, k2, k3 = jax.random.split(key, 4)

        query = jax.random.normal(k1, (num_seqs, num_q_heads, head_dim), dtype=jnp.float32)
        key_cache = jax.random.normal(k2, (num_kv_heads, num_blocks, block_size, head_dim), dtype=jnp.float32)
        value_cache = jax.random.normal(k3, (num_kv_heads, num_blocks, block_size, head_dim), dtype=jnp.float32)

        context_lens = jnp.array([384, 256], dtype=jnp.int32)
        block_tables = jnp.array(
            [
                [0, 1, 2, 0],
                [3, 4, 0, 0],
            ],
            dtype=jnp.int32,
        )

        output = page_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            context_lens=context_lens,
            block_tables=block_tables,
            pages_per_compute_block=2,
        )

        assert output.shape == (num_seqs, num_q_heads, head_dim)
        assert jnp.isfinite(output).all()

    def test_gqa_with_different_head_ratios(self):
        """Test Grouped Query Attention with various head count ratios."""
        block_size = 128
        num_blocks = 16
        head_dim = 128

        configs = [
            (8, 2),
            (16, 4),
            (32, 8),
        ]

        for num_q_heads, num_kv_heads in configs:
            key = jax.random.PRNGKey(789 + num_q_heads)
            key, k1, k2, k3 = jax.random.split(key, 4)

            query = jax.random.normal(k1, (4, num_q_heads, head_dim), dtype=jnp.float32)
            key_cache = jax.random.normal(k2, (num_kv_heads, num_blocks, block_size, head_dim), dtype=jnp.float32)
            value_cache = jax.random.normal(k3, (num_kv_heads, num_blocks, block_size, head_dim), dtype=jnp.float32)

            context_lens = jnp.array([256, 256, 384, 128], dtype=jnp.int32)
            block_tables = jnp.array([[0, 1, 0, 0], [2, 3, 0, 0], [4, 5, 6, 0], [7, 0, 0, 0]], dtype=jnp.int32)

            output = page_attention(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                context_lens=context_lens,
                block_tables=block_tables,
                pages_per_compute_block=2,
            )

            assert output.shape == (4, num_q_heads, head_dim)
            assert jnp.isfinite(output).all()

    def test_varying_context_lengths(self):
        """Test with varying context lengths across sequences."""
        num_seqs = 4
        num_q_heads = 8
        num_kv_heads = 4
        head_dim = 128
        block_size = 128
        num_blocks = 16

        key = jax.random.PRNGKey(999)
        key, k1, k2, k3 = jax.random.split(key, 4)

        query = jax.random.normal(k1, (num_seqs, num_q_heads, head_dim), dtype=jnp.float32)
        key_cache = jax.random.normal(k2, (num_kv_heads, num_blocks, block_size, head_dim), dtype=jnp.float32)
        value_cache = jax.random.normal(k3, (num_kv_heads, num_blocks, block_size, head_dim), dtype=jnp.float32)

        context_lens = jnp.array([128, 256, 384, 512], dtype=jnp.int32)
        block_tables = jnp.array(
            [
                [0, 0, 0, 0],
                [1, 2, 0, 0],
                [3, 4, 5, 0],
                [6, 7, 8, 9],
            ],
            dtype=jnp.int32,
        )

        output = page_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            context_lens=context_lens,
            block_tables=block_tables,
            pages_per_compute_block=2,
        )

        assert output.shape == (num_seqs, num_q_heads, head_dim)
        assert jnp.isfinite(output).all()

    def test_dtypes_consistency(self):
        """Test that different dtypes are handled correctly."""
        num_seqs = 4
        num_q_heads = 8
        num_kv_heads = 2
        head_dim = 128
        block_size = 128
        num_blocks = 8

        for dtype in [jnp.float32, jnp.bfloat16]:
            key = jax.random.PRNGKey(111)
            key, k1, k2, k3 = jax.random.split(key, 4)

            query = jax.random.normal(k1, (num_seqs, num_q_heads, head_dim), dtype=dtype)
            key_cache = jax.random.normal(k2, (num_kv_heads, num_blocks, block_size, head_dim), dtype=dtype)
            value_cache = jax.random.normal(k3, (num_kv_heads, num_blocks, block_size, head_dim), dtype=dtype)

            context_lens = jnp.array([128, 128, 256, 128], dtype=jnp.int32)
            block_tables = jnp.array([[0, 0], [1, 0], [2, 3], [4, 0]], dtype=jnp.int32)

            output = page_attention(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                context_lens=context_lens,
                block_tables=block_tables,
                pages_per_compute_block=2,
            )

            assert output.shape == (num_seqs, num_q_heads, head_dim)
            assert output.dtype == dtype
            assert jnp.isfinite(output).all()

    def test_numerical_correctness_vs_xla(self):
        """Test numerical correctness against XLA reference implementation."""
        num_seqs = 2
        num_q_heads = 8
        num_kv_heads = 2
        head_dim = 128
        block_size = 128
        num_blocks = 8

        key = jax.random.PRNGKey(2024)
        key, k1, k2, k3 = jax.random.split(key, 4)

        query = jax.random.normal(k1, (num_seqs, num_q_heads, head_dim), dtype=jnp.float32)
        key_cache = jax.random.normal(k2, (num_kv_heads, num_blocks, block_size, head_dim), dtype=jnp.float32)
        value_cache = jax.random.normal(k3, (num_kv_heads, num_blocks, block_size, head_dim), dtype=jnp.float32)

        context_lens = jnp.array([256, 512], dtype=jnp.int32)
        block_tables = jnp.array([[0, 1, 0, 0], [2, 3, 4, 5]], dtype=jnp.int32)

        output_tpu = page_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            context_lens=context_lens,
            block_tables=block_tables,
            pages_per_compute_block=2,
        )

        output_xla = page_attention_xla(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            context_lens=context_lens,
            block_tables=block_tables,
        )

        assert output_tpu.shape == output_xla.shape
        err = jnp.mean(jnp.abs(output_tpu.ravel())) - jnp.mean(jnp.abs(output_xla.ravel()))
        assert jnp.allclose(output_tpu, output_xla, rtol=0, atol=0.125), f"err : {err}"

    def test_error_on_invalid_inputs(self):
        """Test that invalid inputs raise appropriate errors."""
        num_seqs = 4
        num_q_heads = 8
        num_kv_heads = 2
        head_dim = 128
        block_size = 128
        num_blocks = 16

        key = jax.random.PRNGKey(555)
        key, k1, k2, k3 = jax.random.split(key, 4)

        query = jax.random.normal(k1, (num_seqs, num_q_heads, head_dim), dtype=jnp.float32)
        key_cache = jax.random.normal(k2, (num_kv_heads, num_blocks, block_size, head_dim), dtype=jnp.float32)
        value_cache = jax.random.normal(k3, (num_kv_heads, num_blocks, block_size, head_dim), dtype=jnp.float32)

        context_lens = jnp.array([128, 256, 128, 384], dtype=jnp.int32)
        block_tables = jnp.array([[0, 0, 0], [1, 2, 0], [3, 0, 0], [4, 5, 6]], dtype=jnp.int32)

        with pytest.raises(ValueError, match="pages_per_compute_block must be divisible"):
            page_attention(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                context_lens=context_lens,
                block_tables=block_tables,
                pages_per_compute_block=2,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
