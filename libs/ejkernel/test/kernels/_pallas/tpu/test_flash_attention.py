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

from ejkernel.kernels._pallas.tpu.flash_attention import flash_attention
from ejkernel.kernels._xla.flash_attention import flash_attention as flash_attention_xla
from ejkernel.ops import FwdParams


def _has_tpu():
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU tests require TPU backend")


class TestFlashAttentionTPU:
    """Test suite for TPU Pallas flash attention implementation."""

    def test_basic_forward_shape_and_finite(self):
        """Test basic forward pass with simple configuration."""
        batch_size = 2
        seq_len_q = 512
        seq_len_k = 512
        num_heads = 8
        head_dim = 128

        key = jax.random.PRNGKey(42)
        key, kq, kk, kv = jax.random.split(key, 4)

        query = jax.random.normal(kq, (batch_size, seq_len_q, num_heads, head_dim), dtype=jnp.float32)
        key_tensor = jax.random.normal(kk, (batch_size, seq_len_k, num_heads, head_dim), dtype=jnp.float32)
        value = jax.random.normal(kv, (batch_size, seq_len_k, num_heads, head_dim), dtype=jnp.float32)

        output = flash_attention(
            query=query,
            key=key_tensor,
            value=value,
        )

        assert output.shape == (batch_size, seq_len_q, num_heads, head_dim)
        assert jnp.isfinite(output).all()
        assert output.dtype == query.dtype

    def test_causal_masking(self):
        """Test causal masking functionality."""
        batch_size = 2
        seq_len = 256
        num_heads = 8
        head_dim = 128

        key = jax.random.PRNGKey(123)
        key, kq, kk, kv = jax.random.split(key, 4)

        query = jax.random.normal(kq, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)
        key_tensor = jax.random.normal(kk, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)
        value = jax.random.normal(kv, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)

        output = flash_attention(
            query=query,
            key=key_tensor,
            value=value,
            causal=True,
        )

        assert output.shape == (batch_size, seq_len, num_heads, head_dim)
        assert jnp.isfinite(output).all()

    def test_with_bias(self):
        """Test attention with bias."""
        batch_size = 2
        seq_len_q = 256
        seq_len_k = 256
        num_heads = 8
        head_dim = 64

        key = jax.random.PRNGKey(456)
        key, kq, kk, kv, kb = jax.random.split(key, 5)

        query = jax.random.normal(kq, (batch_size, seq_len_q, num_heads, head_dim), dtype=jnp.float32)
        key_tensor = jax.random.normal(kk, (batch_size, seq_len_k, num_heads, head_dim), dtype=jnp.float32)
        value = jax.random.normal(kv, (batch_size, seq_len_k, num_heads, head_dim), dtype=jnp.float32)
        bias = jax.random.normal(kb, (batch_size, num_heads, seq_len_q, seq_len_k), dtype=jnp.float32)

        output = flash_attention(
            query=query,
            key=key_tensor,
            value=value,
            bias=bias,
        )

        assert output.shape == (batch_size, seq_len_q, num_heads, head_dim)
        assert jnp.isfinite(output).all()

    def test_different_chunk_sizes(self):
        """Test with different chunk sizes."""
        batch_size = 1
        seq_len = 512
        num_heads = 8
        head_dim = 128

        key = jax.random.PRNGKey(789)
        key, kq, kk, kv = jax.random.split(key, 4)

        query = jax.random.normal(kq, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)
        key_tensor = jax.random.normal(kk, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)
        value = jax.random.normal(kv, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)

        for chunk_q, chunk_k in [(128, 128), (256, 256)]:
            output = flash_attention(
                query=query,
                key=key_tensor,
                value=value,
                fwd_params=FwdParams(q_blocksize=int(chunk_q), kv_blocksize=int(chunk_k)),
            )

            assert output.shape == (batch_size, seq_len, num_heads, head_dim)
            assert jnp.isfinite(output).all()

    def test_different_sequence_lengths(self):
        """Test with different Q and K sequence lengths."""
        batch_size = 2
        num_heads = 8
        head_dim = 128

        configs = [
            (256, 256),
            (128, 256),
            (512, 256),
        ]

        for seq_len_q, seq_len_k in configs:
            key = jax.random.PRNGKey(1000 + seq_len_q)
            key, kq, kk, kv = jax.random.split(key, 4)

            query = jax.random.normal(kq, (batch_size, seq_len_q, num_heads, head_dim), dtype=jnp.float32)
            key_tensor = jax.random.normal(kk, (batch_size, seq_len_k, num_heads, head_dim), dtype=jnp.float32)
            value = jax.random.normal(kv, (batch_size, seq_len_k, num_heads, head_dim), dtype=jnp.float32)

            output = flash_attention(
                query=query,
                key=key_tensor,
                value=value,
            )

            assert output.shape == (batch_size, seq_len_q, num_heads, head_dim)
            assert jnp.isfinite(output).all()

    def test_gqa_support(self):
        """Test Grouped Query Attention support."""
        batch_size = 2
        seq_len = 256
        num_q_heads = 32
        num_kv_heads = 8
        head_dim = 128

        key = jax.random.PRNGKey(2024)
        key, kq, kk, kv = jax.random.split(key, 4)

        query = jax.random.normal(kq, (batch_size, seq_len, num_q_heads, head_dim), dtype=jnp.float32)
        key_tensor = jax.random.normal(kk, (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)
        value = jax.random.normal(kv, (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)

        output = flash_attention(
            query=query,
            key=key_tensor,
            value=value,
        )

        assert output.shape == (batch_size, seq_len, num_q_heads, head_dim)
        assert jnp.isfinite(output).all()

    def test_mqa_support(self):
        """Test Multi-Query Attention (1 KV head) support."""
        batch_size = 2
        seq_len = 256
        num_q_heads = 8
        num_kv_heads = 1
        head_dim = 128

        key = jax.random.PRNGKey(2025)
        key, kq, kk, kv = jax.random.split(key, 4)

        query = jax.random.normal(kq, (batch_size, seq_len, num_q_heads, head_dim), dtype=jnp.float32)
        key_tensor = jax.random.normal(kk, (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)
        value = jax.random.normal(kv, (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)

        output = flash_attention(
            query=query,
            key=key_tensor,
            value=value,
        )

        assert output.shape == (batch_size, seq_len, num_q_heads, head_dim)
        assert jnp.isfinite(output).all()

    def test_dtypes(self):
        """Test with different dtypes."""
        batch_size = 2
        seq_len = 256
        num_heads = 8
        head_dim = 128

        for dtype in [jnp.float32, jnp.bfloat16]:
            key = jax.random.PRNGKey(3000)
            key, kq, kk, kv = jax.random.split(key, 4)

            query = jax.random.normal(kq, (batch_size, seq_len, num_heads, head_dim), dtype=dtype)
            key_tensor = jax.random.normal(kk, (batch_size, seq_len, num_heads, head_dim), dtype=dtype)
            value = jax.random.normal(kv, (batch_size, seq_len, num_heads, head_dim), dtype=dtype)

            output = flash_attention(
                query=query,
                key=key_tensor,
                value=value,
            )

            assert output.shape == (batch_size, seq_len, num_heads, head_dim)
            assert output.dtype == dtype
            assert jnp.isfinite(output).all()

    def test_different_head_dimensions(self):
        """Test with different head dimensions."""
        batch_size = 2
        seq_len = 256
        num_heads = 8

        for head_dim in [64, 128, 256]:
            key = jax.random.PRNGKey(4000 + head_dim)
            key, kq, kk, kv = jax.random.split(key, 4)

            query = jax.random.normal(kq, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)
            key_tensor = jax.random.normal(kk, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)
            value = jax.random.normal(kv, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)

            output = flash_attention(
                query=query,
                key=key_tensor,
                value=value,
            )

            assert output.shape == (batch_size, seq_len, num_heads, head_dim)
            assert jnp.isfinite(output).all()

    def test_custom_softmax_scale(self):
        """Test with custom softmax scale."""
        batch_size = 2
        seq_len = 256
        num_heads = 8
        head_dim = 128

        key = jax.random.PRNGKey(5000)
        key, kq, kk, kv = jax.random.split(key, 4)

        query = jax.random.normal(kq, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)
        key_tensor = jax.random.normal(kk, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)
        value = jax.random.normal(kv, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)

        for scale in [0.1, 0.5, 1.0]:
            output = flash_attention(
                query=query,
                key=key_tensor,
                value=value,
                softmax_scale=scale,
            )

            assert output.shape == (batch_size, seq_len, num_heads, head_dim)
            assert jnp.isfinite(output).all()

    def test_segment_ids(self):
        """Test with segment IDs for document boundaries."""
        batch_size = 2
        seq_len = 256
        num_heads = 8
        head_dim = 128

        key = jax.random.PRNGKey(6000)
        key, kq, kk, kv = jax.random.split(key, 4)

        query = jax.random.normal(kq, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)
        key_tensor = jax.random.normal(kk, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)
        value = jax.random.normal(kv, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)

        q_segment_ids = jnp.concatenate(
            [
                jnp.zeros((batch_size, seq_len // 2), dtype=jnp.int32),
                jnp.ones((batch_size, seq_len // 2), dtype=jnp.int32),
            ],
            axis=1,
        )
        kv_segment_ids = q_segment_ids

        output = flash_attention(
            query=query,
            key=key_tensor,
            value=value,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
        )

        assert output.shape == (batch_size, seq_len, num_heads, head_dim)
        assert jnp.isfinite(output).all()

    def test_numerical_correctness_vs_xla(self):
        """Test numerical correctness against XLA reference."""
        batch_size = 2
        seq_len = 256
        num_heads = 8
        head_dim = 128

        key = jax.random.PRNGKey(7000)
        key, kq, kk, kv = jax.random.split(key, 4)

        query_tpu = jax.random.normal(kq, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)
        key_tpu = jax.random.normal(kk, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)
        value_tpu = jax.random.normal(kv, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)

        output_tpu = flash_attention(
            query=query_tpu,
            key=key_tpu,
            value=value_tpu,
        )

        output_xla = flash_attention_xla(
            query=query_tpu,
            key=key_tpu,
            value=value_tpu,
        )

        assert output_tpu.shape == output_xla.shape
        assert jnp.allclose(output_tpu, output_xla, rtol=1e-2, atol=1e-2)

    def test_causal_correctness_vs_xla(self):
        """Test causal masking correctness against XLA reference."""
        batch_size = 2
        seq_len = 256
        num_heads = 8
        head_dim = 128

        key = jax.random.PRNGKey(8000)
        key, kq, kk, kv = jax.random.split(key, 4)

        query_tpu = jax.random.normal(kq, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)
        key_tpu = jax.random.normal(kk, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)
        value_tpu = jax.random.normal(kv, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)

        output_tpu = flash_attention(
            query=query_tpu,
            key=key_tpu,
            value=value_tpu,
            causal=True,
        )

        output_xla = flash_attention_xla(
            query=query_tpu,
            key=key_tpu,
            value=value_tpu,
            causal=True,
        )

        assert output_tpu.shape == output_xla.shape
        assert jnp.allclose(output_tpu, output_xla, rtol=1e-2, atol=1e-2)

    def test_backward_pass(self):
        """Test that backward pass works (gradient computation)."""
        batch_size = 2
        seq_len = 128
        num_heads = 4
        head_dim = 64

        key = jax.random.PRNGKey(9000)
        key, kq, kk, kv = jax.random.split(key, 4)

        query = jax.random.normal(kq, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)
        key_tensor = jax.random.normal(kk, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)
        value = jax.random.normal(kv, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)

        def loss_fn(q, k, v):
            out = flash_attention(query=q, key=k, value=v)
            return jnp.sum(out**2)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(query, key_tensor, value)

        assert len(grads) == 3
        for grad in grads:
            assert grad.shape in [query.shape, key_tensor.shape, value.shape]
            assert jnp.isfinite(grad).all()

    def test_single_batch_element(self):
        """Test with single batch element."""
        batch_size = 1
        seq_len = 256
        num_heads = 8
        head_dim = 128

        key = jax.random.PRNGKey(10000)
        key, kq, kk, kv = jax.random.split(key, 4)

        query = jax.random.normal(kq, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)
        key_tensor = jax.random.normal(kk, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)
        value = jax.random.normal(kv, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)

        output = flash_attention(
            query=query,
            key=key_tensor,
            value=value,
        )

        assert output.shape == (batch_size, seq_len, num_heads, head_dim)
        assert jnp.isfinite(output).all()

    def test_large_sequence(self):
        """Test with larger sequence length."""
        batch_size = 1
        seq_len = 2048
        num_heads = 8
        head_dim = 128

        key = jax.random.PRNGKey(11000)
        key, kq, kk, kv = jax.random.split(key, 4)

        query = jax.random.normal(kq, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)
        key_tensor = jax.random.normal(kk, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)
        value = jax.random.normal(kv, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)

        output = flash_attention(
            query=query,
            key=key_tensor,
            value=value,
            fwd_params=FwdParams(q_blocksize=512, kv_blocksize=512),
        )

        assert output.shape == (batch_size, seq_len, num_heads, head_dim)
        assert jnp.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
