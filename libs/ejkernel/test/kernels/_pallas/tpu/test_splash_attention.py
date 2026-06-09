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


"""Tests for Splash Attention with sliding window and chunked causal mask support."""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._pallas.tpu.blocksparse_attention import (
    CausalMask,
    ChunkedCausalMask,
    LocalMask,
    blocksparse_attention,
    make_causal_mask,
    make_chunk_attention_mask,
    make_local_attention_mask,
)
from ejkernel.kernels._xla.blocksparse_attention import blocksparse_attention as blocksparse_attention_xla
from ejkernel.ops import FwdParams


def _has_tpu() -> bool:
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU tests require TPU backend")


class TestBasicFunctionality:
    """Test basic blocksparse_attention attention functionality."""

    def test_basic_causal_attention(self):
        """Test basic causal attention."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 1024
        num_heads = 8
        head_dim = 64

        query = jax.random.normal(keys[0], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        key_array = jax.random.normal(keys[1], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        value = jax.random.normal(keys[2], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)

        output = blocksparse_attention(
            query=query,
            key=key_array,
            value=value,
            causal=True,
        )

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert jnp.all(jnp.isfinite(output))

    def test_full_attention(self):
        """Test full attention (non-causal)."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 512
        num_heads = 4
        head_dim = 64

        query = jax.random.normal(keys[0], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        key_array = jax.random.normal(keys[1], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        value = jax.random.normal(keys[2], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)

        output = blocksparse_attention(
            query=query,
            key=key_array,
            value=value,
            causal=False,
        )

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert jnp.all(jnp.isfinite(output))

    def test_numerical_correctness_vs_xla(self):
        """Test numerical correctness against XLA fallback on a small case."""
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 3)

        batch_size = 1
        seq_len = 256
        num_heads = 4
        head_dim = 64

        query = jax.random.normal(keys[0], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        key_array = jax.random.normal(keys[1], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        value = jax.random.normal(keys[2], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)

        out_tpu = blocksparse_attention(query=query, key=key_array, value=value, causal=True, sliding_window=64)
        out_xla = blocksparse_attention_xla(query=query, key=key_array, value=value, causal=True, sliding_window=64)

        assert out_tpu.shape == out_xla.shape
        assert jnp.allclose(out_tpu, out_xla, rtol=0.0, atol=0.15)


class TestSlidingWindowAttention:
    """Test sliding window attention variants."""

    def test_symmetric_sliding_window(self):
        """Test symmetric sliding window attention."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 1024
        num_heads = 8
        head_dim = 64
        window_size = 256

        query = jax.random.normal(keys[0], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        key_array = jax.random.normal(keys[1], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        value = jax.random.normal(keys[2], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)

        output = blocksparse_attention(
            query=query,
            key=key_array,
            value=value,
            sliding_window=window_size,
            causal=False,
        )

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert jnp.all(jnp.isfinite(output))

    def test_asymmetric_sliding_window(self):
        """Test asymmetric sliding window attention."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 1024
        num_heads = 8
        head_dim = 64
        left_window = 128
        right_window = 64

        query = jax.random.normal(keys[0], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        key_array = jax.random.normal(keys[1], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        value = jax.random.normal(keys[2], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)

        output = blocksparse_attention(
            query=query,
            key=key_array,
            value=value,
            sliding_window=(left_window, right_window),
            causal=False,
        )

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert jnp.all(jnp.isfinite(output))

    def test_causal_sliding_window(self):
        """Test causal sliding window attention (like Mistral)."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 1024
        num_heads = 8
        head_dim = 64
        window_size = 256

        query = jax.random.normal(keys[0], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        key_array = jax.random.normal(keys[1], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        value = jax.random.normal(keys[2], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)

        output_causal_sliding = blocksparse_attention(
            query=query,
            key=key_array,
            value=value,
            sliding_window=window_size,
            causal=True,
        )

        output_sliding = blocksparse_attention(
            query=query,
            key=key_array,
            value=value,
            sliding_window=window_size,
            causal=False,
        )

        assert output_causal_sliding.shape == (batch_size, num_heads, seq_len, head_dim)
        assert output_sliding.shape == (batch_size, num_heads, seq_len, head_dim)

        diff = float(jnp.mean(jnp.abs(output_causal_sliding - output_sliding)))
        assert diff > 1e-6, "Causal masking should affect sliding window output"


class TestChunkedCausalAttention:
    """Test chunked causal attention (Llama4 style)."""

    def test_chunked_causal_mask(self):
        """Test chunked causal attention."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 1024
        num_heads = 8
        head_dim = 64
        chunk_size = 128

        query = jax.random.normal(keys[0], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        key_array = jax.random.normal(keys[1], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        value = jax.random.normal(keys[2], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)

        output = blocksparse_attention(
            query=query,
            key=key_array,
            value=value,
            chunk_size=chunk_size,
        )

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert jnp.all(jnp.isfinite(output))

    def test_chunked_vs_causal_difference(self):
        """Test that chunked causal differs from standard causal."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 512
        num_heads = 4
        head_dim = 64
        chunk_size = 128

        query = jax.random.normal(keys[0], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        key_array = jax.random.normal(keys[1], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        value = jax.random.normal(keys[2], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)

        output_chunked = blocksparse_attention(
            query=query,
            key=key_array,
            value=value,
            chunk_size=chunk_size,
        )

        output_causal = blocksparse_attention(
            query=query,
            key=key_array,
            value=value,
            causal=True,
        )

        diff = float(jnp.mean(jnp.abs(output_chunked - output_causal)))
        assert diff > 1e-6, "Chunked causal should differ from standard causal"


class TestSegmentIds:
    """Test segment ID functionality for packed sequences."""

    def test_segment_ids_with_sliding_window(self):
        """Test segment IDs with sliding window."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 1024
        num_heads = 8
        head_dim = 64

        query = jax.random.normal(keys[0], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        key_array = jax.random.normal(keys[1], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        value = jax.random.normal(keys[2], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)

        q_segment_ids = jnp.concatenate(
            [
                jnp.zeros((batch_size, seq_len // 2), dtype=jnp.int32),
                jnp.ones((batch_size, seq_len // 2), dtype=jnp.int32),
            ],
            axis=-1,
        )
        kv_segment_ids = q_segment_ids

        output = blocksparse_attention(
            query=query,
            key=key_array,
            value=value,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            sliding_window=256,
            causal=True,
        )

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert jnp.all(jnp.isfinite(output))


class TestCustomMaskBuilder:
    """Test custom mask builder functionality."""

    def test_custom_mask_builder(self):
        """Test custom mask builder with different masks per head."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 512
        num_heads = 4
        head_dim = 64

        query = jax.random.normal(keys[0], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        key_array = jax.random.normal(keys[1], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        value = jax.random.normal(keys[2], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)

        def custom_mask_builder(q_len: int, kv_len: int, num_heads: int, head_idx: int, num_reps: int):
            """Half heads causal, half heads sliding window."""

            if head_idx < num_heads // 2:
                return CausalMask((q_len, kv_len))
            else:
                return LocalMask(shape=(q_len, kv_len), window_size=(128, 128), offset=0)

        output = blocksparse_attention(
            query=query,
            key=key_array,
            value=value,
            mask_builder=custom_mask_builder,
        )

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert jnp.all(jnp.isfinite(output))


class TestPerformanceTuning:
    """Test performance tuning with different chunk sizes."""

    def test_different_chunk_sizes(self):
        """Test different query and key chunk sizes."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 1024
        num_heads = 8
        head_dim = 64

        query = jax.random.normal(keys[0], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        key_array = jax.random.normal(keys[1], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        value = jax.random.normal(keys[2], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)

        for q_blocksize, kv_blocksize in [(128, 128), (256, 256), (512, 512)]:
            output = blocksparse_attention(
                query=query,
                key=key_array,
                value=value,
                fwd_params=FwdParams(q_blocksize=q_blocksize, kv_blocksize=kv_blocksize, num_stages=2, num_warps=4),
                sliding_window=512,
                causal=True,
            )
            assert output.shape == (batch_size, num_heads, seq_len, head_dim)
            assert jnp.all(jnp.isfinite(output))


class TestSoftCapAndScale:
    """Test soft capping and softmax softmax_scale functionality."""

    def test_soft_cap(self):
        """Test soft capping for attention logits (Gemma2 style)."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 512
        num_heads = 4
        head_dim = 64

        query = jax.random.normal(keys[0], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        key_array = jax.random.normal(keys[1], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        value = jax.random.normal(keys[2], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)

        output_no_cap = blocksparse_attention(
            query=query,
            key=key_array,
            value=value,
            logits_soft_cap=None,
            causal=True,
        )

        output_with_cap = blocksparse_attention(
            query=query,
            key=key_array,
            value=value,
            logits_soft_cap=50.0,
            causal=True,
        )

        assert output_no_cap.shape == (batch_size, num_heads, seq_len, head_dim)
        assert output_with_cap.shape == (batch_size, num_heads, seq_len, head_dim)

        diff = float(jnp.mean(jnp.abs(output_no_cap - output_with_cap)))
        assert diff > 1e-6, "Soft capping should affect output"

    def test_softmax_scale(self):
        """Test custom softmax softmax_scale."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 512
        num_heads = 4
        head_dim = 64

        query = jax.random.normal(keys[0], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        key_array = jax.random.normal(keys[1], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        value = jax.random.normal(keys[2], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)

        output_default = blocksparse_attention(
            query=query,
            key=key_array,
            value=value,
            softmax_scale=None,
            causal=True,
        )

        output_custom = blocksparse_attention(
            query=query,
            key=key_array,
            value=value,
            softmax_scale=0.5,
            causal=True,
        )

        assert output_default.shape == (batch_size, num_heads, seq_len, head_dim)
        assert output_custom.shape == (batch_size, num_heads, seq_len, head_dim)

        diff = float(jnp.mean(jnp.abs(output_default - output_custom)))
        assert diff > 1e-6, "Different softmax scales should affect output"

    def test_softmax_aux(self):
        """Test auxiliary softmax values."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)

        batch_size = 2
        seq_len = 512
        num_heads = 4
        head_dim = 64

        query = jax.random.normal(keys[0], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        key_array = jax.random.normal(keys[1], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        value = jax.random.normal(keys[2], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)

        num_sinks = 4
        softmax_aux = jax.random.normal(keys[3], (num_sinks,), dtype=jnp.float32)

        output = blocksparse_attention(
            query=query,
            key=key_array,
            value=value,
            softmax_aux=softmax_aux,
            causal=True,
        )

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert jnp.all(jnp.isfinite(output))

    def test_combined_soft_cap_and_scale(self):
        """Test combining soft cap with custom softmax softmax_scale."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 512
        num_heads = 4
        head_dim = 64

        query = jax.random.normal(keys[0], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        key_array = jax.random.normal(keys[1], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        value = jax.random.normal(keys[2], (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)

        output = blocksparse_attention(
            query=query,
            key=key_array,
            value=value,
            logits_soft_cap=30.0,
            softmax_scale=0.2,
            sliding_window=256,
            causal=True,
        )

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert jnp.all(jnp.isfinite(output))


class TestMaskHelpers:
    """Test mask helper functions."""

    def test_make_causal_mask(self):
        """Test make_causal_mask helper."""
        seq_len = 16
        mask = make_causal_mask((seq_len, seq_len))

        assert mask.shape == (seq_len, seq_len)
        assert mask[5, 5]
        assert not mask[5, 6]
        assert mask[5, 4]

    def test_make_local_attention_mask(self):
        """Test make_local_attention_mask helper."""
        seq_len = 16
        window_size = (4, 4)

        mask = make_local_attention_mask(shape=(seq_len, seq_len), window_size=window_size)

        assert mask.shape == (seq_len, seq_len)

        assert mask[8, 8]
        assert mask[8, 7]
        assert mask[8, 9]
        assert not mask[8, 3]
        assert not mask[8, 13]

    def test_make_chunk_attention_mask(self):
        """Test make_chunk_attention_mask helper."""
        seq_len = 16
        chunk_size = 4

        mask = make_chunk_attention_mask(shape=(seq_len, seq_len), chunk_size=chunk_size)

        assert mask.shape == (seq_len, seq_len)

        assert mask[6, 4]
        assert mask[6, 6]
        assert not mask[6, 7]

        assert not mask[6, 3]
        assert not mask[6, 8]


class TestMaskClasses:
    """Test individual mask classes."""

    def test_causal_mask_class(self):
        """Test CausalMask class."""
        seq_len = 16
        mask = CausalMask((seq_len, seq_len))

        assert mask.shape == (seq_len, seq_len)
        mask_array = mask[0:seq_len, 0:seq_len]

        for i in range(seq_len):
            for j in range(seq_len):
                if i >= j:
                    assert mask_array[i, j]
                else:
                    assert not mask_array[i, j]

    def test_local_mask_class(self):
        """Test LocalMask class."""
        seq_len = 16
        window_size = (4, 4)

        mask = LocalMask(shape=(seq_len, seq_len), window_size=window_size, offset=0)

        assert mask.shape == (seq_len, seq_len)
        mask_array = mask[0:seq_len, 0:seq_len]

        for i in range(seq_len):
            for j in range(seq_len):
                if abs(i - j) <= 4:
                    assert mask_array[i, j]
                else:
                    assert not mask_array[i, j]

    def test_chunked_causal_mask_class(self):
        """Test ChunkedCausalMask class."""
        seq_len = 16
        chunk_size = 4

        mask = ChunkedCausalMask(shape=(seq_len, seq_len), chunk_size=chunk_size)

        assert mask.shape == (seq_len, seq_len)
        mask_array = mask[0:seq_len, 0:seq_len]

        for i in range(seq_len):
            for j in range(seq_len):
                same_chunk = (i // chunk_size) == (j // chunk_size)
                is_causal = i >= j
                expected = same_chunk and is_causal
                assert mask_array[i, j] == expected

    def test_mask_combination(self):
        """Test combining masks with AND and OR operations."""
        seq_len = 16

        causal = CausalMask((seq_len, seq_len))
        local = LocalMask(shape=(seq_len, seq_len), window_size=(4, 4), offset=0)

        and_mask = causal & local
        and_array = and_mask[0:seq_len, 0:seq_len]

        assert and_array[8, 7]
        assert not and_array[8, 10]
        assert not and_array[8, 1]

        or_mask = causal | local
        or_array = or_mask[0:seq_len, 0:seq_len]

        assert or_array[8, 7]
        assert or_array[8, 10]
        assert or_array[8, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
