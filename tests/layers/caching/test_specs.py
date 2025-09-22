"""Test cache specification classes."""

import jax.numpy as jnp
import pytest
from triton import cdiv

from easydel.layers.caching._specs import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)


class TestFullAttentionSpec:
    """Test FullAttentionSpec."""

    def test_creation(self):
        """Test spec creation."""
        spec = FullAttentionSpec(
            page_size=128,
            num_kv_heads=8,
            head_size=64,
            dtype=jnp.bfloat16,
            use_mla=False,
        )

        assert spec.page_size == 128
        assert spec.num_kv_heads == 8
        assert spec.head_size == 64
        assert spec.dtype == jnp.bfloat16
        assert spec.use_mla is False
        assert spec.sliding_window is None
        assert spec.attention_chunk_size is None

    def test_type_id(self):
        """Test type ID generation."""
        spec = FullAttentionSpec(
            page_size=128,
            num_kv_heads=8,
            head_size=64,
            dtype=jnp.bfloat16,
            use_mla=False,
        )

        type_id = spec.type_id
        assert "full_attention" in type_id
        assert "128" in type_id  # page_size

    def test_page_size_bytes(self):
        """Test page size calculation in bytes."""
        # Without MLA (stores both keys and values)
        spec = FullAttentionSpec(
            page_size=128,
            num_kv_heads=8,
            head_size=64,
            dtype=jnp.bfloat16,
            use_mla=False,
        )
        # 2 * 128 * 8 * 64 * 2 bytes = 262144 bytes
        assert spec.page_size_bytes == 2 * 128 * 8 * 64 * 2

        # With MLA (stores combined representation)
        spec_mla = FullAttentionSpec(
            page_size=128,
            num_kv_heads=8,
            head_size=64,
            dtype=jnp.bfloat16,
            use_mla=True,
        )
        # 1 * 128 * 8 * 64 * 2 bytes = 131072 bytes
        assert spec_mla.page_size_bytes == 1 * 128 * 8 * 64 * 2

    def test_max_memory_usage_bytes(self):
        """Test maximum memory calculation."""
        spec = FullAttentionSpec(
            page_size=128,
            num_kv_heads=8,
            head_size=64,
            dtype=jnp.bfloat16,
            use_mla=False,
        )

        # For max_model_len=1024
        memory = spec.max_memory_usage_bytes(max_model_len=1024)
        num_pages = cdiv(1024, 128)  # 8 pages
        expected = num_pages * spec.page_size_bytes
        assert memory == expected

    def test_merge_specs(self):
        """Test merging multiple specs."""
        spec1 = FullAttentionSpec(
            page_size=128,
            num_kv_heads=8,
            head_size=64,
            dtype=jnp.bfloat16,
            use_mla=False,
            sliding_window=256,
        )

        spec2 = FullAttentionSpec(
            page_size=128,
            num_kv_heads=8,
            head_size=64,
            dtype=jnp.bfloat16,
            use_mla=False,
            sliding_window=256,
        )

        # Successful merge with same window sizes
        merged = FullAttentionSpec.merge([spec1, spec2])
        assert merged.sliding_window == 256

        # Failed merge with different window sizes
        spec3 = FullAttentionSpec(
            page_size=128,
            num_kv_heads=8,
            head_size=64,
            dtype=jnp.bfloat16,
            use_mla=False,
            sliding_window=512,
        )

        with pytest.raises(ValueError, match="same window size"):
            FullAttentionSpec.merge([spec1, spec3])

    def test_merge_with_chunk_size(self):
        """Test merging with attention chunk size."""
        spec1 = FullAttentionSpec(
            page_size=128,
            num_kv_heads=8,
            head_size=64,
            dtype=jnp.bfloat16,
            use_mla=False,
            attention_chunk_size=256,
        )

        spec2 = FullAttentionSpec(
            page_size=128,
            num_kv_heads=8,
            head_size=64,
            dtype=jnp.bfloat16,
            use_mla=False,
            attention_chunk_size=256,
        )

        merged = FullAttentionSpec.merge([spec1, spec2])
        assert merged.attention_chunk_size == 256
        assert merged.sliding_window is None

    def test_invalid_merge_both_window_and_chunk(self):
        """Test that having both sliding window and chunk size fails."""
        spec1 = FullAttentionSpec(
            page_size=128,
            num_kv_heads=8,
            head_size=64,
            dtype=jnp.bfloat16,
            use_mla=False,
            sliding_window=256,
        )

        spec2 = FullAttentionSpec(
            page_size=128,
            num_kv_heads=8,
            head_size=64,
            dtype=jnp.bfloat16,
            use_mla=False,
            attention_chunk_size=256,
        )

        with pytest.raises(AssertionError, match="both sliding window.*chunked local"):
            FullAttentionSpec.merge([spec1, spec2])


class TestSlidingWindowSpec:
    """Test SlidingWindowSpec."""

    def test_creation(self):
        """Test sliding window spec creation."""
        spec = SlidingWindowSpec(
            page_size=64,
            num_kv_heads=4,
            head_size=32,
            dtype=jnp.float32,
            use_mla=False,
            sliding_window=512,
        )

        assert spec.sliding_window == 512
        assert spec.use_mla is False

    def test_mla_not_supported(self):
        """Test that MLA is not supported for sliding window."""
        with pytest.raises(AssertionError, match="MLA is not supported"):
            SlidingWindowSpec(
                page_size=64,
                num_kv_heads=4,
                head_size=32,
                dtype=jnp.float32,
                use_mla=True,
                sliding_window=512,
            )

    def test_type_id(self):
        """Test sliding window type ID."""
        spec = SlidingWindowSpec(
            page_size=64,
            num_kv_heads=4,
            head_size=32,
            dtype=jnp.float32,
            use_mla=False,
            sliding_window=512,
        )

        type_id = spec.type_id
        assert "sliding_window" in type_id
        assert "512" in type_id  # window size

    def test_max_memory_usage_bytes(self):
        """Test memory calculation for sliding window."""
        spec = SlidingWindowSpec(
            page_size=64,
            num_kv_heads=4,
            head_size=32,
            dtype=jnp.float32,
            use_mla=False,
            sliding_window=256,
        )

        # Memory bounded by window size plus batch
        memory = spec.max_memory_usage_bytes(max_model_len=1024, max_num_batched_tokens=32)

        # min(256 - 1 + 32, 1024) = 287 tokens
        num_tokens = min(256 - 1 + 32, 1024)
        # (cdiv(287, 64) + 1) pages = (5 + 1) = 6 pages
        num_pages = cdiv(num_tokens, 64) + 1
        expected = num_pages * spec.page_size_bytes
        assert memory == expected


class TestChunkedLocalAttentionSpec:
    """Test ChunkedLocalAttentionSpec."""

    def test_creation(self):
        """Test chunked attention spec creation."""
        spec = ChunkedLocalAttentionSpec(
            page_size=32,
            num_kv_heads=8,
            head_size=64,
            dtype=jnp.bfloat16,
            use_mla=False,
            attention_chunk_size=128,
        )

        assert spec.attention_chunk_size == 128

    def test_type_id(self):
        """Test chunked attention type ID."""
        spec = ChunkedLocalAttentionSpec(
            page_size=32,
            num_kv_heads=8,
            head_size=64,
            dtype=jnp.bfloat16,
            use_mla=False,
            attention_chunk_size=128,
        )

        type_id = spec.type_id
        assert "local_attention" in type_id
        assert "128" in type_id  # chunk size

    def test_max_memory_usage_bytes(self):
        """Test memory calculation for chunked attention."""
        spec = ChunkedLocalAttentionSpec(
            page_size=32,
            num_kv_heads=8,
            head_size=64,
            dtype=jnp.bfloat16,
            use_mla=False,
            attention_chunk_size=128,
        )

        memory = spec.max_memory_usage_bytes(max_model_len=1024, max_num_batched_tokens=64)

        # min(128 + 64, 1024) = 192 tokens
        num_tokens = min(128 + 64, 1024)
        num_pages = cdiv(num_tokens, 32)
        expected = num_pages * spec.page_size_bytes
        assert memory == expected


class TestMambaSpec:
    """Test MambaSpec."""

    def test_creation(self):
        """Test Mamba spec creation."""
        shapes = ((2, 4, 16), (2, 4, 32))
        spec = MambaSpec(
            page_size=1,
            shapes=shapes,
            dtype=jnp.float32,
            page_size_padded=None,
        )

        assert spec.shapes == shapes
        assert spec.dtype == jnp.float32

        # Check num_elements calculation
        expected_elements = 2 * 4 * 16 + 2 * 4 * 32
        assert spec.num_elements == expected_elements

    def test_type_id(self):
        """Test Mamba type ID."""
        shapes = ((2, 4, 16),)
        spec = MambaSpec(
            page_size=1,
            shapes=shapes,
            dtype=jnp.float32,
        )

        type_id = spec.type_id
        assert "mamba" in type_id
        assert str(shapes) in type_id

    def test_page_size_bytes(self):
        """Test Mamba page size calculation."""
        shapes = ((2, 4, 8),)  # 64 elements
        spec = MambaSpec(
            page_size=1,
            shapes=shapes,
            dtype=jnp.float32,  # 4 bytes per element
        )

        # 64 elements * 4 bytes = 256 bytes
        assert spec.page_size_bytes == 256

        # With padding
        spec_padded = MambaSpec(
            page_size=1,
            shapes=shapes,
            dtype=jnp.float32,
            page_size_padded=512,
        )
        assert spec_padded.page_size_bytes == 512

    def test_page_size_padding_too_small(self):
        """Test that padded size must be large enough."""
        shapes = ((2, 4, 8),)  # 64 elements
        spec = MambaSpec(
            page_size=1,
            shapes=shapes,
            dtype=jnp.float32,  # 4 bytes per element
            page_size_padded=100,  # Too small (needs 256)
        )

        with pytest.raises(AssertionError):
            _ = spec.page_size_bytes

    def test_max_memory_usage_bytes(self):
        """Test Mamba memory calculation."""
        shapes = ((2, 4, 8),)
        spec = MambaSpec(
            page_size=1,
            shapes=shapes,
            dtype=jnp.float32,
        )

        # Mamba has fixed size regardless of sequence length
        memory = spec.max_memory_usage_bytes(max_model_len=1024)
        assert memory == spec.page_size_bytes


class TestKVCacheSpecMerge:
    """Test KVCacheSpec merge functionality."""

    def test_merge_compatible_specs(self):
        """Test merging compatible specs."""
        spec1 = FullAttentionSpec(
            page_size=128,
            num_kv_heads=8,
            head_size=64,
            dtype=jnp.bfloat16,
            use_mla=False,
        )

        spec2 = FullAttentionSpec(
            page_size=128,
            num_kv_heads=8,
            head_size=64,
            dtype=jnp.bfloat16,
            use_mla=False,
        )

        merged = KVCacheSpec.merge([spec1, spec2])
        assert merged.page_size == spec1.page_size
        assert merged.num_kv_heads == spec1.num_kv_heads

    def test_merge_incompatible_specs(self):
        """Test merging incompatible specs fails."""
        spec1 = FullAttentionSpec(
            page_size=128,
            num_kv_heads=8,
            head_size=64,
            dtype=jnp.bfloat16,
            use_mla=False,
        )

        spec2 = SlidingWindowSpec(
            page_size=128,
            num_kv_heads=8,
            head_size=64,
            dtype=jnp.bfloat16,
            use_mla=False,
            sliding_window=256,
        )

        # Different type_ids should cause assertion error
        with pytest.raises(AssertionError, match="same type_id"):
            KVCacheSpec.merge([spec1, spec2])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
