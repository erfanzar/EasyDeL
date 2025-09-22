"""Test cache utility functions and classes."""

from enum import Enum

import pytest

from easydel.layers.caching._utils import AttnMaskDetail


class TestAttnMaskType(Enum):
    """Test enum for attention mask types."""

    FULL = "full"
    SLIDING = "sliding"
    CHUNKED = "chunked"
    SPARSE = "sparse"


class TestAttnMaskDetail:
    """Test AttnMaskDetail utility class."""

    def test_basic_creation(self):
        """Test basic mask detail creation."""
        mask = AttnMaskDetail(
            mask_type=TestAttnMaskType.FULL,
            size=128,
        )

        assert mask.mask_type == TestAttnMaskType.FULL
        assert mask.size == 128
        assert mask.offset is None
        assert mask.chunks is None
        assert mask.bricks is None

    def test_sliding_window_mask(self):
        """Test sliding window mask configuration."""
        mask = AttnMaskDetail(
            mask_type=TestAttnMaskType.SLIDING,
            size=256,
            offset=10,
        )

        assert mask.mask_type == TestAttnMaskType.SLIDING
        assert mask.size == 256
        assert mask.offset == 10

    def test_chunked_mask(self):
        """Test chunked attention mask configuration."""
        mask = AttnMaskDetail(
            mask_type=TestAttnMaskType.CHUNKED,
            size=64,
            chunks=8,
            offset=0,
        )

        assert mask.mask_type == TestAttnMaskType.CHUNKED
        assert mask.size == 64
        assert mask.chunks == 8
        assert mask.offset == 0

    def test_sparse_mask_with_bricks(self):
        """Test sparse mask with brick pattern."""
        mask = AttnMaskDetail(
            mask_type=TestAttnMaskType.SPARSE,
            size=512,
            bricks=16,
            offset=32,
        )

        assert mask.mask_type == TestAttnMaskType.SPARSE
        assert mask.size == 512
        assert mask.bricks == 16
        assert mask.offset == 32

    def test_mask_as_pytree(self):
        """Test that AttnMaskDetail works as JAX pytree."""
        import jax.tree_util as tree

        mask = AttnMaskDetail(
            mask_type=TestAttnMaskType.FULL,
            size=128,
            offset=5,
            chunks=4,
        )

        # Should be able to tree_flatten and tree_unflatten
        leaves, treedef = tree.tree_flatten(mask)
        reconstructed = tree.tree_unflatten(treedef, leaves)

        assert reconstructed.mask_type == mask.mask_type
        assert reconstructed.size == mask.size
        assert reconstructed.offset == mask.offset
        assert reconstructed.chunks == mask.chunks

    def test_mask_in_tree_map(self):
        """Test AttnMaskDetail in jax.tree.map operations."""
        import jax

        mask1 = AttnMaskDetail(
            mask_type=TestAttnMaskType.FULL,
            size=128,
            offset=10,
        )

        _mask2 = AttnMaskDetail(
            mask_type=TestAttnMaskType.SLIDING,
            size=256,
            offset=20,
        )

        # Test tree_map with a function that modifies numeric fields
        def double_numeric(x):
            if isinstance(x, int):
                return x * 2
            return x

        doubled_mask1 = jax.tree.map(double_numeric, mask1)
        assert doubled_mask1.size == 256  # 128 * 2
        assert doubled_mask1.offset == 20  # 10 * 2

    def test_mask_combinations(self):
        """Test various mask parameter combinations."""
        # Full mask with minimal params
        full_mask = AttnMaskDetail(
            mask_type=TestAttnMaskType.FULL,
            size=1024,
        )
        assert full_mask.mask_type == TestAttnMaskType.FULL
        assert full_mask.size == 1024

        # Sliding with all params
        sliding_mask = AttnMaskDetail(
            mask_type=TestAttnMaskType.SLIDING,
            size=512,
            offset=16,
            chunks=None,  # Not used for sliding
            bricks=None,  # Not used for sliding
        )
        assert sliding_mask.mask_type == TestAttnMaskType.SLIDING
        assert sliding_mask.size == 512
        assert sliding_mask.offset == 16

        # Chunked with chunks
        chunked_mask = AttnMaskDetail(
            mask_type=TestAttnMaskType.CHUNKED,
            size=128,
            offset=0,
            chunks=16,
        )
        assert chunked_mask.mask_type == TestAttnMaskType.CHUNKED
        assert chunked_mask.chunks == 16

    def test_mask_equality(self):
        """Test mask detail equality."""
        mask1 = AttnMaskDetail(
            mask_type=TestAttnMaskType.FULL,
            size=128,
            offset=10,
        )

        mask2 = AttnMaskDetail(
            mask_type=TestAttnMaskType.FULL,
            size=128,
            offset=10,
        )

        mask3 = AttnMaskDetail(
            mask_type=TestAttnMaskType.FULL,
            size=256,  # Different size
            offset=10,
        )

        # Python equality
        assert mask1 == mask2
        assert mask1 != mask3

    def test_mask_with_none_values(self):
        """Test mask with None values for optional fields."""
        mask = AttnMaskDetail(
            mask_type=TestAttnMaskType.FULL,
            size=512,
            offset=None,
            chunks=None,
            bricks=None,
        )

        assert mask.offset is None
        assert mask.chunks is None
        assert mask.bricks is None

    def test_mask_tree_leaves(self):
        """Test the pytree leaves of AttnMaskDetail."""
        import jax.tree_util as tree

        mask = AttnMaskDetail(
            mask_type=TestAttnMaskType.CHUNKED,
            size=256,
            offset=16,
            chunks=8,
            bricks=4,
        )

        leaves, _ = tree.tree_flatten(mask)

        # Check that all fields are in leaves
        assert TestAttnMaskType.CHUNKED in leaves
        assert 256 in leaves
        assert 16 in leaves
        assert 8 in leaves
        assert 4 in leaves


class TestCacheUtilityPatterns:
    """Test common utility patterns used across cache implementations."""

    def test_mask_detail_for_different_attention_types(self):
        """Test mask configurations for different attention patterns."""
        # Standard causal mask
        causal_mask = AttnMaskDetail(
            mask_type=TestAttnMaskType.FULL,
            size=2048,
        )
        assert causal_mask.mask_type == TestAttnMaskType.FULL

        # Sliding window for local attention
        local_mask = AttnMaskDetail(
            mask_type=TestAttnMaskType.SLIDING,
            size=256,
            offset=0,
        )
        assert local_mask.size == 256

        # Chunked for block-sparse attention
        chunked_mask = AttnMaskDetail(
            mask_type=TestAttnMaskType.CHUNKED,
            size=64,
            chunks=32,
        )
        assert chunked_mask.chunks == 32

        # Sparse with brick pattern
        sparse_mask = AttnMaskDetail(
            mask_type=TestAttnMaskType.SPARSE,
            size=1024,
            bricks=64,
            offset=128,
        )
        assert sparse_mask.bricks == 64

    def test_mask_detail_serialization(self):
        """Test that mask detail can be serialized/deserialized."""
        import pickle

        original_mask = AttnMaskDetail(
            mask_type=TestAttnMaskType.SLIDING,
            size=512,
            offset=32,
            chunks=8,
            bricks=None,
        )

        # Serialize
        serialized = pickle.dumps(original_mask)

        # Deserialize
        restored_mask = pickle.loads(serialized)

        assert restored_mask.mask_type == original_mask.mask_type
        assert restored_mask.size == original_mask.size
        assert restored_mask.offset == original_mask.offset
        assert restored_mask.chunks == original_mask.chunks
        assert restored_mask.bricks == original_mask.bricks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
