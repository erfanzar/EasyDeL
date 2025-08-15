"""Test script to verify KV cache write functionality."""

import jax
import jax.numpy as jnp
from eformer.escale import PartitionAxis, PartitionManager

from easydel.layers.caching.page import PagesCache, PagesCacheMetaData, PagesMetadata


def test_kv_cache_write():
    """Test that KV cache writes are working correctly."""

    # Setup
    mesh = jax.sharding.Mesh(jax.devices(), ("tp",))
    partition_manager = PartitionManager(PartitionAxis(kv_head_axis="tp"))

    # Create metadata
    metadata = PagesCacheMetaData.create(
        mesh=mesh,
        partition_manager=partition_manager,
        kvdtype=jnp.float32,
        num_hidden_layers=1,
        num_kv_heads=8,
        max_model_length=256,
        kv_head_dim_size=128,
        hbm_utilization=0.1,
        page_size=16,
    )

    print("Metadata created:")
    print(f"  num_pages: {metadata.num_pages}")
    print(f"  page_size: {metadata.page_size}")
    print(f"  max_num_pages_per_req: {metadata.max_num_pages_per_req}")

    # Initialize cache
    cache = PagesCache.init_cache(
        mesh=mesh,
        metadata=metadata,
        partition_manager=partition_manager,
    )

    print("\nCache initialized:")
    print(f"  KV pages shape: {cache.views[0].kv_pages.shape}")

    # Create test data
    batch_size = 1
    seq_len = 3
    num_kv_heads = 8
    head_dim = 128

    # Generate test keys and values
    key = jnp.ones((batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)
    value = jnp.ones((batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float32) * 2

    # Create cache metadata for writing
    # slot_mapping: [kv_cache_start_indices, new_kv_start_indices, slice_lens]
    slot_mapping = jnp.array(
        [
            [16, -1, -1, -1],  # kv_cache_start_indices (page 1 * page_size 16)
            [0, -1, -1, -1],  # new_kv_start_indices
            [3, -1, -1, -1],  # slice_lens
        ],
        dtype=jnp.int32,
    )

    cache_metadata = PagesMetadata(
        pages_tables=jnp.array([[1, -1, -1, -1]], dtype=jnp.int32),  # Use page 1
        context_lens=jnp.array([3], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 3], dtype=jnp.int32),
        num_seqs=jnp.array([1], dtype=jnp.int32),
        slot_mapping=slot_mapping,
        num_kv_update_slices=jnp.array([1], dtype=jnp.int32),
        num_slices_per_kv_cache_update_page=metadata.num_slices_per_kv_cache_update_page,
        page_size=metadata.page_size,
    )

    print("\nCache metadata created:")
    print(f"  slot_mapping shape: {cache_metadata.slot_mapping.shape}")
    print(f"  slot_mapping:\n{cache_metadata.slot_mapping}")

    # Get initial cache state
    initial_page = cache.views[0].kv_pages[1, :3, :, :].copy()
    print("\nInitial cache page 1 (first 3 positions):")
    print(f"  Shape: {initial_page.shape}")
    print(f"  All zeros: {jnp.allclose(initial_page, 0)}")

    # Write to cache (must be done within mesh context)
    with mesh:
        print("\nBefore concatenate_to_cache:")
        print(f"  Key input shape: {key.shape}, mean: {jnp.mean(key):.4f}")
        print(f"  Value input shape: {value.shape}, mean: {jnp.mean(value):.4f}")
        cache.views[0].concatenate_to_cache(key, value, cache_metadata)

    # Check if cache was updated
    updated_page = cache.views[0].kv_pages[1, :3, :, :]
    print("\nUpdated cache page 1 (first 3 positions):")
    print(f"  Shape: {updated_page.shape}")
    print(f"  All zeros: {jnp.allclose(updated_page, 0)}")

    # Extract key and value parts (interleaved in kv_pages)
    key_cache = updated_page[:, 0::2, :]  # Even indices are keys
    value_cache = updated_page[:, 1::2, :]  # Odd indices are values

    print("\nExtracted from cache:")
    print(f"  Key cache shape: {key_cache.shape}")
    print(f"  Value cache shape: {value_cache.shape}")
    print(f"  Key cache mean: {jnp.mean(key_cache):.4f} (expected ~1.0)")
    print(f"  Value cache mean: {jnp.mean(value_cache):.4f} (expected ~2.0)")

    # Verify the write worked
    key_correct = jnp.allclose(key_cache, 1.0, atol=1e-5)
    value_correct = jnp.allclose(value_cache, 2.0, atol=1e-5)

    if key_correct and value_correct:
        print("\n✓ KV cache write is working correctly!")
    else:
        print("\n✗ KV cache write verification failed!")
        print(f"  Key correct: {key_correct}")
        print(f"  Value correct: {value_correct}")


if __name__ == "__main__":
    test_kv_cache_write()
