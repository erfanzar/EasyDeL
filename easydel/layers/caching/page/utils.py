import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from easydel.utils.compiling_utils import ejit


def cdiv(a, v):
    return (a + v - 1) // v


def _kv_cache_update_kernel(
    slice_indices_ref,
    new_kv_tokens_hbm_ref,
    kv_cache_pages_hbm_ref,
    _,
    vmem_scratch_buffer,
    dma_semaphore,
):
    pending_async_copies = []
    current_page_id = pl.program_id(0)
    slices_per_processing_page = vmem_scratch_buffer.shape[0]

    # First phase: Copy from new KV tokens to scratch buffer
    for slice_idx in range(slices_per_processing_page):
        global_slice_idx = slice_idx + current_page_id * slices_per_processing_page
        new_kv_start_pos = slice_indices_ref[1, global_slice_idx]
        slice_length = slice_indices_ref[2, global_slice_idx]

        copy_operation = pltpu.make_async_copy(
            new_kv_tokens_hbm_ref.at[pl.ds(new_kv_start_pos, slice_length), ...],
            vmem_scratch_buffer.at[slice_idx, pl.ds(0, slice_length), ...],
            dma_semaphore,
        )
        copy_operation.start()
        pending_async_copies.append(copy_operation)

    # Wait for all copies to complete
    for copy_operation in pending_async_copies:
        copy_operation.wait()

    # Second phase: Copy from scratch buffer to KV cache
    pending_async_copies.clear()
    for slice_idx in range(slices_per_processing_page):
        global_slice_idx = slice_idx + current_page_id * slices_per_processing_page
        kv_cache_start_pos = slice_indices_ref[0, global_slice_idx]
        slice_length = slice_indices_ref[2, global_slice_idx]

        copy_operation = pltpu.make_async_copy(
            vmem_scratch_buffer.at[slice_idx, pl.ds(0, slice_length), ...],
            kv_cache_pages_hbm_ref.at[pl.ds(kv_cache_start_pos, slice_length), ...],
            dma_semaphore,
        )
        copy_operation.start()
        pending_async_copies.append(copy_operation)

    # Wait for all final copies to complete
    for copy_operation in pending_async_copies:
        copy_operation.wait()


@ejit(static_argnames=["page_size", "slices_per_processing_page"])
def kv_cache_update(
    new_kv_tokens: jax.Array,  # [total_num_token, num_combined_kv_heads, head_dim]
    slice_indices: jax.Array,  # [3, slices], list of (kv_cache_start, new_kv_start, slice_len)
    kv_cache_pages: jax.Array,  # [total_num_pages * page_size, num_combined_kv_heads, head_dim]
    total_update_slices: jax.Array,  # [1]
    *,
    page_size: int = 32,
    slices_per_processing_page: int = 8,
):
    assert (
        slice_indices.shape[1] % slices_per_processing_page == 0
    ), f"{slices_per_processing_page=}, {slice_indices.shape[1]=}"
    _, num_kv_heads, head_dimension = new_kv_tokens.shape
    assert kv_cache_pages.shape[1] == num_kv_heads
    assert kv_cache_pages.shape[2] == head_dimension
    assert head_dimension % 128 == 0

    prefetch_scalars = [slice_indices]
    vmem_scratch_buffer = pltpu.VMEM(
        (slices_per_processing_page, page_size, num_kv_heads, head_dimension), new_kv_tokens.dtype
    )

    pallas_kernel = pl.pallas_call(
        _kv_cache_update_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(prefetch_scalars),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
                pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
            ],
            out_specs=[pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY)],
            grid=(cdiv(total_update_slices[0], slices_per_processing_page),),
            scratch_shapes=[vmem_scratch_buffer, pltpu.SemaphoreType.DMA],
        ),
        out_shape=[jax.ShapeDtypeStruct(kv_cache_pages.shape, dtype=kv_cache_pages.dtype)],
        input_output_aliases={len(prefetch_scalars) + 1: 0},
    )

    return pallas_kernel(*prefetch_scalars, new_kv_tokens, kv_cache_pages)[0]


@ejit(static_argnames=["page_size"])
def kv_cache_update_jax(
    new_kv_tokens: jax.Array,  # [total_num_token, num_kv_heads, head_dim]
    slice_indices: jax.Array,  # [3, num_slices] - (cache_start, new_kv_start, length)
    kv_cache_pages: jax.Array,  # [total_pages * page_size, num_kv_heads, head_dim]
    total_update_slices: jax.Array,  # [1] - number of valid slices
    *,
    page_size: int = 32,
) -> jax.Array:
    """
    Pure JAX implementation of KV cache update using scatter operations.

    Args:
        new_kv_tokens: New key/value tokens to add to cache
        slice_indices: [3, N] array with (cache_pos, new_kv_pos, length) for each slice
        kv_cache_pages: Existing KV cache to update
        total_update_slices: Number of valid slices to process
        page_size: page size (must be static)

    Returns:
        Updated KV cache
    """
    num_valid_slices = total_update_slices[0]
    padded_new_kv = jnp.pad(new_kv_tokens, [(0, page_size), (0, 0), (0, 0)], mode="constant")

    def update_single_slice(cache, slice_idx):
        """Update cache with a single slice."""
        cache_start_pos = slice_indices[0, slice_idx]
        new_kv_start_pos = slice_indices[1, slice_idx]
        actual_length = slice_indices[2, slice_idx]
        new_slice = jax.lax.dynamic_slice(
            padded_new_kv,
            (new_kv_start_pos, 0, 0),
            (page_size, padded_new_kv.shape[1], padded_new_kv.shape[2]),
        )
        current_cache_slice = jax.lax.dynamic_slice(
            cache,
            (cache_start_pos, 0, 0),
            (page_size, cache.shape[1], cache.shape[2]),
        )
        mask = jnp.arange(page_size)[:, None, None] < actual_length
        updated_slice = jnp.where(mask, new_slice, current_cache_slice)
        updated_cache = jax.lax.dynamic_update_slice(cache, updated_slice, (cache_start_pos, 0, 0))

        return updated_cache

    def scan_fn(cache, slice_idx):
        updated_cache = jax.lax.cond(
            slice_idx < num_valid_slices,
            lambda c: update_single_slice(c, slice_idx),
            lambda c: c,
            cache,
        )
        return updated_cache, None

    return jax.lax.scan(scan_fn, kv_cache_pages, jnp.arange(slice_indices.shape[1]))[0]
