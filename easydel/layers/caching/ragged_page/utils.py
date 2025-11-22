"""Utility functions for paged attention cache operations.

This module provides low-level utilities for efficient paged KV cache
updates, including both TPU-optimized kernel implementations and
pure JAX fallbacks for other backends.

The paged attention mechanism divides the KV cache into fixed-size
pages, enabling more efficient memory management and reducing
fragmentation in long-context scenarios.

Key Functions:
    - cdiv: Ceiling division utility
    - kv_cache_update: TPU-optimized paged cache update
    - kv_cache_update_jax: Pure JAX implementation for compatibility
    - _kv_cache_update_kernel: Low-level TPU kernel

Optimizations:
    - Asynchronous DMA transfers on TPU
    - Vectorized memory operations
    - Efficient page-based updates
    - Minimal memory copies
"""

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Int

from easydel.utils.compiling_utils import ejit


def cdiv(a: int, v: int) -> int:
    """Ceiling division: divide a by v and round up.

    Calculates the ceiling of a/v using integer arithmetic,
    avoiding floating point operations for efficiency.

    Args:
        a (int): Numerator (e.g., total items)
        v (int): Denominator (e.g., items per page)

    Returns:
        int: The ceiling of a/v

    Example:
        >>> cdiv(10, 3)  # 10 items, 3 per page
        4  # Need 4 pages
    """
    return (a + v - 1) // v


def _kv_cache_update_kernel(
    slice_indices_ref,  # Pallas reference type
    new_kv_tokens_hbm_ref,  # Pallas reference type
    kv_cache_pages_hbm_ref,  # Pallas reference type
    _,  # Placeholder
    vmem_scratch_buffer,  # VMEM buffer
    dma_semaphore,  # Semaphore
) -> None:
    """Low-level TPU kernel for paged KV cache updates.

    Implements a two-phase DMA transfer strategy:
    1. Copy new KV tokens from HBM to VMEM scratch buffer
    2. Copy from scratch buffer to final cache pages

    This approach maximizes TPU memory bandwidth utilization
    by overlapping computation with asynchronous DMA transfers.

    Args:
        slice_indices_ref: Reference to slice mapping information.
            Shape: [3, num_slices] containing (cache_pos, new_kv_pos, length)
        new_kv_tokens_hbm_ref: Reference to new KV tokens in HBM.
        kv_cache_pages_hbm_ref: Reference to cache pages in HBM.
        _: Unused placeholder argument.
        vmem_scratch_buffer: VMEM scratch space for staging transfers.
        dma_semaphore: Semaphore for DMA synchronization.

    Note:
        This kernel is called by Pallas and executes on TPU.
        Each invocation processes one 'processing page' of slices.
    """
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
    new_kv_tokens: Float[Array, "total_tokens num_combined_kv_heads head_dim"],
    slice_indices: Int[Array, "3 num_slices"],
    kv_cache_pages: Float[Array, "total_cache_positions num_combined_kv_heads head_dim"],
    total_update_slices: Int[Array, ""],
    *,
    page_size: int = 32,
    slices_per_processing_page: int = 8,
) -> Float[Array, "total_cache_positions num_combined_kv_heads head_dim"]:
    """TPU-optimized paged KV cache update using Pallas kernels.

    Efficiently updates the KV cache with new tokens using hardware-accelerated
    DMA transfers and vectorized operations. The update is performed in pages
    to minimize memory fragmentation and improve cache locality.

    This function:
    1. Validates input dimensions and alignment
    2. Configures VMEM scratch buffers for staging
    3. Launches parallel Pallas kernels for updates
    4. Returns updated cache pages

    Args:
        new_kv_tokens (jax.Array): New key-value tokens to insert.
            Shape: [total_tokens, num_combined_kv_heads, head_dim]
            where num_combined_kv_heads = 2 * num_kv_heads (keys + values)
        slice_indices (jax.Array): Mapping of update operations.
            Shape: [3, num_slices] where each column contains:
            - Row 0: Starting position in cache
            - Row 1: Starting position in new_kv_tokens
            - Row 2: Length of slice to copy
        kv_cache_pages (jax.Array): Existing cache pages to update.
            Shape: [total_pages * page_size, num_combined_kv_heads, head_dim]
        total_update_slices (jax.Array): Number of valid slices to process.
            Shape: [1] - scalar wrapped in array for XLA compatibility
        page_size (int): Number of tokens per cache page. Default: 32.
            Must be static for compilation.
        slices_per_processing_page (int): Slices processed per kernel invocation.
            Default: 8. Must divide slice_indices.shape[1] evenly.

    Returns:
        jax.Array: Updated KV cache pages with same shape as kv_cache_pages.

    Raises:
        AssertionError: If dimensions are incompatible or alignment is wrong.

    Note:
        - Requires TPU backend for hardware acceleration
        - head_dim must be divisible by 128 for alignment
        - Automatically falls back to JAX implementation on non-TPU

    Example:
        >>> cache = kv_cache_update(
        ...     new_kv_tokens=new_kv,
        ...     slice_indices=indices,
        ...     kv_cache_pages=cache,
        ...     total_update_slices=jnp.array([10]),
        ...     page_size=32
        ... )
    """
    assert slice_indices.shape[1] % slices_per_processing_page == 0, (
        f"{slices_per_processing_page=}, {slice_indices.shape[1]=}"
    )
    _, num_kv_heads, head_dimension = new_kv_tokens.shape
    assert kv_cache_pages.shape[1] == num_kv_heads
    assert kv_cache_pages.shape[2] == head_dimension, f"{kv_cache_pages.shape[2]=}!={head_dimension=}"
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
                pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
            ],
            out_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY)],
            grid=(cdiv(total_update_slices[0], slices_per_processing_page),),
            scratch_shapes=[vmem_scratch_buffer, pltpu.SemaphoreType.DMA],
        ),
        out_shape=[jax.ShapeDtypeStruct(kv_cache_pages.shape, dtype=kv_cache_pages.dtype)],
        input_output_aliases={len(prefetch_scalars) + 1: 0},
    )

    return pallas_kernel(*prefetch_scalars, new_kv_tokens, kv_cache_pages)[0]


@ejit(static_argnames=["page_size"])
def kv_cache_update_jax(
    new_kv_tokens: Float[Array, "total_tokens num_kv_heads head_dim"],
    slice_indices: Int[Array, "3 num_slices"],
    kv_cache_pages: Float[Array, "total_cache_positions num_kv_heads head_dim"],
    total_update_slices: Int[Array, ""],
    *,
    page_size: int = 32,
) -> Float[Array, "total_cache_positions num_kv_heads head_dim"]:
    """Pure JAX implementation of paged KV cache update.

    Provides a portable fallback implementation using JAX operations
    instead of hardware-specific kernels. While slower than the TPU
    kernel version, this ensures compatibility across all backends.

    The implementation uses dynamic slicing and scanning to update
    cache pages functionally, maintaining JAX's immutability guarantees.

    Algorithm:
    1. Pad new tokens to page boundaries
    2. For each slice in slice_indices:
       - Extract slice from new tokens
       - Create mask for partial updates
       - Merge with existing cache content
       - Update cache slice
    3. Return updated cache

    Args:
        new_kv_tokens (jax.Array): New key/value tokens to insert.
            Shape: [total_tokens, num_kv_heads, head_dim]
        slice_indices (jax.Array): Update mapping information.
            Shape: [3, num_slices] where each column contains:
            - Row 0: Cache starting position
            - Row 1: New tokens starting position
            - Row 2: Number of tokens to copy
        kv_cache_pages (jax.Array): Existing cache to update.
            Shape: [total_pages * page_size, num_kv_heads, head_dim]
        total_update_slices (jax.Array): Number of valid slices.
            Shape: [1] - wrapped scalar for XLA
        page_size (int): Tokens per cache page. Default: 32.
            Must be static for JIT compilation.

    Returns:
        jax.Array: Updated cache with same shape as kv_cache_pages.

    Note:
        This implementation is automatically used on non-TPU backends
        or when the TPU kernel is unavailable.

    Example:
        >>> # Fallback for CPU/GPU
        >>> updated_cache = kv_cache_update_jax(
        ...     new_kv_tokens=tokens,
        ...     slice_indices=indices,
        ...     kv_cache_pages=cache,
        ...     total_update_slices=jnp.array([5]),
        ...     page_size=32
        ... )
    """
    num_valid_slices = total_update_slices[0]
    padded_new_kv = jnp.pad(new_kv_tokens, [(0, page_size), (0, 0), (0, 0)], mode="constant")

    def update_single_slice(
        cache: Float[Array, "total_cache_positions num_kv_heads head_dim"], slice_idx: int
    ) -> Float[Array, "total_cache_positions num_kv_heads head_dim"]:
        """Update cache with a single slice of new tokens.

        Performs a masked update of a cache slice, handling partial
        page updates correctly by preserving unmodified elements.

        Args:
            cache: Current cache state.
            slice_idx: Index of slice to process.

        Returns:
            Updated cache array.
        """
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

    def scan_fn(
        cache: Float[Array, "total_cache_positions num_kv_heads head_dim"], slice_idx: Int[Array, ""]
    ) -> tuple[Float[Array, "total_cache_positions num_kv_heads head_dim"], None]:
        """Scan function for iterating over cache slices.

        Conditionally updates cache based on slice validity.

        Args:
            cache: Current cache state.
            slice_idx: Current slice index.

        Returns:
            tuple: (updated_cache, None)
        """
        updated_cache = jax.lax.cond(
            slice_idx < num_valid_slices,
            lambda c: update_single_slice(c, slice_idx),
            lambda c: c,
            cache,
        )
        return updated_cache, None

    return jax.lax.scan(scan_fn, kv_cache_pages, jnp.arange(slice_indices.shape[1]))[0]
