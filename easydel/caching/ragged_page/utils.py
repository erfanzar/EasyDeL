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
from ejkernel.callib import ejit  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Int


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


def localize_slice_indices_for_page_shard(
    slice_indices: Int[Array, "3 num_slices"],
    total_update_slices: Int[Array, ""],
    *,
    page_size: int,
    local_flat_cache_positions: int,
    page_shard_index: Int[Array, ""] | int = 0,
) -> Int[Array, "3 num_slices"]:
    """Translate global v2 slice mapping offsets to one page shard.

    Slot-mapping metadata stores flattened cache offsets in global page space.
    When KV pages are sharded over a page axis (for example ``dp``), each shard
    receives only a contiguous subset of pages. This helper remaps row-0
    destination offsets into local shard coordinates and zeroes non-local slices.

    Args:
        slice_indices: Global slice mapping ``[3, num_slices]``.
        total_update_slices: Number of valid slices.
        page_size: Tokens per page.
        local_flat_cache_positions: Local flattened cache capacity on this shard.
        page_shard_index: Linear page-shard index for current shard.

    Returns:
        Localized slice mapping with same shape/dtype as ``slice_indices``.
    """
    num_valid_slices = jnp.asarray(total_update_slices).reshape(-1)[0].astype(jnp.int32)
    num_slices = int(slice_indices.shape[1])
    if num_slices == 0:
        return slice_indices

    page_size_i32 = jnp.asarray(page_size, dtype=jnp.int32)
    local_flat_cache_positions_i32 = jnp.asarray(local_flat_cache_positions, dtype=jnp.int32)
    # Keep at least one page to avoid div-by-zero in shape-polymorphic traces.
    local_num_pages = jnp.maximum(local_flat_cache_positions_i32 // page_size_i32, jnp.int32(1))
    shard_index = jnp.asarray(page_shard_index, dtype=jnp.int32)

    slice_ids = jnp.arange(num_slices, dtype=jnp.int32)
    active_slices = slice_ids < num_valid_slices

    global_cache_start = slice_indices[0].astype(jnp.int32)
    global_page = global_cache_start // page_size_i32
    page_offset = global_cache_start % page_size_i32

    local_page_base = shard_index * local_num_pages
    local_page = global_page - local_page_base
    is_local_page = (local_page >= 0) & (local_page < local_num_pages)
    keep = active_slices & is_local_page

    local_cache_start = local_page * page_size_i32 + page_offset
    row0 = jnp.where(keep, local_cache_start, jnp.int32(0))
    row1 = jnp.where(keep, slice_indices[1].astype(jnp.int32), jnp.int32(0))
    row2 = jnp.where(keep, slice_indices[2].astype(jnp.int32), jnp.int32(0))
    return jnp.stack((row0, row1, row2), axis=0).astype(slice_indices.dtype)


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


@ejit(static_argnames=["page_size", "slices_per_processing_page"])  # pyright: ignore[reportUntypedFunctionDecorator]
def kv_cache_update(
    new_kv_tokens: Float[Array, "total_tokens num_combined_kv_heads head_dim"],
    slice_indices: Int[Array, "3 num_slices"],
    kv_cache_pages: Float[Array, "total_cache_positions num_combined_kv_heads head_dim"],
    total_update_slices: Int[Array, ""],
    *,
    page_size: int = 32,
    slices_per_processing_page: int = 8,
    page_shard_index: Int[Array, ""] | int = 0,
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
        page_shard_index (jax.Array | int): Linear page-shard index for local
            cache buffer when pages are sharded. Defaults to 0 (unsharded/global).

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
    slice_indices = localize_slice_indices_for_page_shard(
        slice_indices,
        total_update_slices,
        page_size=page_size,
        local_flat_cache_positions=kv_cache_pages.shape[0],
        page_shard_index=page_shard_index,
    )
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


@ejit(static_argnames=["page_size"], donate_argnames=["kv_cache_pages"])  # pyright: ignore[reportUntypedFunctionDecorator]
def kv_cache_update_jax(
    new_kv_tokens: Float[Array, "total_tokens num_kv_heads head_dim"],
    slice_indices: Int[Array, "3 num_slices"],
    kv_cache_pages: Float[Array, "total_cache_positions num_kv_heads head_dim"],
    total_update_slices: Int[Array, ""],
    *,
    page_size: int = 32,
    page_shard_index: Int[Array, ""] | int = 0,
) -> Float[Array, "total_cache_positions num_kv_heads head_dim"]:
    """Portable JAX implementation of paged KV-cache update.

    The v2 slot-mapping format describes updates as *slices* (start positions +
    lengths) to minimize metadata size. A naive loop/scan over slices becomes
    a latency bottleneck on GPU due to sequential dependence.

    This implementation vectorizes the slice representation into per-token
    scatter indices and applies a single `.at[...].set(...)` update. For GPU
    workloads this is typically much faster than a `lax.scan` of
    `dynamic_update_slice`.
    """
    slice_indices = localize_slice_indices_for_page_shard(
        slice_indices,
        total_update_slices,
        page_size=page_size,
        local_flat_cache_positions=kv_cache_pages.shape[0],
        page_shard_index=page_shard_index,
    )
    # `total_update_slices` is typically shape (1,), but accept scalar too.
    num_valid_slices = jnp.asarray(total_update_slices).reshape(-1)[0].astype(jnp.int32)
    if kv_cache_pages.size == 0:
        return kv_cache_pages

    num_slices = int(slice_indices.shape[1])
    slice_ids = jnp.arange(num_slices, dtype=jnp.int32)
    active_slices = slice_ids < num_valid_slices

    # Slice lengths, masked to 0 for inactive slices.
    slice_lens = jnp.where(active_slices, slice_indices[2].astype(jnp.int32), 0)

    # Build a monotonic "slice ends" vector via cumsum so we can map each token
    # to the slice it belongs to using `searchsorted`.
    slice_ends = jnp.cumsum(slice_lens, dtype=jnp.int32)  # [num_slices]
    total_tokens = slice_ends[-1]  # scalar int32

    # Make searchsorted safe even when `total_tokens == 0` by ensuring the last
    # element is >= the maximum token index.
    slice_ends_safe = slice_ends.at[-1].set(jnp.int32(new_kv_tokens.shape[0]))

    token_ids = jnp.arange(new_kv_tokens.shape[0], dtype=jnp.int32)  # [num_tokens_bucket]
    slice_for_token = jnp.searchsorted(slice_ends_safe, token_ids, side="right").astype(jnp.int32)

    # Token offsets within their slice: offset = packed_token_id - slice_start.
    token_slice_len = slice_lens[slice_for_token]
    token_slice_end = slice_ends[slice_for_token]
    token_slice_start = token_slice_end - token_slice_len
    token_offset = token_ids - token_slice_start

    # Source and destination indices for each packed token.
    src_start = slice_indices[1].astype(jnp.int32)[slice_for_token]
    src = src_start + token_offset
    dst_start = slice_indices[0].astype(jnp.int32)[slice_for_token]
    dst = dst_start + token_offset

    # Drop tokens beyond valid packed prefix (and any accidental src OOB).
    token_mask = (token_ids < total_tokens) & (src >= 0) & (src < new_kv_tokens.shape[0])
    src = jnp.where(token_mask, src, 0)
    dst_oob = jnp.asarray(kv_cache_pages.shape[0], dtype=jnp.int32)
    dst = jnp.where(token_mask, dst, dst_oob)
    values = new_kv_tokens[src]
    return kv_cache_pages.at[dst].set(values, mode="drop")
