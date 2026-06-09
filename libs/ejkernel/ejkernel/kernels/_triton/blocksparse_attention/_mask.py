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


"""Sparse mask generation for block-sparse attention.

This module provides utilities for creating and managing sparse attention masks
at the block level. Instead of computing full token-level attention masks,
block-sparse attention uses coarse-grained masks that specify which blocks of
queries attend to which blocks of keys.

The SparseMask dataclass encapsulates four boundary arrays that define:
1. lower_bounds: First KV block each query block attends to (sparse pattern)
2. upper_bounds: Last KV block each query block attends to (sparse pattern)
3. lower_full_bounds: First fully-attended KV block (optimization)
4. upper_full_bounds: Last fully-attended KV block (optimization)

These masks enable significant performance optimizations:
- Skip computation for masked-out blocks entirely
- Use faster kernels for fully-attended blocks (no masking logic)
- Support causal masking, sliding windows, and custom patterns
- Handle segment IDs for packed variable-length sequences

The mask computation is performed on GPU using Triton kernels for efficiency,
and the resulting masks are used by both forward and backward attention passes.

Key functions:
- create_sparsity_mask: High-level API for generating masks
- define_sparse_mask_fn: Core mask generation logic
- SparseMask.from_inputs: Create mask from positions and segment IDs

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._triton.blocksparse_attention._mask import create_sparsity_mask
    >>>
    >>> batch, seq_len = 2, 512
    >>> q_positions = jnp.arange(seq_len).reshape(1, -1).repeat(batch, 0)
    >>> kv_positions = jnp.arange(seq_len).reshape(1, -1).repeat(batch, 0)
    >>> q_segment_ids = jnp.zeros((batch, seq_len), dtype=jnp.int32)
    >>> kv_segment_ids = jnp.zeros((batch, seq_len), dtype=jnp.int32)
    >>>
    >>>
    >>> masks = create_sparsity_mask(
    ...     q_positions, q_segment_ids,
    ...     kv_positions, kv_segment_ids,
    ...     q_blocksize=64, kv_blocksize=64,
    ...     causal=True
    ... )
"""

import chex
import jax
import jax.numpy as jnp
import triton
import triton.language as tl
from jax.sharding import Mesh
from jaxtyping import ArrayLike

from ejkernel.callib import cdiv, triton_call

from ._utilities import PADDING_SEGMENT_ID, pad_to_block_size


@chex.dataclass
class SparseMask:
    """Sparse attention mask at the block level.

    This dataclass represents a sparse attention pattern by defining which blocks
    of keys/values each block of queries should attend to. It uses boundary arrays
    to efficiently encode the sparsity pattern without materializing a full mask.

    Attributes:
        lower_bounds: First KV block index each query block attends to, shape
            (batch, 1, num_q_blocks). Defines the start of the attention range.
        upper_bounds: Last KV block index (+1) each query block attends to, shape
            (batch, 1, num_q_blocks). Defines the end of the attention range.
        lower_full_bounds: First KV block index that is fully attended (no partial
            masking), shape (batch, 1, num_q_blocks). Used for kernel optimization.
        upper_full_bounds: Last KV block index (+1) that is fully attended, shape
            (batch, 1, num_q_blocks). Used for kernel optimization.

    The bounds define half-open intervals [lower, upper) for each query block.
    Blocks outside these bounds are completely masked out. Blocks between
    lower_full_bounds and upper_full_bounds can use optimized kernels without
    per-token masking logic.

    Example:
        A causal mask for a sequence divided into 4 blocks might have:
        - Block 0: [0, 1) with [0, 1) fully attended
        - Block 1: [0, 2) with [0, 2) fully attended
        - Block 2: [0, 3) with [0, 3) fully attended
        - Block 3: [0, 4) with [0, 4) fully attended
    """

    lower_bounds: ArrayLike | None
    upper_bounds: ArrayLike | None
    lower_full_bounds: ArrayLike | None
    upper_full_bounds: ArrayLike | None

    @classmethod
    def from_inputs(
        cls,
        q_positions: ArrayLike,
        q_segment_ids: ArrayLike,
        kv_positions: ArrayLike,
        kv_segment_ids: ArrayLike,
        kv_blocksize: int,
        q_blocksize: int,
        calculate_dkdv_mask: bool = False,
        causal: bool = True,
        window_left: int = -1,
        window_right: int = -1,
        mesh: Mesh | None = None,
    ):
        """Create a SparseMask from query and key-value positions and segments.

        This factory method generates a sparse attention mask by analyzing the
        positions and segment IDs of queries and keys/values. It automatically
        determines which blocks should attend to each other based on the specified
        attention pattern (causal, windowed, etc.).

        Args:
            q_positions: Query token positions, shape (batch, q_seq_len).
            q_segment_ids: Query segment IDs for packed sequences, shape (batch, q_seq_len).
            kv_positions: Key/value token positions, shape (batch, kv_seq_len).
            kv_segment_ids: Key/value segment IDs, shape (batch, kv_seq_len).
            kv_blocksize: Size of key/value blocks in tokens.
            q_blocksize: Size of query blocks in tokens.
            calculate_dkdv_mask: If True, compute mask for gradient computation
                with respect to keys/values (backward pass).
            causal: If True, apply causal masking (lower triangular).
            window_left: Left window size for sliding window attention (-1 for unlimited).
            window_right: Right window size for sliding window attention (-1 for unlimited).
            mesh: Optional device mesh for distributed computation.

        Returns:
            A SparseMask instance with computed boundary arrays.

        Example:
            >>> import jax.numpy as jnp
            >>> batch, seq_len = 2, 256
            >>> positions = jnp.arange(seq_len).reshape(1, -1).repeat(batch, 0)
            >>> segments = jnp.zeros((batch, seq_len), dtype=jnp.int32)
            >>>
            >>> mask = SparseMask.from_inputs(
            ...     positions, segments, positions, segments,
            ...     kv_blocksize=64, q_blocksize=64, causal=True
            ... )
        """
        lower_bounds, upper_bounds, lower_full_bounds, upper_full_bounds = define_sparse_mask_fn(
            q_positions,
            q_segment_ids,
            kv_positions,
            kv_segment_ids,
            kv_blocksize=kv_blocksize,
            q_blocksize=q_blocksize,
            calculate_dkdv_mask=calculate_dkdv_mask,
            causal=causal,
            window_left=window_left,
            window_right=window_right,
        )
        return SparseMask(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            upper_full_bounds=upper_full_bounds,
            lower_full_bounds=lower_full_bounds,
        )


@triton.jit
def _compute_sparse_mask(
    outer_positions_ptr,
    outer_segment_id_ptr,
    inner_positions_ptr,
    inner_segment_ids_ptr,
    lower_block_ptr,
    upper_block_ptr,
    lower_full_block_ptr,
    upper_full_block_ptr,
    INNER_BLOCK_SIZE: tl.constexpr,
    INNER_SEQ_LEN: tl.constexpr,
    OUTER_SEQ_LEN: tl.constexpr,
    OUTER_BLOCK_SIZE: tl.constexpr,
    PADDING_SEGMENT_ID: tl.constexpr,
    USE_SEGMENT_MASK: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    QUERY_IS_OUTER: tl.constexpr,
):
    """Triton kernel to compute sparse attention mask boundaries.

    Analyzes block-level attention patterns to determine which blocks of
    the inner sequence each block of the outer sequence should attend to.
    Computes both full bounds (for unmasked computation) and partial bounds
    (where masking is needed).

    Args:
        outer_positions_ptr: Pointer to outer (query or KV) positions
        outer_segment_id_ptr: Pointer to outer segment IDs
        inner_positions_ptr: Pointer to inner (KV or query) positions
        inner_segment_ids_ptr: Pointer to inner segment IDs
        lower_block_ptr: Output pointer for lower bound indices
        upper_block_ptr: Output pointer for upper bound indices
        lower_full_block_ptr: Output pointer for lower fully-attended bounds
        upper_full_block_ptr: Output pointer for upper fully-attended bounds
        INNER_BLOCK_SIZE: Block size for inner sequence (constexpr)
        INNER_SEQ_LEN: Total inner sequence length (constexpr)
        OUTER_SEQ_LEN: Total outer sequence length (constexpr)
        OUTER_BLOCK_SIZE: Block size for outer sequence (constexpr)
        PADDING_SEGMENT_ID: ID used for padding tokens (constexpr)
        USE_SEGMENT_MASK: Enable segment-based masking (constexpr)
        CAUSAL: Enable causal masking (constexpr)
        WINDOW_LEFT: Left window boundary for sliding window (constexpr)
        WINDOW_RIGHT: Right window boundary for sliding window (constexpr)
        QUERY_IS_OUTER: True if outer is query, False if outer is KV (constexpr)
    """
    outer_block_id = tl.program_id(0)
    batch_size_id = tl.program_id(1)
    num_outer_block_programs = tl.num_programs(0)

    outer_positions_ptr += batch_size_id * OUTER_SEQ_LEN
    outer_segment_id_ptr += batch_size_id * OUTER_SEQ_LEN
    inner_positions_ptr += batch_size_id * INNER_SEQ_LEN
    inner_segment_ids_ptr += batch_size_id * INNER_SEQ_LEN

    lower_block_ptr += batch_size_id * num_outer_block_programs + outer_block_id
    upper_block_ptr += batch_size_id * num_outer_block_programs + outer_block_id
    lower_full_block_ptr += batch_size_id * num_outer_block_programs + outer_block_id
    upper_full_block_ptr += batch_size_id * num_outer_block_programs + outer_block_id

    outer_arange = outer_block_id * OUTER_BLOCK_SIZE + tl.arange(0, OUTER_BLOCK_SIZE)

    outer_positions_block = tl.load(outer_positions_ptr + outer_arange)
    outer_segments_block = tl.load(outer_segment_id_ptr + outer_arange)

    outer_max_seg_id = tl.max(outer_segments_block)
    outer_min_seg_id = tl.min(outer_segments_block)
    outer_same_segment = outer_max_seg_id == outer_min_seg_id
    if (outer_same_segment) & (outer_min_seg_id == PADDING_SEGMENT_ID):
        tl.store(lower_block_ptr, 0)
        tl.store(upper_block_ptr, 0)
        tl.store(lower_full_block_ptr, 0)
        tl.store(upper_full_block_ptr, 0)
        return

    max_outer_position = tl.max(outer_positions_block)
    min_outer_position = tl.min(outer_positions_block)

    upper_block_to_attend = 0
    lower_block_to_attend = INNER_SEQ_LEN // INNER_BLOCK_SIZE

    upper_full_block = 0
    lower_full_block = INNER_SEQ_LEN // INNER_BLOCK_SIZE

    for inner_idx in range(0, INNER_SEQ_LEN // INNER_BLOCK_SIZE):
        inner_offset = inner_idx * INNER_BLOCK_SIZE + tl.arange(0, INNER_BLOCK_SIZE)
        inner_positions_block = tl.load(inner_positions_ptr + inner_offset)
        inner_segments_block = tl.load(inner_segment_ids_ptr + inner_offset)
        inner_min_seg_id = tl.min(inner_segments_block)
        inner_max_seg_id = tl.max(inner_segments_block)

        inner_same_segment = inner_max_seg_id == inner_min_seg_id

        should_attend_segments = (inner_min_seg_id <= outer_max_seg_id) & (outer_min_seg_id <= inner_max_seg_id)
        full_block_segments = outer_same_segment & inner_same_segment

        min_inner_position = tl.min(inner_positions_block)
        max_inner_position = tl.max(inner_positions_block)

        USE_WINDOW: tl.constexpr = WINDOW_LEFT >= 0 or WINDOW_RIGHT >= 0

        if QUERY_IS_OUTER:
            if CAUSAL:
                should_attend_positions = max_outer_position >= min_inner_position
                full_block_positions = min_outer_position >= max_inner_position
            else:
                should_attend_positions = True
                full_block_positions = True

            if USE_WINDOW:
                if WINDOW_LEFT >= 0:
                    window_attend_left = min_inner_position >= (min_outer_position - WINDOW_LEFT)
                    window_full_left = min_inner_position >= (max_outer_position - WINDOW_LEFT)
                else:
                    window_attend_left = True
                    window_full_left = True

                if WINDOW_RIGHT >= 0:
                    window_attend_right = max_inner_position <= (max_outer_position + WINDOW_RIGHT)
                    window_full_right = max_inner_position <= (min_outer_position + WINDOW_RIGHT)
                else:
                    window_attend_right = True
                    window_full_right = True

                should_attend_positions = should_attend_positions & window_attend_left & window_attend_right
                full_block_positions = full_block_positions & window_full_left & window_full_right
        else:
            if CAUSAL:
                should_attend_positions = max_inner_position >= min_outer_position
                full_block_positions = min_inner_position >= max_outer_position
            else:
                should_attend_positions = True
                full_block_positions = True

            if USE_WINDOW:
                if WINDOW_LEFT >= 0:
                    window_attend_left = max_inner_position >= (min_outer_position - WINDOW_LEFT)
                    window_full_left = min_inner_position >= (min_outer_position - WINDOW_LEFT)
                else:
                    window_attend_left = True
                    window_full_left = True

                if WINDOW_RIGHT >= 0:
                    window_attend_right = min_inner_position <= (max_outer_position + WINDOW_RIGHT)
                    window_full_right = max_inner_position <= (max_outer_position + WINDOW_RIGHT)
                else:
                    window_attend_right = True
                    window_full_right = True

                should_attend_positions = should_attend_positions & window_attend_left & window_attend_right
                full_block_positions = full_block_positions & window_full_left & window_full_right

        if USE_SEGMENT_MASK:
            should_attend = should_attend_positions & should_attend_segments
        else:
            should_attend = should_attend_positions

        is_pad_tokens = inner_min_seg_id == PADDING_SEGMENT_ID
        should_attend = should_attend & (~is_pad_tokens)

        should_not_attend = 1 - should_attend

        upper_block_to_attend = tl.maximum(upper_block_to_attend, should_attend * (inner_idx + 1))

        lower_block_to_attend = tl.minimum(
            lower_block_to_attend,
            should_attend * inner_idx + should_not_attend * lower_block_to_attend,
        )

        full_block = (full_block_segments & full_block_positions) & should_attend
        not_full_block = 1 - full_block
        upper_full_block = tl.maximum(upper_full_block, full_block * (inner_idx + 1))

        lower_full_block = tl.minimum(
            lower_full_block,
            full_block * inner_idx + not_full_block * lower_full_block,
        )

    tl.store(lower_block_ptr, lower_block_to_attend)
    tl.store(upper_block_ptr, upper_block_to_attend)
    tl.store(lower_full_block_ptr, lower_full_block)
    tl.store(upper_full_block_ptr, upper_full_block)


def define_sparse_mask_fn(
    q_positions: ArrayLike,
    q_segment_ids: ArrayLike,
    kv_positions: ArrayLike,
    kv_segment_ids: ArrayLike,
    kv_blocksize: int,
    q_blocksize: int,
    calculate_dkdv_mask: bool = False,
    causal: bool = True,
    window_left: int = -1,
    window_right: int = -1,
) -> SparseMask:
    """Generate sparse attention mask boundaries using Triton kernel.

    Core function that invokes the Triton kernel to compute block-level
    attention boundaries. Handles padding, segment ID remapping, and
    boundary clipping for both forward and backward masks.

    Args:
        q_positions: Query token positions [batch, seq_len_q]
        q_segment_ids: Query segment IDs [batch, seq_len_q]
        kv_positions: Key/value token positions [batch, seq_len_k]
        kv_segment_ids: Key/value segment IDs [batch, seq_len_k]
        kv_blocksize: Block size for key/value sequence
        q_blocksize: Block size for query sequence
        calculate_dkdv_mask: If True, compute mask for dK/dV gradients
        causal: Enable causal (lower triangular) masking
        window_left: Left window size for sliding window (-1 = unlimited)
        window_right: Right window size for sliding window (-1 = unlimited)

    Returns:
        tuple: (lower_block, upper_block, lower_full_block, upper_full_block)
        boundary arrays defining which blocks each query/KV block attends to
    """
    _, q_positions, q_segment_ids = pad_to_block_size(
        inputs=None,
        indexs=q_positions,
        segment_ids=q_segment_ids,
        block_size=q_blocksize,
        pos_fill_value=-1,
    )
    _, kv_positions, kv_segment_ids = pad_to_block_size(
        inputs=None,
        indexs=kv_positions,
        segment_ids=kv_segment_ids,
        block_size=kv_blocksize,
        pos_fill_value=jnp.iinfo(jnp.int32).max,
    )

    batch_size, query_seq_len = q_positions.shape
    _, kv_seq_len = kv_positions.shape

    chex.assert_shape([kv_positions, kv_segment_ids], [batch_size, kv_seq_len])
    num_query_blocks = cdiv(query_seq_len, q_blocksize)
    num_kv_blocks = cdiv(kv_seq_len, kv_blocksize)

    if calculate_dkdv_mask:
        output_shape = jax.ShapeDtypeStruct(shape=(batch_size, 1, num_kv_blocks), dtype=jnp.int32)
    else:
        output_shape = jax.ShapeDtypeStruct(shape=(batch_size, 1, num_query_blocks), dtype=jnp.int32)

    INNER_PADDING_SEGMENT_ID = kv_seq_len + 1

    q_segment_ids = jnp.where(q_segment_ids == PADDING_SEGMENT_ID, INNER_PADDING_SEGMENT_ID, q_segment_ids)
    kv_segment_ids = jnp.where(kv_segment_ids == PADDING_SEGMENT_ID, INNER_PADDING_SEGMENT_ID, kv_segment_ids)

    common_params = dict(
        kernel=_compute_sparse_mask,
        out_shape=(output_shape, output_shape, output_shape, output_shape),
        debug=False,
        PADDING_SEGMENT_ID=INNER_PADDING_SEGMENT_ID,
        USE_SEGMENT_MASK=True,
        CAUSAL=causal,
        WINDOW_LEFT=window_left,
        WINDOW_RIGHT=window_right,
        num_warps=2,
        num_stages=4,
    )

    if calculate_dkdv_mask:
        lower_block, upper_block, lower_full_block, upper_full_block = triton_call(
            kv_positions,
            kv_segment_ids,
            q_positions,
            q_segment_ids,
            INNER_BLOCK_SIZE=q_blocksize,
            OUTER_BLOCK_SIZE=kv_blocksize,
            INNER_SEQ_LEN=query_seq_len,
            OUTER_SEQ_LEN=kv_seq_len,
            QUERY_IS_OUTER=False,
            grid=(num_kv_blocks, batch_size),
            name="ejkernel::triton::blocksparse_attn_mask_dkdv",
            **common_params,
        )

        lower_block = jnp.clip(lower_block, 0, num_query_blocks)
        upper_block = jnp.clip(upper_block, 0, num_query_blocks)
        lower_block = jnp.minimum(lower_block, upper_block)

        lower_full_block = jnp.clip(lower_full_block, 0, num_query_blocks)
        no_full_block = upper_full_block == 0
        lower_full_block = jnp.where(no_full_block, lower_block, lower_full_block)
        lower_full_block = jnp.maximum(lower_full_block, lower_block)

        upper_full_block = jnp.clip(upper_full_block, 0, num_query_blocks)
        upper_full_block = jnp.where(no_full_block, lower_block, upper_full_block)
        upper_full_block = jnp.maximum(upper_full_block, lower_full_block)
        upper_full_block = jnp.minimum(upper_full_block, upper_block)

        return lower_block, upper_block, lower_full_block, upper_full_block

    lower_block, upper_block, lower_full_block, upper_full_block = triton_call(
        q_positions,
        q_segment_ids,
        kv_positions,
        kv_segment_ids,
        INNER_BLOCK_SIZE=kv_blocksize,
        OUTER_BLOCK_SIZE=q_blocksize,
        INNER_SEQ_LEN=kv_seq_len,
        OUTER_SEQ_LEN=query_seq_len,
        QUERY_IS_OUTER=True,
        grid=(num_query_blocks, batch_size),
        name="ejkernel::triton::blocksparse_attn_mask_dq",
        **common_params,
    )

    lower_block = jnp.clip(lower_block, 0, num_kv_blocks)
    upper_block = jnp.clip(upper_block, 0, num_kv_blocks)
    lower_block = jnp.minimum(lower_block, upper_block)
    lower_full_block = jnp.clip(lower_full_block, 0, num_kv_blocks)
    no_full_block = upper_full_block == 0
    lower_full_block = jnp.where(no_full_block, lower_block, lower_full_block)
    lower_full_block = jnp.maximum(lower_full_block, lower_block)
    upper_full_block = jnp.clip(upper_full_block, 0, num_kv_blocks)
    upper_full_block = jnp.where(no_full_block, lower_block, upper_full_block)
    upper_full_block = jnp.maximum(upper_full_block, lower_full_block)
    upper_full_block = jnp.minimum(upper_full_block, upper_block)
    return lower_block, upper_block, lower_full_block, upper_full_block


def create_sparsity_mask(
    q_positions: ArrayLike,
    q_segment_ids: ArrayLike,
    kv_positions: ArrayLike,
    kv_segment_ids: ArrayLike,
    mesh: Mesh | None = None,
    kv_blocksize: int = 64,
    q_blocksize: int = 64,
    causal: bool = True,
    window_left: int = -1,
    window_right: int = -1,
) -> tuple[SparseMask, ...]:
    """Create sparse attention masks for forward and backward block-sparse attention passes.

    Generates three ``SparseMask`` objects needed for training:

    - Index 0 / 1 (``fwd_bwd_q_mask``): forward pass query mask, also reused for dQ
      in the backward pass.
    - Index 2 (``dkdv_mask``): backward pass dK/dV mask (transposed sparsity, each KV
      block sees which query blocks attend to it).

    Args:
        q_positions: Query token positions, shape (batch_size, query_seq_length).
        q_segment_ids: Segment IDs for query tokens, shape (batch_size, query_seq_length).
        kv_positions: Key/value token positions, shape (batch_size, kv_seq_length).
        kv_segment_ids: Segment IDs for key/value tokens, shape (batch_size, kv_seq_length).
        mesh: Optional JAX device mesh for distributed execution. Currently accepted but
            forwarded to ``SparseMask.from_inputs`` without effect on sharding.
        kv_blocksize: Block size for the KV sequence dimension (default: 64).
        q_blocksize: Block size for the query sequence dimension (default: 64).
        causal: If True, apply causal (lower-triangular) masking (default: True).
        window_left: Left window size for sliding-window attention (-1 = unlimited).
        window_right: Right window size for sliding-window attention (-1 = unlimited).

    Returns:
        A 3-tuple ``(fwd_bwd_q_mask, fwd_bwd_q_mask, dkdv_mask)`` of ``SparseMask``
        instances. The first two elements are identical (the forward/dQ mask) and the
        third is the transposed dK/dV mask.
    """
    fwd_bwd_q_mask = SparseMask.from_inputs(
        q_positions,
        q_segment_ids,
        kv_positions,
        kv_segment_ids,
        kv_blocksize=kv_blocksize,
        q_blocksize=q_blocksize,
        causal=causal,
        window_left=window_left,
        window_right=window_right,
        mesh=mesh,
    )
    dkdv_mask = SparseMask.from_inputs(
        q_positions,
        q_segment_ids,
        kv_positions,
        kv_segment_ids,
        kv_blocksize=kv_blocksize,
        q_blocksize=q_blocksize,
        calculate_dkdv_mask=True,
        causal=causal,
        window_left=window_left,
        window_right=window_right,
        mesh=mesh,
    )
    return fwd_bwd_q_mask, fwd_bwd_q_mask, dkdv_mask
