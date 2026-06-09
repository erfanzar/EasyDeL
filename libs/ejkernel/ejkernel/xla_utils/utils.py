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


"""Utility functions for packed sequence processing in XLA.

This module provides efficient utilities for working with packed (variable-length)
sequences in JAX/XLA computations, commonly used in attention mechanisms.

Packed sequences are represented using cumulative sequence lengths (cu_seqlens),
which define the boundaries of each sequence in a flattened 1D tensor.

Key Concepts:
    - cu_seqlens: [0, len1, len1+len2, ...] - cumulative start positions
    - position_ids: Position within each sequence (0-indexed)
    - sequence_ids: Which sequence each token belongs to (0-indexed)
    - chunk_indices: For tiled processing with fixed chunk sizes

Functions:
    cdiv: Ceiling division for computing block counts
    prepare_lens: Extract individual lengths from cumulative lengths
    prepare_position_ids: Generate per-token position indices
    prepare_sequence_ids: Generate per-token sequence membership
    prepare_token_indices: Combined (seq_id, pos_id) pairs
    prepare_chunk_indices: Chunk-level indices for tiled attention
    prepare_chunk_offsets: Cumulative chunk counts per sequence
    identity_dtype_convert: Create identity function with dtype conversion on backward

Example:
    >>> cu_seqlens = jnp.array([0, 3, 5, 9])  # 3 sequences
    >>> lens = prepare_lens(cu_seqlens)  # [3, 2, 4]
    >>> pos_ids = prepare_position_ids(cu_seqlens)  # [0, 1, 2, 0, 1, 0, 1, 2, 3]
"""

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, DTypeLike, Int


def cdiv(a: Int[Array, "..."], b: int) -> Int[Array, "..."]:
    """
    Compute ceiling division for integers in a JAX-compatible way.

    Calculates ceil(a / b) without using floating-point operations, making it
    suitable for computing block/chunk counts in tiled computations.

    Args:
        a: Numerator array of integers (any shape).
        b: Divisor (positive integer).

    Returns:
        Array of ceiling division results with same shape as input.

    Examples:
        >>> import jax.numpy as jnp
        >>> from ejkernel.xla_utils import cdiv
        >>>
        >>> # Compute number of chunks needed
        >>> seq_lens = jnp.array([100, 200, 150])
        >>> chunk_size = 64
        >>> num_chunks = cdiv(seq_lens, chunk_size)
        >>> # Returns: [2, 4, 3] (100/64=1.56->2, 200/64=3.125->4, 150/64=2.34->3)
    """
    return (a + b - 1) // b


def prepare_lens(cu_seqlens: Int[Array, "num_seqs_plus_one"]) -> Int[Array, "num_seqs"]:
    """
    Calculate individual sequence lengths from cumulative sequence lengths.

    Extracts the length of each sequence from a cumulative length array by
    computing consecutive differences.

    Args:
        cu_seqlens: A 1D array of cumulative sequence lengths with shape
            (num_sequences + 1,). First element must be 0, subsequent elements
            are cumulative sums of sequence lengths.

    Returns:
        A 1D array of sequence lengths with shape (num_sequences,).

    Examples:
        >>> import jax.numpy as jnp
        >>> from ejkernel.xla_utils import prepare_lens
        >>>
        >>> # Three sequences of lengths 3, 2, 4
        >>> cu_seqlens = jnp.array([0, 3, 5, 9])
        >>> lens = prepare_lens(cu_seqlens)
        >>> # Returns: [3, 2, 4]
    """
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_lens_from_mask(mask: Bool[Array, "batch seq_len"]) -> Int[Array, "batch"]:
    """
    Calculate sequence lengths from a boolean attention mask.

    Counts the number of True values along the sequence dimension for each
    batch element to determine actual sequence lengths.

    Args:
        mask: A 2D boolean attention mask of shape (batch_size, seq_len).
            True indicates valid tokens, False indicates padding.

    Returns:
        A 1D array of sequence lengths with shape (batch_size,) and dtype int32.

    Examples:
        >>> import jax.numpy as jnp
        >>> from ejkernel.xla_utils import prepare_lens_from_mask
        >>>
        >>> # Batch of 2 sequences with different lengths
        >>> mask = jnp.array([[True, True, True, False],
        ...                   [True, True, False, False]])
        >>> lens = prepare_lens_from_mask(mask)
        >>> # Returns: [3, 2]
    """
    return mask.sum(axis=-1, dtype=jnp.int32)


def prepare_cu_seqlens_from_mask(
    mask: Bool[Array, "batch seq_len"], out_dtype: DTypeLike = jnp.int32
) -> Int[Array, "batch_plus_one"]:
    """
    Create cumulative sequence lengths from a boolean attention mask.

    Converts a batch of attention masks into a single cumulative length array
    suitable for packed sequence processing.

    Args:
        mask: A 2D boolean attention mask of shape (batch_size, seq_len).
            True indicates valid tokens, False indicates padding.
        out_dtype: The desired dtype for the output array. Defaults to int32.

    Returns:
        A 1D array of cumulative sequence lengths with shape (batch_size + 1,).
        First element is always 0, subsequent elements are cumulative sums.

    Examples:
        >>> import jax.numpy as jnp
        >>> from ejkernel.xla_utils import prepare_cu_seqlens_from_mask
        >>>
        >>> # Batch of 3 sequences with lengths 3, 2, 4
        >>> mask = jnp.array([[True, True, True, False],
        ...                   [True, True, False, False],
        ...                   [True, True, True, True]])
        >>> cu_seqlens = prepare_cu_seqlens_from_mask(mask)
        >>> # Returns: [0, 3, 5, 9]
    """
    cumsum_lens = prepare_lens_from_mask(mask).cumsum(axis=0, dtype=out_dtype)
    return jnp.pad(cumsum_lens, (1, 0))


def prepare_position_ids(cu_seqlens: Int[Array, "num_seqs_plus_one"]) -> Int[Array, "total_tokens"]:
    """
    Generate position IDs for a batch of packed sequences.

    Creates a single 1D array of position indices where positions reset to 0
    at the start of each sequence. This is essential for transformer models
    that use position embeddings with packed/concatenated sequences.

    Args:
        cu_seqlens: A 1D array of cumulative sequence lengths with shape
            (num_sequences + 1,). First element must be 0.

    Returns:
        A 1D array of position IDs with shape (total_tokens,) where total_tokens
        equals the last element of cu_seqlens.

    Examples:
        >>> import jax.numpy as jnp
        >>> from ejkernel.xla_utils import prepare_position_ids
        >>>
        >>> # Three sequences of lengths 3, 2, 4
        >>> cu_seqlens = jnp.array([0, 3, 5, 9])
        >>> pos_ids = prepare_position_ids(cu_seqlens)
        >>> # Returns: [0, 1, 2, 0, 1, 0, 1, 2, 3]
        >>> #          |seq 1 | seq2| seq 3    |
    """
    lens = prepare_lens(cu_seqlens)
    total_length = cu_seqlens[-1]

    indices = jnp.arange(total_length, dtype=cu_seqlens.dtype)

    start_offsets = jnp.repeat(cu_seqlens[:-1], repeats=lens)

    return indices - start_offsets


def prepare_sequence_ids(cu_seqlens: Int[Array, "num_seqs_plus_one"]) -> Int[Array, "total_tokens"]:
    """
    Generate sequence IDs (0-indexed) for a batch of packed sequences.

    Creates a single 1D array indicating which sequence each token belongs to.
    Useful for masking attention to prevent cross-sequence attention in packed
    batches.

    Args:
        cu_seqlens: A 1D array of cumulative sequence lengths with shape
            (num_sequences + 1,). First element must be 0.

    Returns:
        A 1D array of sequence IDs with shape (total_tokens,), where each
        element is the 0-indexed sequence number that token belongs to.

    Examples:
        >>> import jax.numpy as jnp
        >>> from ejkernel.xla_utils import prepare_sequence_ids
        >>>
        >>> # Three sequences of lengths 3, 2, 4
        >>> cu_seqlens = jnp.array([0, 3, 5, 9])
        >>> seq_ids = prepare_sequence_ids(cu_seqlens)
        >>> # Returns: [0, 0, 0, 1, 1, 2, 2, 2, 2]
        >>> #          |seq 0 |seq 1| seq 2    |
    """
    position_ids = prepare_position_ids(cu_seqlens)
    return (position_ids == 0).cumsum(axis=0) - 1


def prepare_token_indices(cu_seqlens: Int[Array, "num_seqs_plus_one"]) -> Int[Array, "total_tokens 2"]:
    """
    Generate (sequence_id, position_id) pairs for each token in the packed batch.

    Combines sequence and position information into a single array for efficient
    indexing operations during attention computation.

    Args:
        cu_seqlens: A 1D array of cumulative sequence lengths with shape
            (num_sequences + 1,). First element must be 0.

    Returns:
        A 2D array of shape (total_tokens, 2) where each row contains
        [sequence_id, position_id] for the corresponding token.

    Examples:
        >>> import jax.numpy as jnp
        >>> from ejkernel.xla_utils import prepare_token_indices
        >>>
        >>> # Two sequences of lengths 3, 2
        >>> cu_seqlens = jnp.array([0, 3, 5])
        >>> indices = prepare_token_indices(cu_seqlens)
        >>> # Returns: [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]
        >>> #          [seq_id, pos_id] for each token
    """
    position_ids = prepare_position_ids(cu_seqlens)

    sequence_ids = (position_ids == 0).cumsum(axis=0) - 1

    stacked = jnp.stack([sequence_ids, position_ids], axis=1)
    return stacked.astype(cu_seqlens.dtype)


def prepare_chunk_indices(cu_seqlens: Int[Array, "num_seqs_plus_one"], chunk_size: int) -> Int[Array, "total_chunks 2"]:
    """
    Generate (sequence_id, chunk_id) pairs for each chunk in the packed batch.

    Useful for tiled/chunked attention implementations where the sequence is
    processed in fixed-size blocks. Each sequence may have a different number
    of chunks depending on its length.

    Args:
        cu_seqlens: A 1D array of cumulative sequence lengths with shape
            (num_sequences + 1,). First element must be 0.
        chunk_size: The size of each chunk. Sequences are divided into
            ceil(length / chunk_size) chunks.

    Returns:
        A 2D array of shape (total_chunks, 2) where each row contains
        [sequence_id, chunk_id_within_sequence] for that chunk.

    Examples:
        >>> import jax.numpy as jnp
        >>> from ejkernel.xla_utils import prepare_chunk_indices
        >>>
        >>> # Two sequences of lengths 100, 64 with chunk_size=64
        >>> cu_seqlens = jnp.array([0, 100, 164])
        >>> chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size=64)
        >>> # Returns: [[0, 0], [0, 1], [1, 0]]
        >>> # seq0 needs 2 chunks (100/64=1.56->2), seq1 needs 1 chunk (64/64=1)
    """
    lens = prepare_lens(cu_seqlens)
    num_chunks_per_seq = cdiv(lens, chunk_size)

    total_chunks = num_chunks_per_seq.sum()
    cu_chunks = jnp.pad(num_chunks_per_seq.cumsum(), (1, 0))
    start_offsets = jnp.repeat(cu_chunks[:-1], repeats=num_chunks_per_seq)

    indices = jnp.arange(total_chunks) - start_offsets

    sequence_ids_for_chunks = (indices == 0).cumsum(axis=0) - 1

    stacked = jnp.stack([sequence_ids_for_chunks, indices], axis=1)
    return stacked.astype(cu_seqlens.dtype)


def prepare_chunk_offsets(
    cu_seqlens: Int[Array, "num_seqs_plus_one"], chunk_size: int
) -> Int[Array, "num_seqs_plus_one"]:
    """
    Compute cumulative chunk offsets for packed sequences.

    Creates an array similar to cu_seqlens but for chunks instead of tokens.
    This is useful for indexing into chunk-level data structures.

    Args:
        cu_seqlens: A 1D array of cumulative sequence lengths with shape
            (num_sequences + 1,). First element must be 0.
        chunk_size: The size of each chunk.

    Returns:
        A 1D array of cumulative chunk counts with shape (num_sequences + 1,).
        Element i gives the total number of chunks in sequences 0..i-1.

    Examples:
        >>> import jax.numpy as jnp
        >>> from ejkernel.xla_utils import prepare_chunk_offsets
        >>>
        >>> # Two sequences of lengths 100, 64 with chunk_size=64
        >>> cu_seqlens = jnp.array([0, 100, 164])
        >>> chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size=64)
        >>> # Returns: [0, 2, 3]
        >>> # seq0 has 2 chunks, seq1 has 1 chunk
    """
    num_chunks_per_seq = cdiv(prepare_lens(cu_seqlens), chunk_size)
    zero = jnp.array([0], dtype=cu_seqlens.dtype)

    concatenated = jnp.concatenate([zero, num_chunks_per_seq])
    return concatenated.cumsum(axis=-1)


def identity_dtype_convert(dtype: jnp.dtype):
    """Create an identity function that converts gradients to a specific dtype.

    Returns a function that passes inputs unchanged in the forward pass,
    but converts gradients to the specified dtype during backpropagation.
    This is useful for mixed-precision training where gradients need to
    be accumulated in a specific precision.

    Args:
        dtype: The target dtype for gradient conversion.

    Returns:
        A JAX function that acts as identity in forward pass but
        converts gradients to the specified dtype in backward pass.

    Example:
        >>> convert_to_fp32 = identity_dtype_convert(jnp.float32)
        >>> result = convert_to_fp32(bf16_tensor)  # Forward: unchanged
        >>> # Backward: gradients will be converted to float32
    """

    @jax.custom_vjp
    def identity_fn(x):
        return x

    def identity_fn_fwd(x):
        return x, None

    def identity_fn_bwd(res, g):
        return (g.astype(dtype),)

    identity_fn.defvjp(identity_fn_fwd, identity_fn_bwd)

    return identity_fn
