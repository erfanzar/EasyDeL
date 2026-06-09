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


"""Chunked cumulative sum operations for attention mechanisms.

This module provides efficient chunked cumulative sum operations used in
various attention mechanisms, particularly linear attention variants that
require running sums over sequence dimensions.

Key Operations:
    - chunk_local_cumsum: Cumsum within fixed-size chunks (resets at boundaries)
    - chunk_global_cumsum: Cumsum across entire sequences (respects boundaries)

Both operations support:
    - Scalar (3D) and vector (4D) inputs
    - Forward and reverse directions
    - Softmax scaling
    - Variable-length sequences via cu_seqlens
    - Head-first or time-first tensor layouts

Typical Use Cases:
    - Gated Linear Attention (GLA): Computing cumulative gate products
    - Linear Attention: Accumulating key-value outer products
    - RetNet: Computing retention decay factors

Example:
    >>> from ejkernel.xla_utils import chunk_local_cumsum, chunk_global_cumsum
    >>>
    >>> # Local cumsum within 128-token chunks
    >>> local_cumsum = chunk_local_cumsum(g, chunk_size=128)
    >>>
    >>> # Global cumsum respecting sequence boundaries
    >>> global_cumsum = chunk_global_cumsum(s, cu_seqlens=cu_seqlens)
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import DTypeLike


@partial(jit, static_argnames=["chunk_size", "reverse", "head_first"])
def chunk_local_cumsum_scalar(
    g: jnp.ndarray,
    chunk_size: int,
    reverse: bool = False,
    softmax_scale: float | None = None,
    head_first: bool = False,
    output_dtype: DTypeLike | None = None,
) -> jnp.ndarray:
    """
    Compute local cumulative sum within chunks for 3D scalar inputs.

    Performs cumulative sum within fixed-size chunks, resetting at each
    chunk boundary. Supports forward and reverse directions. This is commonly
    used in linear attention mechanisms for computing cumulative gate products.

    Args:
        g: Input tensor of shape [B, H, T] if head_first else [B, T, H],
            where B is batch size, H is number of heads, T is sequence length.
        chunk_size: Size of each chunk (must be power of 2). The sequence is
            divided into T/chunk_size chunks.
        reverse: If True, compute reverse cumsum within each chunk (from right
            to left instead of left to right).
        softmax_scale: Optional scaling factor multiplied to the result.
        head_first: If True, expects [B, H, T] layout; otherwise [B, T, H].
        output_dtype: Optional output dtype (defaults to input dtype).

    Returns:
        Tensor with same shape as input containing chunked cumulative sums.
        Values reset to 0 at each chunk boundary.

    Examples:
        >>> import jax.numpy as jnp
        >>> from ejkernel.xla_utils.cumsum import chunk_local_cumsum_scalar
        >>>
        >>> # Input: [batch=1, time=8, heads=2]
        >>> g = jnp.ones((1, 8, 2))
        >>> result = chunk_local_cumsum_scalar(g, chunk_size=4, head_first=False)
        >>> # Within each chunk of 4, cumsum resets: [1,2,3,4,1,2,3,4]
    """
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size & (chunk_size - 1) == 0, "chunk_size must be a power of 2"
    output_dtype = output_dtype or g.dtype
    if not head_first:
        g = jnp.transpose(g, (0, 2, 1))
    T = g.shape[2]
    pad_length = (chunk_size - T % chunk_size) % chunk_size
    if pad_length > 0:
        g = jnp.pad(g, ((0, 0), (0, 0), (0, pad_length)), mode="constant")
    T_padded = T + pad_length
    num_chunks = T_padded // chunk_size
    g_chunked = g.reshape(B, H, num_chunks, chunk_size)
    if reverse:
        g_flipped = jnp.flip(g_chunked, axis=-1)
        cumsum_flipped = jnp.cumsum(g_flipped, axis=-1)
        result_chunked = jnp.flip(cumsum_flipped, axis=-1)
    else:
        result_chunked = jnp.cumsum(g_chunked, axis=-1)
    if softmax_scale is not None:
        result_chunked *= softmax_scale
    result = result_chunked.reshape(B, H, T_padded)
    if pad_length > 0:
        result = result[:, :, :T]
    if not head_first:
        result = jnp.transpose(result, (0, 2, 1))
    return result.astype(output_dtype)


@partial(jit, static_argnames=["chunk_size", "reverse", "head_first"])
def chunk_local_cumsum_vector(
    g: jnp.ndarray,
    chunk_size: int,
    reverse: bool = False,
    softmax_scale: float | None = None,
    head_first: bool = False,
    output_dtype: DTypeLike | None = None,
) -> jnp.ndarray:
    """
    Compute local cumulative sum within chunks for 4D vector inputs.

    Performs cumulative sum within fixed-size chunks for tensors with
    an additional state dimension. Used for accumulating vector states
    in linear attention variants like GLA.

    Args:
        g: Input tensor of shape [B, H, T, S] if head_first else [B, T, H, S],
            where B is batch size, H is number of heads, T is sequence length,
            and S is the state/feature dimension.
        chunk_size: Size of each chunk (must be power of 2). The sequence is
            divided into T/chunk_size chunks.
        reverse: If True, compute reverse cumsum within each chunk (from right
            to left instead of left to right).
        softmax_scale: Optional scaling factor multiplied to the result.
        head_first: If True, expects [B, H, T, S] layout; otherwise [B, T, H, S].
        output_dtype: Optional output dtype (defaults to input dtype).

    Returns:
        Tensor with same shape as input containing chunked cumulative sums
        along the time dimension. Values reset to 0 at each chunk boundary.

    Examples:
        >>> import jax.numpy as jnp
        >>> from ejkernel.xla_utils.cumsum import chunk_local_cumsum_vector
        >>>
        >>> # Input: [batch=1, time=8, heads=2, state=4]
        >>> g = jnp.ones((1, 8, 2, 4))
        >>> result = chunk_local_cumsum_vector(g, chunk_size=4, head_first=False)
        >>> # Cumsum along time within each chunk, preserving state dimension
    """
    if head_first:
        B, H, T, _S = g.shape
    else:
        B, T, H, _S = g.shape
    assert chunk_size & (chunk_size - 1) == 0, "chunk_size must be a power of 2"
    output_dtype = output_dtype or g.dtype
    if not head_first:
        g = jnp.transpose(g, (0, 2, 1, 3))
    T = g.shape[2]
    pad_length = (chunk_size - T % chunk_size) % chunk_size
    if pad_length > 0:
        g = jnp.pad(g, ((0, 0), (0, 0), (0, pad_length), (0, 0)), mode="constant")
    T_padded = T + pad_length
    num_chunks = T_padded // chunk_size
    g_chunked = g.reshape(B, H, num_chunks, chunk_size, g.shape[-1])
    if reverse:
        g_flipped = jnp.flip(g_chunked, axis=-2)
        cumsum_flipped = jnp.cumsum(g_flipped, axis=-2)
        result_chunked = jnp.flip(cumsum_flipped, axis=-2)
    else:
        result_chunked = jnp.cumsum(g_chunked, axis=-2)
    if softmax_scale is not None:
        result_chunked *= softmax_scale
    result = result_chunked.reshape(B, H, T_padded, g.shape[-1])
    if pad_length > 0:
        result = result[:, :, :T, :]
    if not head_first:
        result = jnp.transpose(result, (0, 2, 1, 3))
    return result.astype(output_dtype)


@partial(jit, static_argnames=["reverse", "head_first"])
def chunk_global_cumsum_scalar(
    s: jnp.ndarray,
    reverse: bool = False,
    cu_seqlens: jnp.ndarray | None = None,
    softmax_scale: float | None = None,
    head_first: bool = False,
    output_dtype: DTypeLike | None = None,
) -> jnp.ndarray:
    """
    Compute global cumulative sum across sequences for 3D scalar inputs.

    Performs cumulative sum across the entire sequence dimension, with optional
    support for variable-length sequences via cu_seqlens. When cu_seqlens is
    provided, the cumsum resets at sequence boundaries.

    Args:
        s: Input tensor of shape [B, H, T] if head_first else [B, T, H],
            where B is batch size, H is number of heads, T is sequence length.
        reverse: If True, compute reverse cumsum (right to left).
        cu_seqlens: Optional cumulative sequence lengths for packed sequences.
            When provided, cumsum resets at each sequence boundary to prevent
            accumulation across different sequences in the batch.
        softmax_scale: Optional scaling factor multiplied to the result.
        head_first: If True, expects [B, H, T] layout; otherwise [B, T, H].
        output_dtype: Optional output dtype (defaults to input dtype).

    Returns:
        Tensor with same shape as input containing global cumulative sums.
        When cu_seqlens is provided, sums reset at sequence boundaries.

    Examples:
        >>> import jax.numpy as jnp
        >>> from ejkernel.xla_utils.cumsum import chunk_global_cumsum_scalar
        >>>
        >>> # Simple global cumsum
        >>> s = jnp.ones((1, 8, 2))  # [batch, time, heads]
        >>> result = chunk_global_cumsum_scalar(s, head_first=False)
        >>> # Result: [[1,2,3,4,5,6,7,8], ...] along time dimension
        >>>
        >>> # With sequence boundaries (packed sequences)
        >>> cu_seqlens = jnp.array([0, 3, 8])  # Two sequences: len 3 and len 5
        >>> result = chunk_global_cumsum_scalar(s, cu_seqlens=cu_seqlens)
        >>> # Cumsum resets at position 3
    """
    output_dtype = output_dtype or s.dtype
    time_axis = 2 if head_first else 1

    if reverse:
        s_flipped = jnp.flip(s, axis=time_axis)
        result = jnp.cumsum(s_flipped, axis=time_axis)
    else:
        result = jnp.cumsum(s, axis=time_axis)

    if cu_seqlens is not None:
        boundary_indices = cu_seqlens[1:-1] - 1
        gather_indices_shape = [1] * s.ndim
        gather_indices_shape[time_axis] = len(boundary_indices)
        gather_indices = boundary_indices.reshape(gather_indices_shape)
        correction_values = jnp.take_along_axis(result, gather_indices, axis=time_axis)

        zero_pad_shape = list(correction_values.shape)
        zero_pad_shape[time_axis] = 1
        full_correction_map = jnp.concatenate(
            [jnp.zeros(zero_pad_shape, dtype=result.dtype), correction_values],
            axis=time_axis,
        )

        total_len = s.shape[time_axis]
        seq_ids = jnp.cumsum(jnp.zeros(total_len, dtype=jnp.int32).at[cu_seqlens[1:]].set(1))

        id_shape = [1] * s.ndim
        id_shape[time_axis] = total_len
        seq_ids = seq_ids.reshape(id_shape)

        correction_tensor = jnp.take_along_axis(full_correction_map, seq_ids, axis=time_axis)
        result -= correction_tensor

    if reverse:
        result = jnp.flip(result, axis=time_axis)

    if softmax_scale is not None:
        result *= softmax_scale

    return result.astype(output_dtype)


@partial(jit, static_argnames=["reverse", "head_first"])
def chunk_global_cumsum_vector(
    s: jnp.ndarray,
    reverse: bool = False,
    cu_seqlens: jnp.ndarray | None = None,
    softmax_scale: float | None = None,
    head_first: bool = False,
    output_dtype: DTypeLike | None = None,
) -> jnp.ndarray:
    """
    Compute global cumulative sum across sequences for 4D vector inputs.

    Performs cumulative sum across the entire sequence dimension for tensors
    with an additional state/feature dimension. Supports variable-length
    sequences via cu_seqlens for packed sequence processing.

    Args:
        s: Input tensor of shape [B, H, T, S] if head_first else [B, T, H, S],
            where B is batch size, H is number of heads, T is sequence length,
            and S is the state/feature dimension.
        reverse: If True, compute reverse cumsum (right to left).
        cu_seqlens: Optional cumulative sequence lengths for packed sequences.
            When provided, cumsum resets at each sequence boundary.
        softmax_scale: Optional scaling factor multiplied to the result.
        head_first: If True, expects [B, H, T, S] layout; otherwise [B, T, H, S].
        output_dtype: Optional output dtype (defaults to input dtype).

    Returns:
        Tensor with same shape as input containing global cumulative sums
        along the time dimension. When cu_seqlens is provided, sums reset
        at sequence boundaries.

    Examples:
        >>> import jax.numpy as jnp
        >>> from ejkernel.xla_utils.cumsum import chunk_global_cumsum_vector
        >>>
        >>> # Global cumsum with state dimension
        >>> s = jnp.ones((1, 8, 2, 4))  # [batch, time, heads, state]
        >>> result = chunk_global_cumsum_vector(s, head_first=False)
        >>> # Cumsum along time, preserving state dimension
    """
    output_dtype = output_dtype or s.dtype
    time_axis = 2 if head_first else 1

    if reverse:
        s_flipped = jnp.flip(s, axis=time_axis)
        result = jnp.cumsum(s_flipped, axis=time_axis)
    else:
        result = jnp.cumsum(s, axis=time_axis)

    if cu_seqlens is not None:
        boundary_indices = cu_seqlens[1:-1] - 1

        gather_indices_shape = [1] * s.ndim
        gather_indices_shape[time_axis] = len(boundary_indices)
        gather_indices = boundary_indices.reshape(gather_indices_shape)

        correction_values = jnp.take_along_axis(result, gather_indices, axis=time_axis)

        zero_pad_shape = list(correction_values.shape)
        zero_pad_shape[time_axis] = 1
        full_correction_map = jnp.concatenate(
            [jnp.zeros(zero_pad_shape, dtype=result.dtype), correction_values],
            axis=time_axis,
        )

        total_len = s.shape[time_axis]
        seq_ids = jnp.cumsum(jnp.zeros(total_len, dtype=jnp.int32).at[cu_seqlens[1:]].set(1))

        id_shape = [1] * s.ndim
        id_shape[time_axis] = total_len
        seq_ids = seq_ids.reshape(id_shape)

        correction_tensor = jnp.take_along_axis(full_correction_map, seq_ids, axis=time_axis)
        result -= correction_tensor

    if reverse:
        result = jnp.flip(result, axis=time_axis)

    if softmax_scale is not None:
        result *= softmax_scale

    return result.astype(output_dtype)


@partial(
    jit,
    static_argnames=[
        "chunk_size",
        "reverse",
        "softmax_scale",
        "head_first",
        "output_dtype",
        "is_vector",
    ],
)
def _chunk_local_cumsum_vmap_core(
    g_padded_batched: jnp.ndarray,
    mask: jnp.ndarray,
    chunk_size: int,
    reverse: bool,
    softmax_scale: float | None,
    head_first: bool,
    output_dtype: DTypeLike | None,
    is_vector: bool,
) -> jnp.ndarray:
    """
    Internal vmapped implementation of chunked local cumsum for variable-length sequences.

    This is an internal helper function that applies chunk_local_cumsum across
    multiple padded sequences using vmap, then masks out padded positions.

    Args:
        g_padded_batched: Padded input tensor with an extra sequence batch dimension.
        mask: Boolean mask indicating valid (non-padded) positions.
        chunk_size: Size of each chunk for local cumsum.
        reverse: If True, compute reverse cumsum within each chunk.
        softmax_scale: Optional scaling factor applied to result.
        head_first: If True, expects head dimension before time dimension.
        output_dtype: Optional output dtype.
        is_vector: If True, use vector (4D) cumsum; otherwise use scalar (3D).

    Returns:
        Tensor with chunked cumulative sums, masked to zero for padded positions.
    """
    base_fn = chunk_local_cumsum_vector if is_vector else chunk_local_cumsum_scalar
    vmapped_fn = vmap(base_fn, in_axes=(0, None, None, None, None, None), out_axes=0)
    result_padded = vmapped_fn(g_padded_batched, chunk_size, reverse, softmax_scale, head_first, output_dtype)
    return result_padded * mask


def chunk_local_cumsum(
    g: jnp.ndarray,
    chunk_size: int,
    reverse: bool = False,
    softmax_scale: float | None = None,
    cu_seqlens: jnp.ndarray | None = None,
    head_first: bool = False,
    output_dtype: DTypeLike | None = None,
    **kwargs,
) -> jnp.ndarray:
    """
    Compute local cumulative sum within fixed-size chunks.

    Main entry point for chunked local cumulative sum, automatically
    dispatching to scalar (3D) or vector (4D) implementations based on input
    rank. The cumulative sum resets at each chunk boundary, making this
    suitable for chunked/tiled attention implementations.

    Args:
        g: Input tensor of shape [B, T, H] (3D scalar) or [B, T, H, S] (4D vector),
            where B is batch size, T is sequence length, H is number of heads,
            and S is optional state dimension.
        chunk_size: Size of each chunk (must be power of 2). The cumulative sum
            resets at every chunk_size positions.
        reverse: If True, compute reverse cumsum within each chunk (accumulate
            from right to left instead of left to right).
        softmax_scale: Optional scaling factor multiplied to the result.
        cu_seqlens: Optional cumulative sequence lengths for packed sequences.
            When provided, enables processing of variable-length sequences
            within a single batch.
        head_first: If True, expects [B, H, T, ...] layout instead of [B, T, H, ...].
        output_dtype: Optional output dtype (defaults to input dtype).
        **kwargs: Additional keyword arguments (ignored for compatibility).

    Returns:
        Tensor with same shape as input containing chunked cumulative sums.
        Values reset to 0 at each chunk boundary.

    Raises:
        AssertionError: If cu_seqlens is provided and batch size is not 1.

    Note:
        When cu_seqlens is provided, only batch size 1 is supported.
        The function handles variable-length sequences by padding and masking.

    Examples:
        >>> import jax.numpy as jnp
        >>> from ejkernel.xla_utils import chunk_local_cumsum
        >>>
        >>> # Basic usage with 3D input
        >>> g = jnp.ones((2, 128, 8))  # [batch, time, heads]
        >>> result = chunk_local_cumsum(g, chunk_size=32)
        >>> # Cumsum resets every 32 positions
        >>>
        >>> # With variable-length sequences
        >>> g = jnp.ones((1, 100, 8))  # Must be batch_size=1
        >>> cu_seqlens = jnp.array([0, 40, 100])  # Two sequences
        >>> result = chunk_local_cumsum(g, chunk_size=32, cu_seqlens=cu_seqlens)
    """
    is_vector = g.ndim == 4
    base_fn = chunk_local_cumsum_vector if is_vector else chunk_local_cumsum_scalar
    if cu_seqlens is None:
        return base_fn(g, chunk_size, reverse, softmax_scale, None, head_first, output_dtype)
    assert g.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    seqlens = jnp.diff(cu_seqlens)
    max_seq_len = jnp.max(seqlens)
    num_seqs = len(seqlens)
    mask_indices = jnp.arange(max_seq_len) < seqlens[:, None]
    squeezed_g = g.squeeze(0)
    other_dims = squeezed_g.shape[1:]
    padded_g = jnp.zeros((num_seqs, max_seq_len, *other_dims), dtype=g.dtype)

    def create_padded_batch(i, _):
        start, length = cu_seqlens[i], seqlens[i]
        seq_slice = jax.lax.dynamic_slice(squeezed_g, (start,) + (0,) * len(other_dims), (length, *other_dims))
        return jax.lax.dynamic_update_slice(padded_g[i], seq_slice, (0,) * (len(other_dims) + 1))

    g_padded_batched = jnp.stack([create_padded_batch(i, None) for i in range(num_seqs)], axis=0)
    g_padded_batched = jnp.expand_dims(g_padded_batched, axis=0)
    mask_shape = (num_seqs, max_seq_len) + (1,) * len(other_dims)
    mask = mask_indices.reshape(mask_shape)
    result_padded = _chunk_local_cumsum_vmap_core(
        g_padded_batched,
        mask,
        chunk_size,
        reverse,
        softmax_scale,
        head_first,
        output_dtype,
        is_vector,
    )
    result_flat = result_padded.reshape(-1, *result_padded.shape[-(len(other_dims)) :])
    final_result = result_flat[mask_indices.flatten()]
    return final_result[None, ...]


def chunk_global_cumsum(
    s: jnp.ndarray,
    reverse: bool = False,
    cu_seqlens: jnp.ndarray | None = None,
    softmax_scale: float | None = None,
    head_first: bool = False,
    output_dtype: DTypeLike | None = None,
) -> jnp.ndarray:
    """
    Compute global cumulative sum across sequences.

    Main entry point for global cumulative sum, automatically dispatching
    to scalar (3D) or vector (4D) implementations based on input rank.
    Unlike chunk_local_cumsum, this computes cumsum across the entire
    sequence rather than within fixed-size chunks.

    Args:
        s: Input tensor of shape [B, T, H] (3D scalar) or [B, T, H, S] (4D vector),
            where B is batch size, T is sequence length, H is number of heads,
            and S is optional state dimension.
        reverse: If True, compute reverse cumsum (accumulate from right to left).
        cu_seqlens: Optional cumulative sequence lengths for packed sequences.
            When provided, cumsum resets at sequence boundaries to prevent
            accumulation across different sequences in the batch.
        softmax_scale: Optional scaling factor multiplied to the result.
        head_first: If True, expects [B, H, T, ...] layout instead of [B, T, H, ...].
        output_dtype: Optional output dtype (defaults to input dtype).

    Returns:
        Tensor with same shape as input containing global cumulative sums.
        When cu_seqlens is provided, sums reset at sequence boundaries.

    Note:
        With cu_seqlens, the cumulative sum respects sequence boundaries
        and does not accumulate across different sequences in the batch.

    Examples:
        >>> import jax.numpy as jnp
        >>> from ejkernel.xla_utils import chunk_global_cumsum
        >>>
        >>> # Simple global cumsum
        >>> s = jnp.ones((2, 128, 8))  # [batch, time, heads]
        >>> result = chunk_global_cumsum(s)
        >>> # result[0, i, :] = i+1 for each position i
        >>>
        >>> # With packed sequences (cumsum resets at boundaries)
        >>> s = jnp.ones((1, 100, 8))
        >>> cu_seqlens = jnp.array([0, 40, 100])  # Two sequences
        >>> result = chunk_global_cumsum(s, cu_seqlens=cu_seqlens)
        >>> # Cumsum resets at position 40
    """
    is_vector = s.ndim == 4
    if is_vector:
        return chunk_global_cumsum_vector(s, reverse, cu_seqlens, softmax_scale, head_first, output_dtype)
    else:
        return chunk_global_cumsum_scalar(s, reverse, cu_seqlens, softmax_scale, head_first, output_dtype)
