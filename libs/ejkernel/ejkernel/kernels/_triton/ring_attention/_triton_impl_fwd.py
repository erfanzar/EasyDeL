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

"""Ring Attention Implementation using Triton Block-sparse Attention.

This module provides a ring attention implementation that wraps the Triton
block-sparse attention kernel for distributed execution across multiple GPU devices.

Ring attention is a distributed attention mechanism that enables processing of
extremely long sequences by partitioning queries across devices in a ring topology.
Each device holds its local query partition and rotates key/value blocks through
all devices, computing partial attention and combining results using online softmax.

Key features:
- Uses block-sparse attention as the inner kernel for optimized GPU execution
- Supports distributed ring topology via lax.ppermute for cross-device communication
- Supports causal and sliding-window masking via explicit positions/segment IDs
- Full backward pass support with custom VJP for gradient computation
- Memory efficient: Only holds local query partition and one remote KV block at a time

Algorithm overview:
1. Each device starts with its local query partition and local KV block
2. In each ring step:
   - Compute attention between local queries and current KV block
   - Combine with running output using online softmax (log-sum-exp accumulation)
   - Send KV block to next device, receive KV block from previous device
3. After N steps (N = number of devices), each device has computed full attention

Supported features:
- Causal and non-causal attention
- Sliding window attention for local context patterns
- Segment-based masking via segment IDs (for packed sequences)
- Custom position IDs for flexible position encoding
- Logits soft capping for numerical stability
- Attention sinks via softmax_aux parameter

Example:
    >>> import jax.numpy as jnp
    >>> from jax.experimental import mesh_utils
    >>> from jax.sharding import Mesh, PartitionSpec as P
    >>> from ejkernel.kernels._triton.ring_attention import ring_attention
    >>>
    >>> # Create mesh for sequence parallelism
    >>> devices = mesh_utils.create_device_mesh((4,))
    >>> mesh = Mesh(devices, axis_names=("sp",))
    >>>
    >>> batch, seq_len, num_heads, head_dim = 2, 8192, 12, 64
    >>> q = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>>
    >>> # Shard along sequence dimension
    >>> with mesh:
    ...     output = ring_attention(q, k, v, causal=True, axis_name="sp")

Reference:
    Ring Attention with Blockwise Transformers for Near-Infinite Context
    https://arxiv.org/abs/2310.01889
"""

from __future__ import annotations

import typing
from collections.abc import Callable

from jax import Array
from jaxtyping import Float, Int

from ejkernel.ops import BwdParams, FwdParams

from ._triton_impl_bwd import ring_blocksparse_attention_call

if typing.TYPE_CHECKING:
    from ejkernel.kernels._pallas.tpu.blocksparse_attention._masks import Mask


def ring_attention(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
    q_position_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_position_ids: Int[Array, "batch seq_len_k"] | None = None,
    softmax_aux: Float[Array, "num_sinks"] | None = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    mask_builder: Callable[[int, int, int, int, int], Mask] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    chunk_size: int | None = None,
    causal: bool = False,
    logits_soft_cap: float | None = None,
    softmax_scale: float | None = None,
    axis_name: str | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    fused_backward: bool = False,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """Ring attention using Triton block-sparse attention kernels.

    Distributes attention computation across devices using a ring topology,
    where each device holds its query partition and rotates key/value blocks
    through all devices, computing partial attention and combining results
    using online softmax.

    This implementation wraps the block-sparse attention kernel for efficient
    GPU execution while handling the distributed communication pattern. It
    supports various masking strategies through segment IDs and position IDs.

    Args:
        query: Query tensor of shape [batch, seq_len_q, num_heads, head_dim].
        key: Key tensor of shape [batch, seq_len_k, num_kv_heads, head_dim].
        value: Value tensor of shape [batch, seq_len_k, num_kv_heads, head_dim].
        q_segment_ids: Optional segment IDs for queries [batch, seq_len_q].
            Used for packed sequence support where different segments should
            not attend to each other. Tokens with different segment IDs are masked.
        kv_segment_ids: Optional segment IDs for keys/values [batch, seq_len_k].
            Must match q_segment_ids semantics for cross-attention masking.
        q_position_ids: Optional position indices for queries [batch, seq_len_q].
            Used to determine relative positions for causal/sliding window masking.
            If None, defaults to sequential positions 0..seq_len_q-1.
        kv_position_ids: Optional position indices for keys/values [batch, seq_len_k].
            Used to determine relative positions for causal/sliding window masking.
            If None, defaults to sequential positions 0..seq_len_k-1.
        softmax_aux: Optional attention sink logits of shape [num_sinks].
            Contributes to the softmax normalizer without producing output values,
            useful for streaming attention or attention sinks.
        bias: Optional attention bias tensor [batch, num_heads, seq_len_q, seq_len_k].
            Note: Currently not supported in Triton ring_attention (raises NotImplementedError).
        mask_builder: Optional callable to build custom sparse mask patterns.
            Currently unused in this implementation.
        sliding_window: Sliding window size for local attention. Can be:
            - int: symmetric window (same size left and right)
            - tuple[int, int]: (left_window, right_window) for asymmetric
            - None: no sliding window (full attention)
        chunk_size: Optional chunk size for block processing.
            Currently unused in this implementation.
        causal: Whether to apply causal masking (default: False). When True,
            queries can only attend to keys at the same or earlier positions.
        logits_soft_cap: Optional soft cap value for attention logits. When specified,
            applies tanh-based soft capping: logits_soft_cap * tanh(logits / logits_soft_cap).
            Helps with numerical stability (e.g., Gemma-2 uses 20.0).
        softmax_scale: Attention score scaling factor. If None, defaults to
            1/sqrt(head_dim).
        axis_name: Name of the axis for ring communication. If None, runs on a
            single device without distribution. Must match the mesh axis name
            used for sequence parallelism.
        fwd_params: Forward pass kernel parameters (block sizes, warps, stages).
            If None, uses default FwdParams(q_blocksize=128, kv_blocksize=128).
        bwd_params: Backward pass kernel parameters (block sizes, warps, stages).
            If None, uses default BwdParams(q_blocksize=128, kv_blocksize=128).
        fused_backward: Whether to use fused backward pass. Currently unused.

    Returns:
        Attention output tensor of shape [batch, seq_len_q, num_heads, head_dim].

    Raises:
        NotImplementedError: If bias is provided (currently not supported with
            block-sparse attention backend).

    Example:
        >>> import jax.numpy as jnp
        >>> from ejkernel.kernels._triton.ring_attention import ring_attention
        >>>
        >>> # Basic causal ring attention
        >>> output = ring_attention(q, k, v, causal=True, axis_name="sp")

        >>> # With sliding window
        >>> output = ring_attention(
        ...     q, k, v,
        ...     sliding_window=256,
        ...     causal=True,
        ...     axis_name="sp",
        ... )

        >>> # With segment IDs for packed sequences
        >>> q_seg = jnp.array([[1, 1, 1, 2, 2, 2, 2, 2]])  # Two sequences packed
        >>> kv_seg = jnp.array([[1, 1, 1, 2, 2, 2, 2, 2]])
        >>> output = ring_attention(
        ...     q, k, v,
        ...     q_segment_ids=q_seg,
        ...     kv_segment_ids=kv_seg,
        ...     causal=True,
        ...     axis_name="sp",
        ... )
    """
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    if fwd_params is None:
        fwd_params = FwdParams(q_blocksize=128, kv_blocksize=128, num_stages=2, num_warps=4)
    if bwd_params is None:
        bwd_params = BwdParams(q_blocksize=128, kv_blocksize=128, num_stages=2, num_warps=4)

    if bias is not None:
        raise NotImplementedError("Triton ring_attention does not currently support `bias` when using blocksparse.")

    del mask_builder, fused_backward

    return ring_blocksparse_attention_call(
        query=query,
        key=key,
        value=value,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        q_position_ids=q_position_ids,
        kv_position_ids=kv_position_ids,
        softmax_aux=softmax_aux,
        softmax_scale=softmax_scale,
        causal=causal,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        axis_name=axis_name,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
    )
