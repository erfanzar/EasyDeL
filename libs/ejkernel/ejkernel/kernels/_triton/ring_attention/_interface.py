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
"""Kernel registration wrapper for ring attention (Triton GPU backend).

Exposes ``ring_attention`` via the ejkernel registry under
``Platform.TRITON / Backend.GPU``.  The implementation delegates to
``ring_blocksparse_attention_call`` in ``_triton_impl_bwd``, which uses
Triton block-sparse attention with explicit position/segment-ID masking
distributed across devices in a ring topology.

Public symbols:
    ring_attention: Registered kernel entry point.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype

from ejkernel.kernels._pallas.tpu.blocksparse_attention._masks import Mask

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_fwd import Array, BwdParams, Callable, Float, FwdParams, Int
from ._triton_impl_fwd import ring_attention as _ring_attention_impl


@kernel_registry.register("ring_attention", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
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
    return _ring_attention_impl(
        query,
        key,
        value,
        q_segment_ids,
        kv_segment_ids,
        q_position_ids,
        kv_position_ids,
        softmax_aux,
        bias,
        mask_builder,
        sliding_window,
        chunk_size,
        causal,
        logits_soft_cap,
        softmax_scale,
        axis_name,
        fwd_params,
        bwd_params,
        fused_backward,
    )


__all__ = ("ring_attention",)
