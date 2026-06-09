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
"""Public interface and kernel-registry wrapper for XLA block-sparse attention.

Registers ``blocksparse_attention`` under ``Platform.XLA / Backend.ANY``.
This backend materialises the full token-level attention mask and delegates to
the dense XLA attention kernel, serving as the correctness reference for the
Pallas (TPU) and Triton (GPU) sparse implementations.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import Array, Bool, BwdParams, Callable, Float, FwdParams, Int
from ._xla_impl_fwd import blocksparse_attention as _blocksparse_attention_impl


@kernel_registry.register("blocksparse_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def blocksparse_attention(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch kv_num_heads kv_len head_dim"],
    value: Float[Array, "batch kv_num_heads kv_len vhead_dim"],
    q_segment_ids: Int[Array, "batch seq_len"] | None = None,
    kv_segment_ids: Int[Array, "batch kv_len"] | None = None,
    q_positions: Int[Array, "batch seq_len"] | None = None,
    kv_positions: Int[Array, "batch kv_len"] | None = None,
    softmax_aux: Float[Array, "num_sinks"] | None = None,
    bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
    attention_mask: (
        Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | Int[Array, "batch num_heads_or_1 seq_len kv_len"] | None
    ) = None,
    sequence_parallelism_mesh_axis_name: str | None = None,
    logits_soft_cap: float | None = None,
    qkv_layouts: tuple["SparseMask"] | None = None,
    softmax_scale: float | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    mask_builder: Callable[[int, int, int, int, int], "Mask"] | Callable[[], "SparseMask"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    chunk_size: int | None = None,
    causal: bool = True,
    fused_backward: bool = False,
) -> Float[Array, "batch num_heads seq_len vhead_dim"]:
    """XLA fallback for block-sparse attention with packed sequence support.
    This implementation serves as a correctness reference for block-sparse attention.
    It materializes the full token-level mask implied by segment IDs, positions,
    causal/sliding-window settings, and computes dense attention using JAX/XLA.
    The masking logic combines:
    1. Segment masking: Tokens can only attend within the same segment
    2. Causal masking: Each position attends only to earlier positions
    3. Sliding window: Optional local attention within a window
    4. Attention mask: Optional explicit attention pattern
    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim].
            Note: Uses BHSD layout (heads before sequence).
        key: Key tensor [batch, num_kv_heads, kv_len, head_dim].
        value: Value tensor [batch, num_kv_heads, kv_len, head_dim].
        q_segment_ids: Query segment IDs [batch, seq_len].
            Identifies which sequence each query token belongs to.
        kv_segment_ids: Key/value segment IDs [batch, kv_len].
            If None and q_segment_ids is provided with matching length, uses q_segment_ids.
        q_positions: Query position IDs [batch, seq_len].
            Position within each segment for causal/window masking.
        kv_positions: Key/value position IDs [batch, kv_len].
        softmax_aux: Optional attention sink logits [num_sinks].
            Participate in softmax but don't contribute to output.
        bias: Optional attention bias [batch, num_heads, seq_len, kv_len].
            NOT SUPPORTED in XLA fallback.
        attention_mask: Optional additional attention mask.
            Combined with segment/position-based masking.
        sequence_parallelism_mesh_axis_name: Unused in XLA backend.
        logits_soft_cap: Soft cap for attention logits via tanh.
        qkv_layouts: Unused in XLA backend.
        softmax_scale: Scaling factor for QK^T. Defaults to 1/sqrt(head_dim).
        fwd_params: Unused in XLA backend.
        bwd_params: Unused in XLA backend.
        mask_builder: Unused in XLA backend.
        sliding_window: Optional local attention window. Can be:
            - int: Symmetric window (same left and right)
            - tuple[int, int]: Asymmetric (left_window, right_window)
        chunk_size: Unused in XLA backend.
        causal: If True, applies causal masking based on positions. Default True.
        fused_backward: Unused in XLA backend.
    Returns:
        Attention output [batch, num_heads, seq_len, head_dim].
    Raises:
        NotImplementedError: If bias is provided (not supported in fallback).
        ValueError: If input tensor ranks are not 4, or if batch/head dimensions mismatch.
    Example:
        >>> import jax.numpy as jnp
        >>>
        >>> batch, num_heads, seq_len, head_dim = 1, 8, 256, 64
        >>> q = jnp.ones((batch, num_heads, seq_len, head_dim))
        >>> k = jnp.ones((batch, num_heads, seq_len, head_dim))
        >>> v = jnp.ones((batch, num_heads, seq_len, head_dim))
        >>>
        >>> # Segment IDs for two sequences of length 128 each
        >>> seg_ids = jnp.concatenate([
        ...     jnp.zeros((batch, 128), dtype=jnp.int32),
        ...     jnp.ones((batch, 128), dtype=jnp.int32)
        ... ], axis=1)
        >>>
        >>> output = blocksparse_attention(q, k, v, q_segment_ids=seg_ids, causal=True)
        >>> output.shape
        (1, 8, 256, 64)
    Note:
        This fallback materializes O(seq_len²) mask, which is memory-intensive.
        For production, use Pallas (TPU) or Triton (GPU) implementations.
    """
    return _blocksparse_attention_impl(
        query,
        key,
        value,
        q_segment_ids,
        kv_segment_ids,
        q_positions,
        kv_positions,
        softmax_aux,
        bias,
        attention_mask,
        sequence_parallelism_mesh_axis_name,
        logits_soft_cap,
        qkv_layouts,
        softmax_scale,
        fwd_params,
        bwd_params,
        mask_builder,
        sliding_window,
        chunk_size,
        causal,
        fused_backward,
    )


__all__ = ("blocksparse_attention",)
