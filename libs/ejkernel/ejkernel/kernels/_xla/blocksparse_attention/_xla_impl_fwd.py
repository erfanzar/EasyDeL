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

"""Block-sparse attention interface for XLA fallback computation.

This module provides a reference implementation of block-sparse attention
that handles packed multi-sequence inputs with segment IDs and positions.
It serves as a correctness fallback when specialized kernels (Pallas TPU,
Triton GPU) are unavailable.

Block-sparse attention is designed for efficient processing of packed
sequences where multiple variable-length sequences are concatenated into
a single tensor. The attention pattern is determined by:
1. Segment IDs: Prevent cross-sequence attention
2. Position IDs: Enable proper causal/sliding window masking within segments
3. Optional attention masks: Additional masking patterns

This XLA fallback materializes the full token-level mask implied by these
constraints and computes dense attention, which is correct but may use
more memory than specialized sparse implementations.

Key features:
1. Packed sequence support: Handle multiple sequences in one tensor
2. Segment-aware masking: Prevent attention across sequence boundaries
3. Position-based masking: Correct causal/window masks for packed sequences
4. GQA/MQA support: Flexible head configurations
5. Attention sinks: Learnable auxiliary logits for probability mass absorption

Use cases:
- Inference with dynamic batching (vLLM-style)
- Training on packed sequences for efficiency
- Variable-length sequence processing
- Correctness verification for sparse implementations

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._xla.blocksparse_attention import blocksparse_attention
    >>>
    >>> # Two packed sequences: lengths 100 and 150
    >>> batch, num_heads, total_len, head_dim = 1, 8, 250, 64
    >>> q = jnp.ones((batch, num_heads, total_len, head_dim))
    >>> k = jnp.ones((batch, num_heads, total_len, head_dim))
    >>> v = jnp.ones((batch, num_heads, total_len, head_dim))
    >>>
    >>> # Segment IDs: first 100 tokens are seq 0, next 150 are seq 1
    >>> segment_ids = jnp.concatenate([
    ...     jnp.zeros((batch, 100), dtype=jnp.int32),
    ...     jnp.ones((batch, 150), dtype=jnp.int32)
    ... ], axis=1)
    >>>
    >>> output = blocksparse_attention(
    ...     q, k, v,
    ...     q_segment_ids=segment_ids,
    ...     kv_segment_ids=segment_ids,
    ...     causal=True
    ... )

Note:
    This is a correctness fallback that materializes the full attention mask.
    For production use on TPU, prefer the Pallas blocksparse_attention kernel.
    For GPU, prefer the Triton implementation.
"""

from __future__ import annotations

import typing as tp

import jax
from beartype.typing import Callable
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ejkernel.ops import BwdParams, FwdParams

from ..attention import attention as dense_attention

if tp.TYPE_CHECKING:
    from ejkernel.kernels._pallas.tpu.blocksparse_attention._masks import Mask
    from ejkernel.kernels._triton.blocksparse_attention._mask import SparseMask


def _normalize_segment_ids(ids: Int[Array, "..."] | None, *, which: str) -> Int[Array, "batch seqlen"] | None:
    """Normalize segment IDs to 2D [batch, seqlen] format.

    Segment IDs identify which sequence each token belongs to in a packed
    representation. Tokens with the same segment ID can attend to each other.

    Args:
        ids: Segment IDs in one of these formats:
            - None: No segment masking
            - 2D [batch, seqlen]: Standard format (returned as-is)
            - 3D [batch, heads, seqlen]: Per-head format (uses head 0)
        which: String identifier ("q" or "kv") for error messages.

    Returns:
        Normalized segment IDs with shape [batch, seqlen], or None.

    Raises:
        ValueError: If ids has unsupported rank.
    """
    if ids is None:
        return None
    ids = jnp.asarray(ids, jnp.int32)
    if ids.ndim == 2:
        return ids
    if ids.ndim == 3:
        return ids[:, 0, :]
    raise ValueError(f"{which}_segment_ids must be 2D or 3D, got shape {ids.shape}")


def _normalize_positions(
    pos: Int[Array, "..."] | None, *, batch: int, seqlen: int, fill: int
) -> Int[Array, "batch seqlen"]:
    """Normalize position IDs to 2D [batch, seqlen] format.

    Position IDs are used for causal and sliding window masking within
    packed sequences. Each token's position determines what it can attend to.

    Args:
        pos: Position IDs with shape [batch, seqlen], or None for default
            sequential positions (0, 1, 2, ..., seqlen-1).
        batch: Batch size for output shape.
        seqlen: Sequence length for output shape.
        fill: Value to use for NaN positions (for floating-point inputs).

    Returns:
        Position IDs with shape [batch, seqlen]. If input is None, returns
        sequential positions.

    Raises:
        ValueError: If pos has incorrect shape.
    """
    if pos is None:
        return jnp.broadcast_to(jnp.arange(seqlen, dtype=jnp.int32)[None, :], (batch, seqlen))
    pos = jnp.asarray(pos, jnp.int32)
    if pos.shape != (batch, seqlen):
        raise ValueError(f"positions must have shape {(batch, seqlen)}, got {pos.shape}")
    return jnp.where(jnp.isnan(pos), fill, pos).astype(jnp.int32) if jnp.issubdtype(pos.dtype, jnp.floating) else pos


def _normalize_attention_mask(
    attention_mask: Bool[Array, "..."] | Int[Array, "..."] | None,
    *,
    batch: int,
    q_len: int,
    kv_len: int,
) -> Bool[Array, "batch q kv"] | None:
    """Normalize attention mask to 3D [batch, q_len, kv_len] boolean format.

    Converts various attention mask formats to a consistent 3D boolean tensor
    that can be combined with segment-based masking.

    Args:
        attention_mask: Attention mask in one of these formats:
            - None: No additional masking
            - 2D [batch, kv_len]: KV padding mask (broadcast to all query positions)
            - 3D [batch, q_len, kv_len]: Standard attention mask
            - 4D [batch, heads, q_len, kv_len]: Per-head mask (uses head 0)
        batch: Expected batch size for validation.
        q_len: Expected query length for validation.
        kv_len: Expected key/value length for validation.

    Returns:
        Boolean attention mask with shape [batch, q_len, kv_len], or None.
        True values indicate positions that CAN be attended to.

    Raises:
        ValueError: If mask has unsupported shape or dimensions don't match.
    """
    if attention_mask is None:
        return None
    m = attention_mask
    if m.dtype != jnp.bool_:
        m = m != 0

    if m.ndim == 4:
        if m.shape[0] != batch or m.shape[2] != q_len or m.shape[3] != kv_len:
            raise ValueError(f"attention_mask must have shape (B, H/1, Q, K); got {m.shape}")
        return m[:, 0, :, :]
    if m.ndim == 3:
        if m.shape != (batch, q_len, kv_len):
            raise ValueError(f"attention_mask must have shape (B, Q, K); got {m.shape}")
        return m
    if m.ndim == 2:
        if m.shape != (batch, kv_len):
            raise ValueError(f"2D attention_mask is treated as KV padding mask with shape (B, K); got {m.shape}")
        return jnp.broadcast_to(m[:, None, :], (batch, q_len, kv_len))

    raise ValueError(f"Unsupported attention_mask rank {m.ndim} with shape {m.shape}")


def _normalize_softmax_aux(
    softmax_aux: Float[Array, "..."] | None,
    *,
    num_heads: int,
    num_kv_heads: int,
    dtype: jnp.dtype,
) -> Float[Array, "num_heads num_sinks"] | None:
    """Normalize softmax_aux into per-head sink logits for block-sparse attention.

    Attention sinks are auxiliary logits that participate in softmax normalization
    but don't contribute to the output. They allow the model to "dump" probability
    mass, which is useful for numerical stability and StreamingLLM-style patterns.

    Args:
        softmax_aux: Attention sink logits in one of these formats:
            - None: No attention sinks
            - 1D [num_heads]: One sink per head
            - 1D [num_kv_heads]: One sink per KV head (expanded for GQA)
            - 1D [num_sinks]: Shared sinks across all heads
            - 2D [num_heads, num_sinks]: Per-head multiple sinks
            - 2D [num_kv_heads, num_sinks]: Per-KV-head sinks (expanded)
        num_heads: Total number of query heads.
        num_kv_heads: Number of key/value heads (may be fewer for GQA).
        dtype: Target dtype for the output tensor.

    Returns:
        Normalized sink logits with shape [num_heads, num_sinks], or None.

    Raises:
        ValueError: If softmax_aux has incompatible shape or rank > 2.
    """
    if softmax_aux is None:
        return None
    aux = jnp.asarray(softmax_aux, dtype=dtype)
    if aux.ndim == 1:
        if aux.shape[0] == num_heads:
            return aux[:, None]
        if aux.shape[0] == num_kv_heads:
            reps = num_heads // num_kv_heads
            return jnp.repeat(aux, repeats=reps, axis=0)[:, None]
        return jnp.broadcast_to(aux[None, :], (num_heads, aux.shape[0]))
    if aux.ndim == 2:
        if aux.shape[0] == num_heads:
            return aux
        if aux.shape[0] == num_kv_heads:
            reps = num_heads // num_kv_heads
            return jnp.repeat(aux, repeats=reps, axis=0)
        raise ValueError(
            f"softmax_aux first dim must be num_kv_heads ({num_kv_heads}) or num_heads ({num_heads}); got {aux.shape[0]}"
        )
    raise ValueError(f"softmax_aux must be 1D or 2D, got shape {aux.shape}")


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
    del (
        fused_backward,
        qkv_layouts,
        fwd_params,
        bwd_params,
        mask_builder,
        chunk_size,
        sequence_parallelism_mesh_axis_name,
    )

    if bias is not None:
        raise NotImplementedError("Bias is not supported in blocksparse_attention (XLA fallback)")

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be rank-4 tensors (B, H, T, D)")

    batch, num_heads, q_len, head_dim = query.shape
    _b2, num_kv_heads, kv_len, _d2 = key.shape
    if _b2 != batch:
        raise ValueError(f"batch mismatch: query batch {batch}, key batch {_b2}")
    if value.shape[:3] != (batch, num_kv_heads, kv_len):
        raise ValueError(f"value must have shape (B, Hkv, K, Vd); got {value.shape}")
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")

    if softmax_scale is None:
        softmax_scale = head_dim**-0.5

    if sliding_window is None:
        window_left = window_right = None
    elif isinstance(sliding_window, int):
        window_left = window_right = int(sliding_window)
    else:
        window_left, window_right = int(sliding_window[0]), int(sliding_window[1])

    q_ids = _normalize_segment_ids(q_segment_ids, which="q")
    kv_ids = _normalize_segment_ids(kv_segment_ids, which="kv")

    if kv_ids is None and q_ids is not None and kv_len == q_len:
        kv_ids = q_ids
    if q_ids is None and kv_ids is not None and kv_len == q_len:
        q_ids = kv_ids

    if q_ids is None:
        q_ids = jnp.ones((batch, q_len), dtype=jnp.int32)
    if kv_ids is None:
        kv_ids = jnp.ones((batch, kv_len), dtype=jnp.int32)

    q_pos = _normalize_positions(q_positions, batch=batch, seqlen=q_len, fill=-1)
    kv_pos = _normalize_positions(kv_positions, batch=batch, seqlen=kv_len, fill=jnp.iinfo(jnp.int32).max)

    q_valid = q_ids >= 0
    kv_valid = kv_ids >= 0

    mask = (q_ids[:, :, None] == kv_ids[:, None, :]) & q_valid[:, :, None] & kv_valid[:, None, :]

    if causal:
        mask = mask & (q_pos[:, :, None] >= kv_pos[:, None, :])

    if window_left is not None or window_right is not None:
        wl = window_left if window_left is not None else jnp.iinfo(jnp.int32).max
        wr = window_right if window_right is not None else jnp.iinfo(jnp.int32).max
        mask = mask & (kv_pos[:, None, :] >= (q_pos[:, :, None] - wl)) & (kv_pos[:, None, :] <= (q_pos[:, :, None] + wr))

    attn_mask = _normalize_attention_mask(attention_mask, batch=batch, q_len=q_len, kv_len=kv_len)
    if attn_mask is not None:
        mask = mask & attn_mask

    row_has_any = jnp.any(mask, axis=-1)

    if softmax_aux is None:
        q_bthd = jnp.transpose(query, (0, 2, 1, 3))
        k_bthd = jnp.transpose(key, (0, 2, 1, 3))
        v_bthd = jnp.transpose(value, (0, 2, 1, 3))
        mask_4d = mask[:, None, :, :]

        out_bthd, _ = dense_attention(
            query=q_bthd,
            key=k_bthd,
            value=v_bthd,
            attention_mask=mask_4d,
            softmax_aux=None,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            dtype=q_bthd.dtype,
            softmax_dtype=None,
            dropout_prob=0.0,
            deterministic=True,
            dropout_rng=None,
            causal=causal,
            sliding_window=None,
            bias=None,
            init_bias=None,
        )

        out_bthd = out_bthd * (row_has_any & q_valid).astype(out_bthd.dtype)[:, :, None, None]
        return jnp.transpose(out_bthd, (0, 2, 1, 3))

    reps = num_heads // num_kv_heads
    if reps != 1:
        key_h = jnp.repeat(key, repeats=reps, axis=1)
        value_h = jnp.repeat(value, repeats=reps, axis=1)
    else:
        key_h = key
        value_h = value

    logits = jnp.einsum("bhtd,bhkd->bhtk", query * softmax_scale, key_h, optimize=True)
    if logits_soft_cap is not None:
        logits = logits_soft_cap * jnp.tanh(logits / logits_soft_cap)

    logits = jnp.where(mask[:, None, :, :], logits, jnp.finfo(logits.dtype).min)

    aux = _normalize_softmax_aux(softmax_aux, num_heads=num_heads, num_kv_heads=num_kv_heads, dtype=logits.dtype)
    assert aux is not None
    sinks = jnp.broadcast_to(aux[None, :, None, :], (batch, num_heads, q_len, aux.shape[-1]))
    combined = jnp.concatenate([logits, sinks], axis=-1)
    probs = jax.nn.softmax(combined.astype(jnp.float32), axis=-1).astype(logits.dtype)
    weights = probs[..., :kv_len]

    out = jnp.einsum("bhtk,bhkd->bhtd", weights, value_h, optimize=True)
    out = out * (row_has_any & q_valid).astype(out.dtype)[:, None, :, None]
    return out
