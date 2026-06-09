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

"""Ring Attention implementation using Splash Attention kernels on TPU.

This module provides a distributed ring attention implementation for TPU using
JAX's Splash Attention with ring communication topology. Ring attention enables
efficient attention computation across multiple TPU devices by distributing
the key-value sequences in a ring pattern and rotating them between devices.

Ring attention is particularly useful for:
1. Extremely long sequences that don't fit in single device memory
2. Distributed training and inference across multiple TPU chips
3. Memory-efficient attention with linear scaling across devices

Key features:
- Ring topology: KV pairs are rotated between devices in a ring pattern
- Splash Attention backend: Uses Google's highly optimized TPU attention
- Multi-head and multi-query attention support
- Flexible masking: Causal, sliding window, and chunked causal masks
- Segment-based masking for document/sequence boundaries
- Attention sinks support for streaming inference

Algorithm overview:
1. Each device holds a shard of the sequence
2. Queries stay local while KV pairs rotate around the ring
3. Each device computes attention with its current KV shard
4. Partial results are accumulated using online softmax
5. After num_devices rotations, full attention is computed

Communication pattern:
- Uses JAX's collective operations (lax.ppermute) for ring rotation
- Overlaps computation with communication for efficiency
- Memory usage scales linearly with number of devices

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._pallas.tpu.ring_attention import ring_attention
    >>>
    >>> # Distributed attention across TPU pod
    >>> batch, seq_len, num_heads, head_dim = 2, 8192, 16, 128
    >>> q = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>>
    >>> # Use within pmap/shard_map with axis_name
    >>> output = ring_attention(q, k, v, causal=True, axis_name="batch")

Reference:
    Ring Attention with Blockwise Transformers for Near-Infinite Context
    https://arxiv.org/abs/2310.01889
"""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax import Array
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask_info as mask_info_lib,
)
from jax.scipy.special import logsumexp
from jaxtyping import Float, Int

from ejkernel.ops import BwdParams, FwdParams

from ...._xla.attention import attention as _xla_attention
from ...._xla.ring_attention import ring_attention as _xla_ring_attention
from ._pallas_impl_bwd import (
    DEFAULT_MASK_VALUE,
    RING_AXIS,
    BlockSizes,
    SegmentIds,
    ring_splash_attention,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from ejkernel.kernels._pallas.tpu.blocksparse_attention._masks import Mask


class _AttentionSinkMask(mask_lib._ComputableMask):
    """Attention mask that allows attending to initial KV positions (attention sinks).

    This mask enables "attention sink" behavior where all query positions can
    attend to the first `attention_sink_size` key-value positions, regardless
    of other masking patterns. This is useful for maintaining context anchors
    in streaming/infinite context scenarios.

    Attributes:
        attention_sink_size: Number of initial KV positions to always attend to.
    """

    attention_sink_size: int

    def __init__(self, *, shape: tuple[int, int], attention_sink_size: int, shard_count: int = 1):
        """Initialize an attention sink mask.

        Args:
            shape: Tuple of (q_seq_len, kv_seq_len) for the attention matrix.
            attention_sink_size: Number of initial KV positions that all queries
                can attend to (the "sink" positions).
            shard_count: Number of shards for distributed computation.
        """
        self.attention_sink_size = int(attention_sink_size)

        def sink_mask_function(q_ids: np.ndarray, kv_ids: np.ndarray) -> np.ndarray:
            return np.broadcast_to(kv_ids < self.attention_sink_size, (q_ids.shape[0], kv_ids.shape[1]))

        super().__init__(shape=shape, mask_function=sink_mask_function, shard_count=shard_count)

    def __eq__(self, other: object):
        """Check equality based on shape, sink size, and query sequence."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.shape == other.shape
            and self.attention_sink_size == other.attention_sink_size
            and np.array_equal(self.q_sequence, other.q_sequence)
        )

    def __hash__(self):
        """Compute hash from shape, sink size, and query sequence bytes."""
        return hash(
            (
                type(self),
                self.shape,
                self.attention_sink_size,
                self.q_sequence.tobytes() if self.q_sequence is not None else None,
            )
        )


def _build_mask(
    q_seq_len: int,
    kv_seq_len: int,
    causal: bool = False,
    sliding_window: int | tuple[int, int] | None = None,
    attention_sink_size: int = 0,
    chunk_size: int | None = None,
) -> mask_lib.Mask:
    """Build a composite attention mask from attention parameters.

    Constructs a mask by combining causal/full base masks with optional
    sliding window and attention sink constraints. Supports chunked
    causal attention (Llama4-style) and sliding window with sink positions.

    Args:
        q_seq_len: Query sequence length.
        kv_seq_len: Key/value sequence length.
        causal: Whether to use causal (autoregressive) masking.
        sliding_window: Sliding window size as int (symmetric) or
            (left, right) tuple. Limits attention to nearby positions.
        attention_sink_size: Number of initial KV positions that are
            always visible regardless of other masking. Used for
            streaming/infinite context patterns.
        chunk_size: Chunk size for chunked causal attention where
            causality is applied within chunks.

    Returns:
        Constructed Mask object combining all specified masking patterns.
    """
    shape = (q_seq_len, kv_seq_len)

    if chunk_size is not None:
        mask = mask_lib.ChunkedCausalMask(shape=shape, chunk_size=chunk_size)
    elif causal:
        mask = mask_lib.CausalMask(shape=shape)
    else:
        mask = mask_lib.FullMask(shape)

    if sliding_window is not None:
        if isinstance(sliding_window, int):
            window = (sliding_window, sliding_window)
        else:
            window = sliding_window
        local_mask = mask_lib.LocalMask(shape=shape, window_size=window, offset=0)
        if attention_sink_size > 0:
            sink_size = min(int(attention_sink_size), kv_seq_len)
            local_mask = local_mask | _AttentionSinkMask(shape=shape, attention_sink_size=sink_size)
        mask = mask & local_mask

    return mask


def _make_block_sizes(
    fwd_params: FwdParams | None,
    bwd_params: BwdParams | None,
) -> BlockSizes:
    """Create BlockSizes from FwdParams and BwdParams.

    Args:
        fwd_params: Forward pass parameters.
        bwd_params: Backward pass parameters.

    Returns:
        BlockSizes configuration.
    """
    if fwd_params is None:
        return BlockSizes.get_default()

    block_q = fwd_params.q_blocksize
    block_kv = fwd_params.kv_blocksize

    if bwd_params is not None:
        block_q_dkv = bwd_params.q_blocksize
        block_kv_dkv = bwd_params.kv_blocksize
        block_q_dq = bwd_params.q_blocksize
        block_kv_dq = bwd_params.kv_blocksize
    else:
        block_q_dkv = block_q
        block_kv_dkv = block_kv
        block_q_dq = block_q
        block_kv_dq = block_kv

    return BlockSizes(
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=block_kv,
        block_q_dkv=block_q_dkv,
        block_kv_dkv=block_kv_dkv,
        block_kv_dkv_compute=block_kv_dkv,
        block_q_dq=block_q_dq,
        block_kv_dq=block_kv_dq,
    )


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
    """Computes ring attention using Splash Attention kernels on TPU.

    This implementation uses JAX's splash attention with ring communication
    topology for distributed attention computation across devices.

    When ``axis_name`` is ``None`` and no ring-specific features are requested,
    the call is delegated to the XLA single-device attention path. When ring
    features (segment IDs, position IDs, custom masks, chunk_size) are
    requested without a named axis, the XLA ring-attention fallback is used.
    When ``axis_name`` is provided, the distributed Pallas Splash Attention
    kernel is used and each batch element is processed independently via
    ``jax.vmap``.

    Args:
        query: Query tensor [batch, q_len, num_heads, head_dim].
        key: Key tensor [batch, kv_len, num_kv_heads, head_dim].
        value: Value tensor [batch, kv_len, num_kv_heads, head_dim].
        q_segment_ids: Optional query segment IDs [batch, q_len] for
            cross-document masking. When set, the first batch element's
            segment IDs are passed to the Splash kernel.
        kv_segment_ids: Optional KV segment IDs [batch, kv_len].
        q_position_ids: Optional query position IDs [batch, q_len]. Forwarded
            to the XLA ring-attention fallback; not consumed by the Pallas
            splash path when ``axis_name`` is set.
        kv_position_ids: Optional KV position IDs [batch, kv_len]. Forwarded
            to the XLA ring-attention fallback; not consumed by the Pallas
            splash path when ``axis_name`` is set.
        softmax_aux: Optional attention sink logits [num_sinks] (1-D float32).
            Reduced to a per-head scalar via ``logsumexp`` and passed to
            ``ring_splash_attention`` as ``sinks``.
        bias: Optional attention bias [batch, num_heads, q_len, kv_len].
            Not supported by the Pallas splash path; raises
            ``NotImplementedError`` when ``axis_name`` is set.
        mask_builder: Optional callable ``(q_len, kv_len, num_heads, ...)``
            that returns a :class:`Mask`. Forces the XLA ring-attention
            fallback when ``axis_name`` is ``None``.
        sliding_window: Sliding window size as int (symmetric) or
            ``(left, right)`` tuple for local attention.
        chunk_size: Chunk size for chunked causal attention (Llama4-style).
        causal: Whether to apply causal masking.
        logits_soft_cap: Optional soft cap for attention logits.
        softmax_scale: Scaling factor for QK^T. Defaults to
            ``head_dim ** -0.5`` when ``None``.
        axis_name: Named JAX collective axis for ring communication. ``None``
            routes to the XLA single-device or XLA ring fallback.
        fwd_params: Forward-pass block size parameters with fields
            ``q_blocksize`` and ``kv_blocksize``. Defaults to
            ``min(512, seq_len)`` for both when ``None``.
        bwd_params: Backward-pass block size parameters with fields
            ``q_blocksize`` and ``kv_blocksize``. Mirrors ``fwd_params``
            when ``None``.
        fused_backward: Whether to use fused backward kernel. Forwarded to
            the XLA fallback only.

    Returns:
        Attention output [batch, q_len, num_heads, head_dim] cast to the
        input dtype.

    Raises:
        NotImplementedError: If ``bias`` is provided when ``axis_name`` is set.
    """
    _, q_len, num_heads, head_dim = query.shape
    _, kv_len, num_kv_heads, _ = key.shape

    aux = None
    if softmax_aux is not None:
        aux = jnp.asarray(softmax_aux, dtype=jnp.float32)
        if aux.ndim != 1:
            raise ValueError(f"softmax_aux must be 1D, got shape {aux.shape}.")

    if axis_name is None:
        needs_ring_semantics = (
            q_segment_ids is not None
            or kv_segment_ids is not None
            or q_position_ids is not None
            or kv_position_ids is not None
            or chunk_size is not None
            or mask_builder is not None
        )
        if not needs_ring_semantics:
            out, _ = _xla_attention(
                query=query,
                key=key,
                value=value,
                bias=bias,
                softmax_aux=softmax_aux,
                softmax_scale=softmax_scale,
                logits_soft_cap=logits_soft_cap,
                causal=causal,
                sliding_window=sliding_window,
            )
            return out.astype(jnp.float32)

        return _xla_ring_attention(
            query=query,
            key=key,
            value=value,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            q_position_ids=q_position_ids,
            kv_position_ids=kv_position_ids,
            softmax_aux=softmax_aux,
            bias=bias,
            mask_builder=mask_builder,
            sliding_window=sliding_window,
            chunk_size=chunk_size,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            softmax_scale=softmax_scale,
            axis_name=None,
            fwd_params=fwd_params,
            bwd_params=bwd_params,
            fused_backward=fused_backward,
        )

    if bias is not None:
        raise NotImplementedError(
            "Attention bias is not supported in splash ring attention. "
            "Please remove the bias parameter or use a different kernel."
        )
    del mask_builder, fused_backward

    input_dtype = query.dtype
    if query.dtype != jnp.float32:
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)
        value = value.astype(jnp.float32)

    is_mqa = num_kv_heads == 1 and num_heads > 1

    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    query = query * softmax_scale

    if fwd_params is None:
        fwd_params = FwdParams(q_blocksize=min(512, q_len), kv_blocksize=min(512, kv_len))

    block_sizes = _make_block_sizes(fwd_params, bwd_params)

    mask = _build_mask(
        q_seq_len=q_len,
        kv_seq_len=kv_len,
        causal=causal,
        sliding_window=sliding_window,
        attention_sink_size=0,
        chunk_size=chunk_size,
    )

    ring_axis = axis_name if axis_name is not None else RING_AXIS

    sinks = None
    if aux is not None:
        sinks = jnp.broadcast_to(logsumexp(aux), (num_heads,))

    segment_ids = None
    if q_segment_ids is not None or kv_segment_ids is not None:
        q_seg = q_segment_ids if q_segment_ids is not None else kv_segment_ids
        kv_seg = kv_segment_ids if kv_segment_ids is not None else q_segment_ids
        segment_ids = SegmentIds(q=q_seg[0], kv=kv_seg[0])

    def single_batch_attention(q, k, v, seg_ids, sinks_batch):
        """Process single batch element."""
        q = rearrange(q, "s h d -> h s d")
        k = rearrange(k, "s h d -> h s d")
        v = rearrange(v, "s h d -> h s d")

        if is_mqa:
            k = k.squeeze(0)
            v = v.squeeze(0)

        multi_head_mask = mask_lib.MultiHeadMask(masks=tuple([mask] * num_heads))

        fwd_mask_info, mask_function = mask_info_lib._process_mask(
            multi_head_mask,
            (block_sizes.block_q, block_sizes.block_kv),
            is_dkv=False,
        )
        fwd_mask_info = jax.tree_util.tree_map(jnp.array, fwd_mask_info)

        dkv_mask_info = None
        if block_sizes.has_backward_blocks:
            dkv_mask_info, _ = mask_info_lib._process_mask(
                multi_head_mask,
                (block_sizes.block_q_dkv, block_sizes.block_kv_dkv),
                is_dkv=True,
            )
            dkv_mask_info = jax.tree_util.tree_map(jnp.array, dkv_mask_info)

        out = ring_splash_attention(
            fwd_mask_info=fwd_mask_info,
            dkv_mask_info=dkv_mask_info,
            q=q,
            k=k,
            v=v,
            segment_ids=seg_ids,
            sinks=sinks_batch,
            is_mqa=is_mqa,
            block_sizes=block_sizes,
            mask_value=DEFAULT_MASK_VALUE,
            mask_function=mask_function,
            logits_soft_cap=logits_soft_cap,
            ring_axis=ring_axis,
            causal=causal,
            sliding_window=sliding_window,
        )

        out = rearrange(out, "h s d -> s h d")
        return out

    outputs = jax.vmap(lambda q, k, v: single_batch_attention(q, k, v, segment_ids, sinks))(query, key, value)

    return outputs.astype(input_dtype)
