# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Transformer key-value caching implementation for EasyDeL.

This module provides the standard key-value caching system for transformer
models, supporting various attention patterns including full attention,
sliding window attention, and local attention.

The transformer cache is designed for efficient autoregressive generation
by storing previously computed key and value states, avoiding redundant
computation during inference.

Key Components:
    - TransformerCacheConfig: Configuration for cache dimensions and behavior
    - TransformerCacheView: Per-layer cache storage and update logic
    - TransformerCache: Multi-layer cache orchestration
    - TransformerMetadata: Runtime metadata for cache operations
    - AttnMaskDetail: Attention masking configuration

Features:
    - Support for multiple attention patterns (full, sliding, local)
    - Quantization support for memory efficiency
    - Distributed caching with JAX sharding
    - Functional cache updates for JAX compatibility
    - Dynamic mask generation and caching

Example:
    >>> # Initialize cache metadata
    >>> metadata = TransformerCacheConfig.create(
    ...     batch_size=2,
    ...     sequence_length=1024,
    ...     num_hidden_layers=12,
    ...     pad_token_id=0,
    ...     num_heads=16,
    ...     head_dim=64
    ... )
    >>>
    >>> # Create cache
    >>> cache = TransformerCache.init_cache(
    ...     mesh=mesh,
    ...     metadata=metadata,
    ...     partition_manager=pm,
    ...     dtype=jnp.bfloat16
    ... )
    >>>
    >>> # Update cache during inference
    >>> for layer_idx in range(12):
    ...     key_cache, value_cache, mask, new_view = cache[layer_idx].concatenate_to_cache(
    ...         query=query_states,
    ...         key=key_states,
    ...         value=value_states,
    ...         attention_mask=attention_mask,
    ...         quantizer=quantizer,
    ...         partition_manager=pm
    ...     )
    ...     cache[layer_idx] = new_view
"""

from __future__ import annotations

import typing as tp
from enum import Enum

import jax
from eformer.jaximus import ImplicitArray, register
from eformer.pytree import auto_pytree, field
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax import lax
from jax import numpy as jnp
from jax.extend.core import Primitive
from jax.sharding import NamedSharding as Ns
from jaxtyping import Array as JAXArray
from jaxtyping import Float, Int
from spectrax import PartitionManager, apply_logical_sharding, common_types

from easydel.infra.sharding import (
    MeshLike,
    RuntimeShardingResolver,
    coerce_runtime_sharding_resolver,
    resolve_stage_cache_mesh,
    sanitize_sharding_axes_for_shape,
)

from .._abstracts import (
    BaseCache,
    BaseCacheConfig,
    BaseCacheView,
    BaseRunTimeMetadata,
    OperationsMetadata,
    unwrap_metadata,
)

if tp.TYPE_CHECKING:
    from easydel.caching.hybrid import HybridMetadata
    from easydel.layers.quantization._quants import EasyQuantizer
else:
    EasyQuantizer = object
    HybridMetadata = object


def _maybe_materialize(x: JAXArray | ImplicitArray | None) -> JAXArray:
    """Materialize an ImplicitArray if needed, or return a regular JAXArray.

    Args:
        x: A JAXArray, ImplicitArray, or None.

    Returns:
        The materialized JAXArray.

    Raises:
        ValueError: If x is None.
        RuntimeError: If ImplicitArray.materialize() returns None.
    """
    if x is None:
        raise ValueError("Cannot materialize None")
    if isinstance(x, ImplicitArray):
        result = x.materialize()
        if result is None:
            raise RuntimeError("ImplicitArray.materialize() returned None")
        return result
    return x


@auto_pytree
class AttnMaskDetail:
    """Configuration for attention masking patterns.

    Defines the type and parameters of attention masking to apply
    during cache operations. Supports various masking strategies
    including sliding windows, chunks, and custom patterns.

    Attributes:
        mask_type (Enum): Type of attention mask (e.g., FULL, SLIDING, CHUNKED).
        size (int): Primary size parameter for the mask (window size, chunk size, etc.).
        offset (int | None): Optional offset for mask positioning.
        chunks (int | None): Number of chunks for chunked attention.
        bricks (int | None): Number of bricks for blocked attention patterns.
    """

    mask_type: str | Enum = field(pytree_node=False)
    size: int = field(pytree_node=False)
    offset: int | None = field(pytree_node=False, default=None)
    chunks: int | None = field(pytree_node=False, default=None)
    bricks: int | None = field(pytree_node=False, default=None)


NOT_GIVEN = common_types.NOT_GIVEN
RUNTIME_MODE_TYPES = common_types.RUNTIME_MODE_TYPES
BATCH = common_types.BATCH
QUERY_LENGTH = common_types.QUERY_LENGTH
KV_LENGTH = common_types.KV_LENGTH
HEAD = common_types.HEAD
KV_HEAD = common_types.KV_HEAD
HEAD_DIM = common_types.HEAD_DIM
KV_HEAD_DIM = common_types.KV_HEAD_DIM
BIAS_HEAD_SEQ = common_types.BIAS_HEAD_SEQ
BIAS_KV_SEQ = common_types.BIAS_KV_SEQ
MODE_PREFILL = common_types.MODE_PREFILL


def _expand_mask_kv_dim(
    mask_info: MaskInfo,
    target_kv_len: int,
    cache_position: jnp.ndarray,
    query_len: int,
) -> MaskInfo:
    """Expand mask's KV dimension to match cache size.

    When using HybridCache with TransformerCacheView, the mask may have
    a smaller KV dimension than the pre-allocated cache. This function
    expands the mask by padding the KV dimension while preserving any existing
    padding/segment masking.

    Args:
        mask_info: Original MaskInfo with potentially smaller KV dimension.
        target_kv_len: Target KV dimension (cache's sequence length).
        cache_position: Current position in the cache [batch].
        query_len: Length of the query sequence.

    Returns:
        MaskInfo with expanded KV dimension.
    """
    current_kv_len = mask_info.kv_len

    if current_kv_len is not None and current_kv_len >= target_kv_len:
        return mask_info

    # Preserve any existing padding/segment mask and expand the KV dimension.
    #
    # Important: the extra KV positions correspond to *future tokens* that will be written
    # into the preallocated cache during generation. They must NOT be treated as padding
    # (otherwise newly generated tokens will remain permanently masked). Instead, we mark
    # them as "valid" and rely on `MaskInfo.apply_kv_lengths` and causal masking to keep
    # them inactive until they are actually populated.
    attn = mask_info.get_or_compute_attention_mask(dtype=jnp.bool_)  # (B, H, Q, K)
    pad_k = target_kv_len - current_kv_len
    attn = jnp.pad(attn, ((0, 0), (0, 0), (0, 0), (0, pad_k)), constant_values=True)

    kv_seg = getattr(mask_info, "_kv_segment_ids", None)
    if kv_seg is not None and kv_seg.shape[-1] == current_kv_len:
        # Segment id -1 is reserved for padding; use 0 so future positions are considered valid.
        kv_seg = jnp.pad(jnp.asarray(kv_seg, dtype=jnp.int32), ((0, 0), (0, pad_k)), constant_values=0)

    del cache_position, query_len
    return mask_info.replace(attention_mask=attn, kv_segment_ids=kv_seg)


def _recover_pairwise_segment_ids(
    mask_info: MaskInfo,
    write_index: jnp.ndarray,
    cached_segment_ids: jnp.ndarray | None,
    *,
    sliding: bool,
) -> MaskInfo:
    """Recover packed ownership that a pairwise mask's padding labels omit.

    ``MaskInfo.from_attention_mask`` deliberately labels every valid token zero.
    For a square, query-local self-attention mask, a packed boundary is possible
    only when no query sees keys on both sides of it. Missing adjacent edges are
    not sufficient: a longer-range edge can span the apparent boundary. Taking
    each query's entire visible key interval conservatively merges such spans,
    preserving every explicitly allowed pairwise edge. Query ownership comes
    from its first visible key (not its diagonal: padding queries can still be
    valid queries). Existing nonzero explicit segment labels win.

    The first local segment continues the last cached segment; later segments
    receive fresh labels. Callers use a zero origin for absolute masks, whose
    pairwise historical restrictions remain unchanged. All-zero valid labels
    denote the first local continuation segment regardless of whether the mask
    was supplied as segments or has already been materialized. Materialize a copy so that interpreting a mask
    for auxiliary selection cannot change its later interpretation by KV append.
    """
    mask_info = mask_info.replace()
    attn = mask_info.get_or_compute_attention_mask(dtype=jnp.bool_)
    if attn.shape[-2] != attn.shape[-1]:
        return mask_info
    q_segments, kv_segments = mask_info.get_or_compute_segment_ids()
    padding_labels = jnp.all(q_segments <= 0, axis=-1) & jnp.all(kv_segments <= 0, axis=-1)
    pairwise = jnp.any(attn, axis=1)
    valid_keys = jnp.any(pairwise, axis=-2)
    first_visible_key = jnp.argmax(pairwise, axis=-1)
    key_positions = jnp.arange(pairwise.shape[-1], dtype=jnp.int32)
    last_visible_key = jnp.max(jnp.where(pairwise, key_positions, -1), axis=-1)
    # A query's visible keys must all receive the same ownership label. Infer
    # boundaries from complete visible intervals, including across heads, rather
    # than just the super/subdiagonal of an arbitrary pairwise mask.
    splits = key_positions[:-1]
    crosses = (first_visible_key[:, :, None] <= splits) & (last_visible_key[:, :, None] > splits)
    boundaries = jnp.pad(~jnp.any(crosses, axis=-2), ((0, 0), (1, 0)), constant_values=False)
    recovered = jnp.cumsum(boundaries, axis=-1, dtype=jnp.int32)
    continuation = jnp.zeros_like(write_index)
    label_origin = continuation
    if cached_segment_ids is not None:
        width = cached_segment_ids.shape[-1]
        last = jnp.minimum(write_index, width) - 1 if sliding else write_index - 1
        continuation = jnp.take_along_axis(cached_segment_ids, jnp.clip(last, 0, width - 1)[:, None], axis=-1)[:, 0]
        continuation = jnp.where(write_index > 0, jnp.maximum(continuation, 0), 0)
        # Explicit historical labels need not be monotone. A fresh packed
        # segment must not collide with an older label before the continuation.
        label_origin = jnp.maximum(jnp.max(cached_segment_ids, axis=-1), continuation)
    recovered = jnp.where(recovered == 0, continuation[:, None], recovered + label_origin[:, None])
    recovered = jnp.where(valid_keys, recovered, -1)
    recovered_queries = jnp.take_along_axis(recovered, first_visible_key, axis=-1)
    recovered_queries = jnp.where(jnp.any(pairwise, axis=-1), recovered_queries, -1)
    return mask_info.replace(
        _q_segment_ids=jnp.where(padding_labels[:, None], recovered_queries, q_segments),
        _kv_segment_ids=jnp.where(padding_labels[:, None], recovered, kv_segments),
    )


def _align_postpadded_mask(
    mask_info: MaskInfo,
    target_kv_len: int,
    write_index: jnp.ndarray,
    new_lengths: jnp.ndarray,
    query_len: int,
    *,
    sliding: bool,
    cached_segment_ids: jnp.ndarray | None = None,
    first_position: jnp.ndarray | None = None,
    causal: bool = True,
) -> MaskInfo:
    """Map an append's mask to persistent or attention-time KV coordinates.

    Args:
        mask_info: Incoming query-local or absolute attention mask.
        target_kv_len: Width of the returned KV storage.
        write_index: Absolute first incoming position per batch item.
        new_lengths: Valid incoming token counts, excluding right padding.
        query_len: Number of incoming query rows, including right padding.
        sliding: Whether storage contains a rolled trailing window.
        cached_segment_ids: Historical segment labels in the cache's previous
            storage coordinates. None supports legacy views, whose history is
            assumed to continue the first incoming segment.
        first_position: Optional absolute first attention-time KV column. This
            permits returning the old tail plus a whole query block while the
            persistent cache retains only its final window.
        causal: Add causal visibility for query-local masks. False preserves
            bidirectional incoming edges (and any explicit caller restrictions).

    Returns:
        MaskInfo with query_len rows and target_kv_len storage-aligned columns.
        Query-local columns are offset by write_index; historical columns remain
        available to valid queries. Absolute masks retain their pairwise masking.
    """
    attn = mask_info.get_or_compute_attention_mask(dtype=jnp.bool_)
    local = attn.shape[-1] == query_len
    query_start = jnp.clip(write_index, 0, max(attn.shape[-2] - query_len, 0))
    attn = jax.vmap(lambda x, start: lax.dynamic_slice_in_dim(x, start, query_len, axis=1))(attn, query_start)
    end = write_index + new_lengths
    first = jnp.maximum(end - target_kv_len, 0) if sliding else jnp.zeros_like(end)
    if first_position is not None:
        first = first_position
    positions = first[:, None] + jnp.arange(target_kv_len, dtype=write_index.dtype)[None, :]
    source_columns = positions - write_index[:, None] if local else positions
    columns_in_mask = (source_columns >= 0) & (source_columns < attn.shape[-1])
    columns = jnp.clip(source_columns, 0, attn.shape[-1] - 1)
    aligned = jnp.take_along_axis(attn, columns[:, None, None, :], axis=-1)
    aligned = aligned & columns_in_mask[:, None, None, :]
    if local:
        # There is no historical padding information in a query-local mask.
        # The view's starts/end bounds identify valid history instead. Preserve
        # all-masked query rows (e.g. padding from a segment-based input mask).
        historical = positions < write_index[:, None]
        query_valid = jnp.any(attn, axis=-1, keepdims=True)
        aligned = jnp.where(historical[:, None, None, :], query_valid, aligned)
        if causal:
            query_positions = write_index[:, None] + jnp.arange(query_len, dtype=write_index.dtype)[None, :]
            aligned = aligned & (positions[:, None, None, :] <= query_positions[:, None, :, None])
    aligned = aligned & (positions < end[:, None])[:, None, None, :]

    def slice_queries(x):
        """Keep optional query metadata aligned with the selected mask rows."""
        if x is None:
            return None
        return jax.vmap(lambda row, start: lax.dynamic_slice_in_dim(row, start, query_len, axis=0))(x, query_start)

    q_segments, kv_segments = mask_info.get_or_compute_segment_ids()
    q_segments = slice_queries(q_segments)
    translated_segments = jnp.take_along_axis(kv_segments, columns, axis=-1)
    translated_segments = jnp.where(columns_in_mask, translated_segments, -1)
    if local:
        if cached_segment_ids is None:
            # A manually constructed legacy view has no historical labels.
            # Only its continuation segment may see that history, never later
            # packed segments from this append.
            historical_segments = jnp.broadcast_to(kv_segments[:, :1], positions.shape)
        else:
            old_width = cached_segment_ids.shape[-1]
            old_first = jnp.maximum(write_index - old_width, 0) if sliding else jnp.zeros_like(write_index)
            old_columns = jnp.clip(positions - old_first[:, None], 0, old_width - 1)
            historical_segments = jnp.take_along_axis(cached_segment_ids, old_columns, axis=-1)
        translated_segments = jnp.where(positions < write_index[:, None], historical_segments, translated_segments)
        same_segment = (q_segments[:, :, None] == translated_segments[:, None, :]) & (q_segments[:, :, None] >= 0)
        aligned = aligned & same_segment[:, None, :, :]
    kv_segments = jnp.where(positions < end[:, None], translated_segments, -1)
    return mask_info.replace(
        attention_mask=aligned,
        _q_segment_ids=q_segments,
        _kv_segment_ids=kv_segments,
        q_positions=slice_queries(mask_info.q_positions),
        kv_positions=positions if mask_info.kv_positions is not None else None,
        causal_mask_baked_in=(local and causal) or mask_info.causal_mask_baked_in,
    )


def _append_cache_rows(old, new, indexes, lengths, *, sliding: bool):
    """Write per-row valid tokens in the shared KV/auxiliary physical layout."""

    @jax.vmap
    def append(row, tokens, index, length):
        width = row.shape[0]
        if not sliding:
            offsets = jnp.arange(tokens.shape[0], dtype=index.dtype)
            positions = index + offsets
            valid = (offsets < length) & (positions >= 0) & (positions < width)
            return row.at[jnp.where(valid, positions, width)].set(tokens.astype(row.dtype), mode="drop")
        end = index + length
        positions = jnp.maximum(end - width, 0) + jnp.arange(width)
        old_rows = jnp.clip(positions - jnp.maximum(index - width, 0), 0, width - 1)
        new_rows = jnp.clip(positions - index, 0, tokens.shape[0] - 1)
        broadcast = (width,) + (1,) * (row.ndim - 1)
        result = jnp.where((positions < index).reshape(broadcast), row[old_rows], tokens[new_rows])
        return jnp.where((positions < end).reshape(broadcast), result, 0).astype(row.dtype)

    return append(old, new, indexes, lengths)


def _attention_cache_rows(old, new, indexes, lengths):
    """Return the old trailing window plus all current rows for block attention."""
    width = old.shape[1]
    attention_width = width + new.shape[1]

    @jax.vmap
    def rows(row, tokens, index, length):
        positions = jnp.maximum(index - width, 0) + jnp.arange(attention_width)
        old_rows = jnp.clip(positions - jnp.maximum(index - width, 0), 0, width - 1)
        new_rows = jnp.clip(positions - index, 0, tokens.shape[0] - 1)
        broadcast = (attention_width,) + (1,) * (row.ndim - 1)
        result = jnp.where((positions < index).reshape(broadcast), row[old_rows], tokens[new_rows])
        return jnp.where((positions < index + length).reshape(broadcast), result, 0)

    return rows(old, new, indexes, lengths)


def _append_cache_mask(mask_info, width, indexes, lengths, starts, segment_ids, query_len, *, sliding_window, causal):
    """Align ownership, padding and windows identically for KV and auxiliary states."""
    local = mask_info.kv_len == query_len
    incoming = _recover_pairwise_segment_ids(
        mask_info,
        indexes if local else jnp.zeros_like(indexes),
        segment_ids if local else None,
        sliding=sliding_window is not None,
    )
    aligned = _align_postpadded_mask(
        incoming,
        width,
        indexes,
        lengths,
        query_len,
        sliding=sliding_window is not None,
        cached_segment_ids=segment_ids,
        causal=causal,
    )
    stored_segments = aligned.get_or_compute_segment_ids()[1]
    first = jnp.zeros_like(indexes)
    if sliding_window is not None:
        first = jnp.maximum(indexes + lengths - width, 0)
        if query_len > 1:
            first = jnp.maximum(indexes - width, 0)
            aligned = _align_postpadded_mask(
                incoming,
                width + query_len,
                indexes,
                lengths,
                query_len,
                sliding=True,
                cached_segment_ids=segment_ids,
                first_position=first,
                causal=causal,
            )
    attn = aligned.get_or_compute_attention_mask(dtype=jnp.bool_)
    kv_positions = first[:, None] + jnp.arange(attn.shape[-1], dtype=jnp.int32)[None, :]
    query_positions = indexes[:, None] + jnp.arange(query_len)[None, :]
    attn = attn & (kv_positions >= starts[:, None])[:, None, None, :]
    if sliding_window is not None:
        relative = kv_positions[:, None, :] - query_positions[:, :, None]
        window = (relative > -sliding_window) & (relative <= 0 if causal else relative < sliding_window)
        attn = attn & window[:, None, :, :]
        aligned = aligned.replace(
            sliding_window_baked_in=True, causal_mask_baked_in=causal or aligned.causal_mask_baked_in
        )
    return aligned.replace(attention_mask=attn, q_positions=query_positions, kv_positions=kv_positions), stored_segments


def _append_metadata(indexes, starts, mask_info, query_len, cache_metadata):
    """Resolve the identical insertion offsets and valid lengths for every cache state."""
    metadata = unwrap_metadata(cache_metadata, "transformer")
    write_index = indexes if metadata is None or metadata.indexes is None else metadata.indexes
    starts = starts if metadata is None or metadata.starts is None else metadata.starts
    write_index = jnp.asarray(write_index, dtype=indexes.dtype)
    starts = jnp.asarray(starts, dtype=indexes.dtype)
    lengths = jnp.full_like(write_index, query_len)
    if metadata is not None and metadata.postpadded:
        valid_keys = jnp.any(mask_info.replace().get_or_compute_attention_mask(dtype=jnp.bool_), axis=(1, 2))
        offsets = jnp.arange(query_len, dtype=jnp.int32)[None, :]
        if valid_keys.shape[-1] != query_len:
            offsets = offsets + write_index[:, None]
        valid_new = jnp.take_along_axis(valid_keys, offsets, axis=1, mode="fill", fill_value=False)
        lengths = jnp.sum(valid_new, axis=-1, dtype=indexes.dtype)
    return write_index, starts, lengths


@register("dynamic_update_slice")
def _(
    primitive: Primitive,
    operand: ImplicitArray,
    update: tp.Any,
    *args,
    **kwargs,
) -> JAXArray:
    """Register handler for dynamic_update_slice with ImplicitArray operand.

    Materializes the implicit array before performing the update operation.
    This ensures compatibility with quantized or lazy-evaluated arrays.

    Args:
        primitive: JAX primitive for dynamic_update_slice.
        operand: ImplicitArray to update (will be materialized).
        update: Update values.
        *args: Additional arguments for the primitive.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of the dynamic_update_slice operation.
    """
    operand = operand.materialize()
    return primitive.bind(operand, update, *args)


@register("dynamic_update_slice")
def _(
    primitive: Primitive,
    operand: tp.Any,
    update: ImplicitArray,
    *args,
    **kwargs,
) -> JAXArray:
    """Handler for ``dynamic_update_slice`` when only ``update`` is implicit.

    Materializes the implicit update tensor before delegating to the primitive.

    Args:
        primitive: The ``dynamic_update_slice`` JAX primitive.
        operand: The destination array (concrete).
        update: The replacement values (``ImplicitArray``, will be materialized).
        *args: Additional positional arguments forwarded to the primitive.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        The result of binding the primitive to the materialized arrays.
    """
    update = update.materialize()
    return primitive.bind(operand, update, *args)


@register("dynamic_update_slice")
def _(
    primitive: Primitive,
    operand: ImplicitArray,
    update: ImplicitArray,
    *args,
    **kwargs,
) -> JAXArray:
    """Handler for ``dynamic_update_slice`` when both inputs are implicit.

    Materializes both arrays before delegating to the primitive.

    Args:
        primitive: The ``dynamic_update_slice`` JAX primitive.
        operand: The destination ``ImplicitArray`` (will be materialized).
        update: The replacement ``ImplicitArray`` (will be materialized).
        *args: Additional positional arguments forwarded to the primitive.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        The result of binding the primitive to the materialized arrays.
    """
    operand = operand.materialize()
    update = update.materialize()
    return primitive.bind(operand, update, *args)


@auto_pytree
class TransformerCacheConfig(BaseCacheConfig):
    """Metadata configuration for transformer key-value caching.

    Stores all static configuration needed to initialize and operate
    a transformer cache. Supports various attention head configurations
    including multi-head, multi-query, and grouped-query attention.

    The metadata defines:
    - Cache dimensions (batch, sequence, layers)
    - Attention head configuration
    - Masking and bias settings
    - Special attention patterns (sliding window)

    Attributes:
        batch_size (int): Number of sequences in batch.
        sequence_length (int): Maximum sequence length to cache.
        num_hidden_layers (int): Number of transformer layers.
        pad_token_id (int): Token ID used for padding.
        num_heads (int | None): Number of attention heads (for regular MHA).
        head_dim (int | None): Dimension of each attention head.
        key_heads (int | None): Number of key heads (for MQA/GQA).
        value_heads (int | None): Number of value heads (for MQA/GQA).
        key_dim (int | None): Dimension of key projections.
        value_dim (int | None): Dimension of value projections.
        sliding_window (int | None): Size of sliding attention window.
        update_causal_mask (bool): Whether to update causal masks dynamically.
        create_attention_bias (bool): Whether to create attention bias terms.
        auxiliary_dim (int | None): Optional per-token indexer feature width.
            Allocated in float32 with the same batch/capacity as KV before tracing.
    """

    batch_size: int = field(pytree_node=False)
    sequence_length: int = field(pytree_node=False)
    num_hidden_layers: int = field(pytree_node=False)
    pad_token_id: int = field(pytree_node=False)
    num_heads: int | None = field(pytree_node=False)
    head_dim: int | None = field(pytree_node=False)
    key_heads: int | None = field(pytree_node=False)
    value_heads: int | None = field(pytree_node=False)
    key_dim: int | None = field(pytree_node=False)
    value_dim: int | None = field(pytree_node=False)
    sliding_window: int | None = field(pytree_node=False)

    update_causal_mask: bool = field(pytree_node=False)
    create_attention_bias: bool = field(pytree_node=False)
    auxiliary_dim: int | None = field(pytree_node=False, default=None)

    @classmethod
    def create(
        cls,
        batch_size: int,
        sequence_length: int,
        num_hidden_layers: int,
        pad_token_id: int = -100,
        num_heads: int | None = None,
        head_dim: int | None = None,
        key_heads: int | None = None,
        value_heads: int | None = None,
        key_dim: int | None = None,
        value_dim: int | None = None,
        update_causal_mask: bool = True,
        create_attention_bias: bool = True,
        sliding_window: int | None = None,
        auxiliary_dim: int | None = None,
    ) -> TransformerCacheConfig:
        """Create a TransformerCacheConfig instance with validation.

        Args:
            batch_size: Number of sequences in the batch. Must be positive.
            sequence_length: Maximum sequence length to cache. Must be positive.
            num_hidden_layers: Number of transformer layers in the model.
            pad_token_id: Token ID used for padding. Default: -100.
            num_heads: Number of attention heads (for standard MHA).
                Either num_heads or both key_heads and value_heads must be set.
            head_dim: Dimension of each attention head. Either head_dim
                or both key_dim and value_dim must be set.
            key_heads: Number of key heads (for MQA/GQA). Defaults to num_heads.
            value_heads: Number of value heads (for MQA/GQA). Defaults to num_heads.
            key_dim: Dimension of key projections. Defaults to head_dim.
            value_dim: Dimension of value projections. Defaults to head_dim.
            update_causal_mask: Whether to update causal masks dynamically.
            create_attention_bias: Whether to create attention bias terms.
            sliding_window: Optional sliding window size for local attention.
            auxiliary_dim: Optional persistent indexer width. For example pooled
                key/gate/valid features use 2 * index_head_dim + 1; token keys
                use index_head_dim. Omitted means no auxiliary allocation.

        Returns:
            TransformerCacheConfig: Validated configuration instance.

        Raises:
            ValueError: If batch_size or sequence_length are non-positive,
                or if head dimensions cannot be determined.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if auxiliary_dim is not None and auxiliary_dim <= 0:
            raise ValueError("auxiliary_dim must be positive when specified")

        if head_dim is not None:
            key_dim = key_dim or head_dim
            value_dim = value_dim or head_dim
        else:
            if key_dim is None or value_dim is None:
                raise ValueError("Either head_dim or both key_dim and value_dim must be specified")

        if num_heads is not None:
            key_heads = key_heads or num_heads
            value_heads = value_heads or num_heads
        else:
            if key_heads is None or value_heads is None:
                raise ValueError("Either num_heads or both key_heads and value_heads must be specified")

        return cls(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_hidden_layers=num_hidden_layers,
            pad_token_id=pad_token_id,
            num_heads=num_heads,
            head_dim=head_dim,
            key_heads=key_heads,
            value_heads=value_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            update_causal_mask=update_causal_mask,
            create_attention_bias=create_attention_bias,
            sliding_window=sliding_window,
            auxiliary_dim=auxiliary_dim,
        )


@auto_pytree(frozen=False)
class TransformerCacheView(BaseCacheView):
    """Single-layer cache view for transformer key-value states.

    Manages the cached key and value tensors for one transformer layer,
    along with position tracking and masking information. Supports
    various attention patterns and quantization strategies.

    The view maintains:
    - Key and value state tensors
    - Current position indices for each sequence
    - Starting positions for relative indexing
    - Masking configuration for attention patterns

    Attributes:
        key (cx.Array | ImplicitArray): Cached key states.
            Shape: [batch_size, seq_length, num_key_heads, key_dim]
        value (cx.Array | ImplicitArray): Cached value states.
            Shape: [batch_size, seq_length, num_value_heads, value_dim]
        indexes (cx.Array | ImplicitArray): Current position index per sequence.
            Shape: [batch_size]
        starts (cx.Array | ImplicitArray): Starting position per sequence.
            Shape: [batch_size]
        metadata (TransformerCacheConfig): Static cache configuration.
        maximum_sequence_length (int): Maximum cacheable sequence length.
        layer_index (int | None): Index of this layer in the model.
        masking_details (AttnMaskDetail | None): Attention mask configuration.
        segment_ids (Array | None): Historical segment labels in the same
            storage coordinates as key/value, populated by cache updates.
        auxiliary_state (Array | None): Optional fixed-capacity float32 indexer
            features [batch, capacity, auxiliary_dim], serialized in the view pytree.
    """

    key: Float[JAXArray, "batch seq_len num_key_heads key_dim"] | ImplicitArray | None
    value: Float[JAXArray, "batch seq_len num_value_heads value_dim"] | ImplicitArray | None
    indexes: Int[JAXArray, "batch"] | ImplicitArray  # noqa: F821
    starts: Int[JAXArray, "batch"] | ImplicitArray  # noqa: F821

    metadata: TransformerCacheConfig

    maximum_sequence_length: int = field(pytree_node=False)

    layer_index: int | None = field(pytree_node=False, default=None)
    masking_details: AttnMaskDetail | None = field(pytree_node=False, default=None)
    kv_sharding_axes: tuple[object | None, ...] = field(
        pytree_node=False, default_factory=lambda: (BATCH, KV_LENGTH, KV_HEAD, KV_HEAD_DIM)
    )
    batch_sharding_axes: tuple[object | None, ...] = field(pytree_node=False, default_factory=lambda: (BATCH,))
    segment_ids: Int[JAXArray, "batch seq_len"] | None = None
    auxiliary_state: Float[JAXArray, "batch seq_len auxiliary_dim"] | None = None

    @classmethod
    def init(
        cls,
        config: TransformerCacheConfig,
        layer_index: int | None = None,
        *,
        mesh: MeshLike | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        runtime_sharding_resolver: RuntimeShardingResolver | PartitionManager | None = None,
        partition_manager: RuntimeShardingResolver | PartitionManager | None = None,
        quantizer: EasyQuantizer | None = None,
        masking_details: AttnMaskDetail | None = None,
        starts: Int[JAXArray, "batch"] | None = None,  # noqa: F821
    ):
        """Initialize a TransformerCacheView from a cache config.

        Creates and allocates cache tensors with appropriate shapes,
        dtypes, and sharding for distributed execution.

        Args:
            config: TransformerCacheConfig with cache dimensions.
            layer_index: Index of this layer in the model.
            mesh: JAX device mesh for sharding.
            dtype: Data type for cache tensors.
            runtime_sharding_resolver: Runtime sharding resolver for cache tensors.
            partition_manager: Deprecated compatibility alias for
                ``runtime_sharding_resolver``.
            quantizer: Quantization configuration.
            masking_details: Attention mask configuration.
            starts: Initial starting positions per sequence.

        Returns:
            TransformerCacheView: Initialized cache view.
        """
        from easydel.infra.utils import AttnMaskType
        from easydel.layers.quantization._quants import EasyQuantizer as EQ

        if quantizer is None:
            quantizer = EQ(quantization_config=None)
        mesh = resolve_stage_cache_mesh(mesh)
        runtime_sharding_resolver = coerce_runtime_sharding_resolver(
            runtime_sharding_resolver if runtime_sharding_resolver is not None else partition_manager,
            mesh=mesh,
        )

        with jax.named_scope("easydel-transformer-cacheview-init"):
            mt = config
            kshape = (mt.batch_size, mt.sequence_length, mt.key_heads, mt.key_dim)
            vshape = (mt.batch_size, mt.sequence_length, mt.value_heads, mt.value_dim)
            kv_sharding_axes: list[object | None] = [BATCH, KV_LENGTH, KV_HEAD, KV_HEAD_DIM]
            if masking_details is not None:
                if masking_details.mask_type == AttnMaskType.SLIDING and masking_details.size is not None:
                    kshape = (mt.batch_size, min(masking_details.size, mt.sequence_length), mt.key_heads, mt.key_dim)
                    vshape = (mt.batch_size, min(masking_details.size, mt.sequence_length), mt.value_heads, mt.value_dim)

            kv_safe_k = sanitize_sharding_axes_for_shape(
                mesh=mesh,
                runtime_sharding_resolver=runtime_sharding_resolver,
                axes=kv_sharding_axes,
                mode=MODE_PREFILL,
                shape=kshape,
            )
            kv_safe_v = sanitize_sharding_axes_for_shape(
                mesh=mesh,
                runtime_sharding_resolver=runtime_sharding_resolver,
                axes=kv_sharding_axes,
                mode=MODE_PREFILL,
                shape=vshape,
            )
            kv_sharding_axes = [
                axis if (kv_safe_k[i] is not None and kv_safe_v[i] is not None) else None
                for i, axis in enumerate(kv_sharding_axes)
            ]

            batch_axes: list[object | None] = [BATCH]
            batch_safe = sanitize_sharding_axes_for_shape(
                mesh=mesh,
                runtime_sharding_resolver=runtime_sharding_resolver,
                axes=batch_axes,
                mode=MODE_PREFILL,
                shape=(mt.batch_size,),
            )
            if batch_safe[0] is None:
                batch_axes = [None]

            kshardings = Ns(
                mesh,
                runtime_sharding_resolver.resolve(axes=kv_sharding_axes, mode=MODE_PREFILL, shape=kshape),
            )
            vshardings = Ns(
                mesh,
                runtime_sharding_resolver.resolve(axes=kv_sharding_axes, mode=MODE_PREFILL, shape=vshape),
            )
            ishardings = Ns(
                mesh,
                runtime_sharding_resolver.resolve(axes=batch_axes, mode=MODE_PREFILL, shape=(mt.batch_size,)),
            )

            if starts is None:
                starts = jnp.zeros((mt.batch_size,), dtype=jnp.int32)

            starts = apply_logical_sharding(
                starts,
                axes=batch_axes,
                mode=MODE_PREFILL,
                partition_manager=runtime_sharding_resolver,
            )

            out = cls(
                key=quantizer(jnp.zeros(shape=kshape, dtype=dtype, device=kshardings)),
                value=quantizer(jnp.zeros(shape=vshape, dtype=dtype, device=vshardings)),
                indexes=jnp.zeros((config.batch_size,), dtype=jnp.int32, device=ishardings),
                starts=starts,
                metadata=config,
                layer_index=layer_index,
                masking_details=masking_details,
                maximum_sequence_length=mt.sequence_length,
                kv_sharding_axes=tuple(kv_sharding_axes),
                batch_sharding_axes=tuple(batch_axes),
                segment_ids=apply_logical_sharding(
                    jnp.full(kshape[:2], -1, dtype=jnp.int32),
                    axes=kv_sharding_axes[:2],
                    mode=MODE_PREFILL,
                    partition_manager=runtime_sharding_resolver,
                ),
                auxiliary_state=None
                if config.auxiliary_dim is None
                else apply_logical_sharding(
                    jnp.zeros((*kshape[:2], config.auxiliary_dim), dtype=jnp.float32),
                    axes=(*kv_sharding_axes[:2], None),
                    mode=MODE_PREFILL,
                    partition_manager=runtime_sharding_resolver,
                ),
            )
        return out

    def reset(self) -> "TransformerCacheView":
        """Return this cache view with key/value storage and position indices zeroed."""
        return self.replace(
            key=None if self.key is None else jnp.zeros_like(self.key),
            value=None if self.value is None else jnp.zeros_like(self.value),
            indexes=jnp.zeros_like(self.indexes),
            starts=jnp.zeros_like(self.starts),
            segment_ids=None if self.segment_ids is None else jnp.full_like(self.segment_ids, -1),
            auxiliary_state=None if self.auxiliary_state is None else jnp.zeros_like(self.auxiliary_state),
        )

    def read_auxiliary(self) -> JAXArray | None:
        """Read fixed-capacity auxiliary storage in the same physical layout as KV.

        Unused slots are zero, not valid candidates. Use the mask returned by
        :meth:`update_auxiliary` for selection rather than scoring unused slots.
        None means auxiliary storage was not requested at allocation time.
        """
        return self.auxiliary_state

    def update_auxiliary(
        self,
        new_states: JAXArray,
        mask_info: MaskInfo,
        cache_metadata: TransformerMetadata | OperationsMetadata | HybridMetadata | None = None,
        *,
        causal: bool = True,
        mode: common_types.RUNTIME_MODE_TYPES = MODE_PREFILL,
        runtime_sharding_resolver: RuntimeShardingResolver | PartitionManager | None = None,
        partition_manager: RuntimeShardingResolver | PartitionManager | None = None,
    ) -> tuple[JAXArray, MaskInfo, TransformerCacheView]:
        """Append indexer features before the matching KV append, without advancing KV.

        Args:
            new_states: Current per-token features [batch, query_len, auxiliary_dim].
            mask_info: The original incoming KV mask, including packed ownership.
            cache_metadata: The same optional offsets, starts and postpadding
                metadata passed to the matching :meth:`concatenate_to_cache`.
            causal: Whether selection is causal (default) or bidirectional.
            mode: Runtime sharding mode.
            runtime_sharding_resolver: Optional runtime sharding policy.
            partition_manager: Compatibility alias for the runtime policy.

        Returns:
            Attention-time features, their aligned visibility mask, and a view
            with updated fixed-capacity auxiliary storage. Indexes, starts and
            segment history are deliberately not advanced: call concatenate_to_cache
            on this returned view with the SAME original mask/metadata next. For a
            sliding block the first output includes the old tail plus current rows,
            exactly like attention-time KV; the stored state keeps the final window.

        Raises:
            ValueError: If storage was not preallocated or feature shapes mismatch.
        """
        from easydel.infra.utils import AttnMaskType

        old = self.auxiliary_state
        if old is None:
            raise ValueError("Request auxiliary_dim in TransformerCacheConfig before decoding")
        if new_states.ndim != 3 or new_states.shape[0] != old.shape[0] or new_states.shape[2] != old.shape[2]:
            raise ValueError("new_states must have shape [cache batch, query_len, auxiliary_dim]")
        if self.key is None or old.shape[:2] != self.key.shape[:2]:
            raise ValueError("auxiliary storage must have the same batch and capacity as KV")
        indexes, starts, lengths = _append_metadata(
            self.indexes, self.starts, mask_info, new_states.shape[1], cache_metadata
        )
        sliding_window = (
            self.masking_details.size
            if self.masking_details is not None and self.masking_details.mask_type == AttnMaskType.SLIDING
            else None
        )
        stored = _append_cache_rows(old, new_states, indexes, lengths, sliding=sliding_window is not None)
        attention = stored
        if sliding_window is not None and new_states.shape[1] > 1:
            attention = _attention_cache_rows(old, new_states, indexes, lengths)
        aligned_mask, _ = _append_cache_mask(
            mask_info,
            old.shape[1],
            indexes,
            lengths,
            starts,
            self.segment_ids,
            new_states.shape[1],
            sliding_window=sliding_window,
            causal=causal,
        )
        resolver = coerce_runtime_sharding_resolver(
            runtime_sharding_resolver if runtime_sharding_resolver is not None else partition_manager
        )
        axes = (*self.kv_sharding_axes[:2], None)
        stored = apply_logical_sharding(stored, axes=axes, mode=mode, partition_manager=resolver)
        attention = apply_logical_sharding(attention, axes=axes, mode=mode, partition_manager=resolver)
        return attention, aligned_mask, self.replace(auxiliary_state=stored)

    @jax.named_scope("easydel-transformer-cacheview-concatenate-to-cache")
    def concatenate_to_cache(
        self,
        query: Float[JAXArray, "batch query_len num_heads head_dim"],
        key: Float[JAXArray, "batch query_len num_key_heads key_dim"],
        value: Float[JAXArray, "batch query_len num_value_heads value_dim"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        quantizer: EasyQuantizer,
        cache_metadata: "TransformerMetadata | OperationsMetadata | HybridMetadata | None",
        mask_info: MaskInfo,
        runtime_sharding_resolver: RuntimeShardingResolver | PartitionManager | None = None,
        partition_manager: RuntimeShardingResolver | PartitionManager | None = None,
        causal: bool = True,
    ) -> tuple[
        Float[JAXArray, "batch seq_len num_key_heads key_dim"],
        Float[JAXArray, "batch seq_len num_value_heads value_dim"],
        MaskInfo,
        TransformerCacheView,
        AttnMaskDetail | None,
    ]:
        """Update the KV cache functionally and return updated tensors with attention mask.

        Inserts new key/value states into the cache at the current position,
        advances position indices, and applies appropriate masking. Supports
        both standard and sliding-window attention patterns.

        Args:
            query: Current query states.
                Shape: [batch, query_len, num_heads, head_dim]
            key: Current key states to add to the cache.
                Shape: [batch, query_len, num_key_heads, key_dim]
            value: Current value states to add to the cache.
                Shape: [batch, query_len, num_value_heads, value_dim]
            mode: Runtime mode (e.g., MODE_PREFILL) for sharding resolution.
            quantizer: Quantizer to apply to stored cache tensors.
            cache_metadata: Runtime metadata (TransformerMetadata,
                OperationsMetadata, or HybridMetadata). Explicit indexes override
                the view's write positions, and starts exclude earlier KV slots.
                With postpadded=True only valid incoming tokens advance indexes;
                padding is identified from mask_info and is not retained in a
                sliding window.
            mask_info: MaskInfo object with attention mask configuration.
                KV dimension is expanded automatically if needed.
            runtime_sharding_resolver: Runtime sharding resolver for cache tensors.
            partition_manager: Deprecated compatibility alias for
                ``runtime_sharding_resolver``.
            causal: Whether to bake causal visibility into query-local masks.
                False retains future keys in the current append and applies a
                symmetric local window when sliding attention is configured.

        Returns:
            tuple: Five-element tuple containing:
                - key_cache: Attention-time keys in runtime dtype. Sliding
                  multi-token appends return the old tail plus current block;
                  only the final window is stored in updated_view.
                - value_cache: Corresponding attention-time values.
                - mask_info: Updated MaskInfo with applied KV lengths.
                - updated_view: New TransformerCacheView with advanced positions.
                - masking_details: AttnMaskDetail for attention computation.
        """
        from easydel.infra.utils import AttnMaskType

        runtime_dtype = query.dtype
        num_updated_cache_vectors = query.shape[1]
        masking_details = self.masking_details
        indexes, starts, new_lengths = _append_metadata(
            self.indexes, self.starts, mask_info, num_updated_cache_vectors, cache_metadata
        )
        runtime_sharding_resolver = coerce_runtime_sharding_resolver(
            runtime_sharding_resolver if runtime_sharding_resolver is not None else partition_manager
        )
        sharding_statics = dict(mode=mode, partition_manager=runtime_sharding_resolver)

        cache_kv_len = self.key.shape[1]

        def _kv_struct_shard(
            x: Float[JAXArray, "batch seq_len num_heads head_dim"],
        ) -> Float[JAXArray, "batch seq_len num_heads head_dim"]:
            """Apply the view's KV logical sharding axes to ``x``.

            Args:
                x: KV tensor to shard.

            Returns:
                The same tensor with its sharding refreshed.
            """
            axes = getattr(self, "kv_sharding_axes", (BATCH, KV_LENGTH, KV_HEAD, KV_HEAD_DIM))
            return tp.cast(JAXArray, apply_logical_sharding(x, axes=axes, **sharding_statics))

        sliding_window = (
            masking_details.size
            if masking_details is not None and masking_details.mask_type == AttnMaskType.SLIDING
            else None
        )
        value_cache_updated = _append_cache_rows(
            _maybe_materialize(self.value), value, indexes, new_lengths, sliding=sliding_window is not None
        )
        key_cache_updated = _append_cache_rows(
            _maybe_materialize(self.key), key, indexes, new_lengths, sliding=sliding_window is not None
        )

        write_index = indexes
        indexes = indexes + new_lengths
        mask_info, segment_ids_updated = _append_cache_mask(
            mask_info,
            cache_kv_len,
            write_index,
            new_lengths,
            starts,
            self.segment_ids,
            num_updated_cache_vectors,
            sliding_window=sliding_window,
            causal=causal,
        )
        attention_key = key_cache_updated
        attention_value = value_cache_updated
        if sliding_window is not None and num_updated_cache_vectors > 1:
            # Earlier queries need the old tail plus the complete current block,
            # whereas persistent storage only retains the final window.
            attention_key = _attention_cache_rows(_maybe_materialize(self.key), key, write_index, new_lengths)
            attention_value = _attention_cache_rows(_maybe_materialize(self.value), value, write_index, new_lengths)

        # Keep the cache storage dtype stable (matches the cache allocation dtype) while
        # returning KV in runtime dtype for the attention computation.
        value_cache_storage = _kv_struct_shard(value_cache_updated)
        key_cache_storage = _kv_struct_shard(key_cache_updated)
        batch_axes = getattr(self, "batch_sharding_axes", (BATCH,))
        indexes_updated = apply_logical_sharding(indexes, axes=batch_axes, **sharding_statics)
        starts_updated = apply_logical_sharding(starts, axes=batch_axes, **sharding_statics)

        value_cache_out = _kv_struct_shard(attention_value).astype(runtime_dtype)
        key_cache_out = _kv_struct_shard(attention_key).astype(runtime_dtype)

        return (
            key_cache_out,
            value_cache_out,
            mask_info,
            self.replace(
                key=quantizer(key_cache_storage),
                value=quantizer(value_cache_storage),
                indexes=indexes_updated,
                starts=starts_updated,
                segment_ids=apply_logical_sharding(
                    segment_ids_updated, axes=self.kv_sharding_axes[:2], **sharding_statics
                ),
            ),
            masking_details,
        )

    def __repr__(self):
        """Return a short ``repr`` showing key/value shapes and layer index.

        Falls back to printing the raw values when shape attributes aren't
        available (for example before allocation).

        Returns:
            A human-readable representation of the cache view.
        """
        try:
            return (
                self.__class__.__name__
                + f"(key={self.key.shape}, value={self.value.shape}, layer_index={self.layer_index})"
            )
        except AttributeError:
            return self.__class__.__name__ + f"(key={self.key}, value={self.value}, layer_index={self.layer_index})"

    @property
    def is_empty(self) -> bool:
        """Whether this view has no allocated key tensor.

        Returns:
            ``True`` when ``self.key`` is ``None`` (placeholder/uninitialized).
        """
        return self.key is None

    __str__ = __repr__


@auto_pytree
class TransformerCache(BaseCache):
    """Multi-layer transformer cache container.

    Orchestrates cache views across all transformer layers, providing
    methods for initialization, access, and batch operations. Supports
    serialization for checkpointing and cache transfer.

    The cache maintains:
    - Ordered list of per-layer cache views
    - Consistent configuration across layers
    - Batch update operations
    - Serialization/deserialization support

    Attributes:
        views (list[TransformerCacheView | None]): Cache views for each layer.
            None indicates uninitialized layer.
    """

    views: list[TransformerCacheView | None]

    @classmethod
    def init_cache(
        cls,
        mesh: MeshLike,
        config: TransformerCacheConfig,
        runtime_sharding_resolver: RuntimeShardingResolver | PartitionManager | None = None,
        partition_manager: RuntimeShardingResolver | PartitionManager | None = None,
        dtype: jnp.dtype | None = None,
        starts: Int[JAXArray, "batch"] | None = None,  # noqa: F821
        quantizer: EasyQuantizer | None = None,
        mask_type_details: dict[int, AttnMaskDetail] | None = None,
    ):
        """Initialize a complete transformer cache with views for all layers.

        Creates a fully initialized KV cache with allocated storage for each
        transformer layer, applying consistent sharding and quantization.

        Args:
            mesh: JAX device mesh for distributed execution.
            config: TransformerCacheConfig with cache dimensions and behavior.
            runtime_sharding_resolver: Runtime sharding resolver for cache tensors.
            partition_manager: Deprecated compatibility alias for
                ``runtime_sharding_resolver``.
            dtype: Data type for cache tensors. Defaults to jnp.bfloat16.
            starts: Initial starting positions per batch sequence.
            quantizer: Optional quantizer to apply to cache tensors.
            mask_type_details: Per-layer attention mask configuration, keyed
                by layer index.

        Returns:
            TransformerCache: Fully initialized cache ready for inference.
        """
        from easydel.layers.quantization._quants import EasyQuantizer

        quantizer = quantizer or EasyQuantizer(quantization_config=None)
        if dtype is None:
            dtype = jnp.bfloat16
        runtime_sharding_resolver = coerce_runtime_sharding_resolver(
            runtime_sharding_resolver if runtime_sharding_resolver is not None else partition_manager,
            mesh=mesh,
        )
        with mesh:
            return cls(
                views=[
                    TransformerCacheView.init(
                        config=config,
                        layer_index=layer_index,
                        mesh=mesh,
                        dtype=dtype,
                        starts=starts,
                        quantizer=quantizer,
                        masking_details=mask_type_details.get(layer_index) if mask_type_details is not None else None,
                        runtime_sharding_resolver=runtime_sharding_resolver,
                    )
                    for layer_index in range(config.num_hidden_layers)
                ]
            )

    @classmethod
    def init_empty(cls, num_hidden_layers: tp.Any) -> TransformerCache:
        """Initialize an empty transformer cache without allocated storage.

        Creates a cache structure with None views that can be populated later.

        Args:
            num_hidden_layers: Number of layers to create placeholders for. For
                convenience an object exposing ``num_hidden_layers`` (a model
                config) is also accepted.

        Returns:
            TransformerCache: Cache instance with uninitialized (None) views.
        """
        if not isinstance(num_hidden_layers, int):
            num_hidden_layers = getattr(num_hidden_layers, "num_hidden_layers", num_hidden_layers)
        return cls(views=[None for _ in range(num_hidden_layers)])

    def __repr__(self):
        """Return a multi-line ``repr`` listing each layer's view.

        Returns:
            A string containing one ``repr`` per layer, indented for clarity.
        """
        return f"{self.__class__.__name__}(\n  " + "\n  ".join(str(view) for view in self.views) + "\n)"

    __str__ = __repr__


@auto_pytree
class TransformerMetadata(BaseRunTimeMetadata):
    """Runtime metadata for transformer cache operations.

    Holds dynamic information needed during cache updates that isn't
    part of the permanent cache state. This includes temporary indices
    and flags for specific computation modes.

    Attributes:
        postpadded (bool): Whether sequences are post-padded.
            Affects mask generation and position calculations.
        starts (jax.Array | None): Starting positions for sequences.
            Used for relative position calculations.
        indexes (jax.Array | None): Current position indices.
            Tracks generation progress per sequence.
    """

    postpadded: bool = field(pytree_node=False, default=False)
    starts: Int[JAXArray, "batch"] | None = None  # noqa: F821
    indexes: Int[JAXArray, "batch"] | None = None  # noqa: F821
