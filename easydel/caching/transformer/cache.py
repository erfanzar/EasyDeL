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
from functools import partial

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
    """

    batch_size: int = field(pytree_node=False)
    sequence_length: int = field(pytree_node=False)
    num_hidden_layers: int = field(pytree_node=False)
    pad_token_id: int = field(pytree_node=False)
    # Optional attention-related fields
    num_heads: int | None = field(pytree_node=False)
    head_dim: int | None = field(pytree_node=False)
    key_heads: int | None = field(pytree_node=False)
    value_heads: int | None = field(pytree_node=False)
    key_dim: int | None = field(pytree_node=False)
    value_dim: int | None = field(pytree_node=False)
    sliding_window: int | None = field(pytree_node=False)

    # Configuration flags
    update_causal_mask: bool = field(pytree_node=False)
    create_attention_bias: bool = field(pytree_node=False)

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

        if head_dim is not None:
            key_dim = key_dim or head_dim
            value_dim = value_dim or head_dim
        else:
            if key_dim is None or value_dim is None:
                raise ValueError("Either head_dim or both key_dim and value_dim must be specified")

        # Derive heads from num_heads if not specified
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
            )
        return out

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
                OperationsMetadata, or HybridMetadata). Unwrapped internally.
            mask_info: MaskInfo object with attention mask configuration.
                KV dimension is expanded automatically if needed.
            runtime_sharding_resolver: Runtime sharding resolver for cache tensors.
            partition_manager: Deprecated compatibility alias for
                ``runtime_sharding_resolver``.

        Returns:
            tuple: Five-element tuple containing:
                - key_cache: Updated key cache in runtime dtype.
                - value_cache: Updated value cache in runtime dtype.
                - mask_info: Updated MaskInfo with applied KV lengths.
                - updated_view: New TransformerCacheView with advanced positions.
                - masking_details: AttnMaskDetail for attention computation.
        """
        from easydel.infra.utils import AttnMaskType

        # Unwrap OperationsMetadata to TransformerMetadata if needed
        cache_metadata = unwrap_metadata(cache_metadata, "transformer")

        runtime_dtype = query.dtype
        num_updated_cache_vectors = query.shape[1]
        masking_details = self.masking_details
        indexes = self.indexes
        runtime_sharding_resolver = coerce_runtime_sharding_resolver(
            runtime_sharding_resolver if runtime_sharding_resolver is not None else partition_manager
        )
        sharding_statics = dict(mode=MODE_PREFILL, partition_manager=runtime_sharding_resolver)

        # Expand mask KV dimension if it doesn't match cache size
        # This is needed when using HybridCache with TransformerCacheView
        key_shape = self.key.shape if hasattr(self.key, "shape") else None
        cache_kv_len = key_shape[1] if key_shape is not None else 0
        mask_kv_len = mask_info.kv_len
        if mask_kv_len is not None and mask_kv_len < cache_kv_len:
            mask_info = _expand_mask_kv_dim(mask_info, cache_kv_len, indexes, num_updated_cache_vectors)

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
            return apply_logical_sharding(x, axes=axes, **sharding_statics)

        @partial(jax.vmap, in_axes=(0, 0, 0), out_axes=(0))
        def _update_kv(
            old: Float[JAXArray, "seq_len num_heads head_dim"],
            new: Float[JAXArray, "query_len num_heads head_dim"],
            slot: Int[JAXArray, ""],
        ) -> Float[JAXArray, "seq_len num_heads head_dim"]:
            """Insert ``new`` into ``old`` at ``slot`` along the sequence dim.

            Vectorized over the batch dimension via ``jax.vmap``.

            Args:
                old: Existing per-batch cache slice.
                new: New tokens to write.
                slot: Sequence position to begin writing at.

            Returns:
                The updated cache slice for this batch element.
            """
            return lax.dynamic_update_slice(old, new.astype(old.dtype), (slot, 0, 0))

        @partial(jax.vmap, in_axes=(0, 0, 0), out_axes=(0))
        def _update_kv_sliding(
            old_cache: Float[JAXArray, "window_size num_heads head_dim"],
            new_values: Float[JAXArray, "query_len num_heads head_dim"],
            current_index: Int[JAXArray, ""],
        ) -> Float[JAXArray, "window_size num_heads head_dim"]:
            """Update a sliding-window KV cache slot in-place.

            Args:
                old_cache: The existing window-sized slot.
                new_values: New tokens to append.
                current_index: Logical write position into the window.

            Returns:
                The updated window contents, either by overwriting in place or
                by rolling the buffer when the new tokens overflow the window.
            """
            new_len = new_values.shape[0]
            window_size = old_cache.shape[0]
            if new_len >= window_size:
                return new_values[-window_size:, :, :].astype(old_cache.dtype)

            total_tokens = current_index + new_len

            def _fits_in_window():
                """Branch: write at ``current_index`` without rolling."""
                return lax.dynamic_update_slice(old_cache, new_values.astype(old_cache.dtype), (current_index, 0, 0))

            def _overflow_window():
                """Branch: shift older tokens out and append the new ones."""
                return jnp.concatenate([old_cache[new_len:, :, :], new_values.astype(old_cache.dtype)], axis=0)

            return lax.cond(total_tokens <= window_size, _fits_in_window, _overflow_window)

        sliding_window = None

        if masking_details is not None and masking_details.mask_type == AttnMaskType.SLIDING:
            value_cache_updated = _update_kv_sliding(_maybe_materialize(self.value), value, indexes)
            key_cache_updated = _update_kv_sliding(_maybe_materialize(self.key), key, indexes)
            sliding_window = masking_details.size
        else:
            value_cache_updated = _update_kv(_maybe_materialize(self.value), value, indexes)
            key_cache_updated = _update_kv(_maybe_materialize(self.key), key, indexes)

        indexes = indexes + num_updated_cache_vectors
        mask_info = mask_info.apply_kv_lengths(
            kv_lengths=indexes,
            q_len=num_updated_cache_vectors,
            end_index=indexes,
            sliding_window=sliding_window,
        )

        # Keep the cache storage dtype stable (matches the cache allocation dtype) while
        # returning KV in runtime dtype for the attention computation.
        value_cache_storage = _kv_struct_shard(value_cache_updated)
        key_cache_storage = _kv_struct_shard(key_cache_updated)
        batch_axes = getattr(self, "batch_sharding_axes", (BATCH,))
        indexes_updated = apply_logical_sharding(indexes, axes=batch_axes, **sharding_statics)

        value_cache_out = value_cache_storage.astype(runtime_dtype)
        key_cache_out = key_cache_storage.astype(runtime_dtype)

        return (
            key_cache_out,
            value_cache_out,
            mask_info,
            self.replace(
                key=quantizer(key_cache_storage),
                value=quantizer(value_cache_storage),
                indexes=indexes_updated,
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

    def to_pure(self) -> tuple[list[list[JAXArray | ImplicitArray | None]], TransformerCacheConfig]:
        """Convert cache to pure Python data structure for serialization.

        Extracts raw tensors and metadata for checkpointing or transfer.
        The pure representation can be pickled or saved to disk.

        Returns:
            tuple: Pair of (cache_data, metadata) where:
                - cache_data: List of [key, value, indexes, starts] per layer
                - metadata: Cache configuration metadata
        """
        return (
            [[layer.key, layer.value, layer.indexes, layer.starts] for layer in self.views],
            self.views[-1].metadata,
        )

    @classmethod
    def from_pure(
        cls, pure: list[list[JAXArray | ImplicitArray | None]], metadata: TransformerCacheConfig
    ) -> TransformerCache:
        """Reconstruct cache from pure Python data structure.

        Restores a cache from serialized tensors and metadata,
        typically after loading from disk or receiving from transfer.

        Args:
            pure: List of [key, value, indexes, starts] per layer.
            metadata: Cache configuration metadata.

        Returns:
            TransformerCache: Reconstructed cache instance.
        """
        return cls(
            views=[
                TransformerCacheView(
                    key=layer[0],
                    value=layer[1],
                    indexes=layer[2],
                    starts=layer[3],
                    metadata=metadata,
                )
                for layer in pure
            ]
        )

    def insert_starts(
        self,
        starts: Int[JAXArray, "..."],
        slot: int,
        runtime_sharding_resolver: RuntimeShardingResolver | PartitionManager | None = None,
        partition_manager: RuntimeShardingResolver | PartitionManager | None = None,
    ) -> TransformerCache:
        """Insert starting positions at specified batch slot.

        Updates the starting position indices for a specific batch slot
        across all layers. Used for dynamic batching and cache management.

        Args:
            starts: New starting positions to insert.
            slot (int): Batch slot index to update.
            runtime_sharding_resolver: Runtime sharding resolver for cache tensors.
            partition_manager: Deprecated compatibility alias for
                ``runtime_sharding_resolver``.

        Returns:
            TransformerCache: Updated cache instance.
        """
        runtime_sharding_resolver = coerce_runtime_sharding_resolver(
            runtime_sharding_resolver if runtime_sharding_resolver is not None else partition_manager
        )
        for idx in range(len(self.views)):
            view = self.views[idx]
            starts = jnp.array(starts).reshape(-1)

            self.views[idx] = self.views[idx].replace(
                starts=apply_logical_sharding(
                    lax.dynamic_update_slice_in_dim(view.starts, starts, slot, 0),
                    axes=[BATCH],
                    mode=MODE_PREFILL,
                    partition_manager=runtime_sharding_resolver,
                )
            )
        return self

    def insert_index(
        self,
        index: Int[JAXArray, "..."],
        slot: int,
        runtime_sharding_resolver: RuntimeShardingResolver | PartitionManager | None = None,
        partition_manager: RuntimeShardingResolver | PartitionManager | None = None,
    ) -> TransformerCache:
        """Insert position indices at specified batch slot.

        Updates the current position indices for a specific batch slot
        across all layers. Used for tracking generation progress.

        Args:
            index: New position index to insert.
            slot (int): Batch slot index to update.
            runtime_sharding_resolver: Runtime sharding resolver for cache tensors.
            partition_manager: Deprecated compatibility alias for
                ``runtime_sharding_resolver``.

        Returns:
            TransformerCache: Updated cache instance.
        """
        runtime_sharding_resolver = coerce_runtime_sharding_resolver(
            runtime_sharding_resolver if runtime_sharding_resolver is not None else partition_manager
        )
        for idx in range(len(self.views)):
            view = self.views[idx]
            index = jnp.array(index).reshape(-1)
            self.views[idx] = self.views[idx].replace(
                indexes=apply_logical_sharding(
                    lax.dynamic_update_slice_in_dim(view.indexes, index, slot, 0),
                    axes=[BATCH],
                    mode=MODE_PREFILL,
                    partition_manager=runtime_sharding_resolver,
                )
            )
        return self

    def insert(
        self,
        other: TransformerCache,
        slot: int,
        quantizer: EasyQuantizer,
        runtime_sharding_resolver: RuntimeShardingResolver | PartitionManager | None = None,
        partition_manager: RuntimeShardingResolver | PartitionManager | None = None,
    ):
        """Insert another cache's contents at specified batch slot.

        Copies key-value states and indices from another cache into
        this cache at the specified batch position. Useful for
        batched generation with different sequences.

        Args:
            other (TransformerCache): Source cache to copy from.
            slot (int): Batch slot index to insert into.
            quantizer (EasyQuantizer): Quantization configuration.
            runtime_sharding_resolver: Runtime sharding resolver for cache tensors.
            partition_manager: Deprecated compatibility alias for
                ``runtime_sharding_resolver``.

        Returns:
            TransformerCache: Updated cache instance.
        """
        runtime_sharding_resolver = coerce_runtime_sharding_resolver(
            runtime_sharding_resolver if runtime_sharding_resolver is not None else partition_manager
        )

        for idx in range(len(self.views)):
            view = self.views[idx]
            oview = other.views[idx]

            new_val = lax.dynamic_update_slice(
                _maybe_materialize(view.value),
                _maybe_materialize(oview.value.astype(view.value.dtype)),
                (slot, 0, 0, 0),
            )
            new_key = lax.dynamic_update_slice(
                _maybe_materialize(view.key),
                _maybe_materialize(oview.key.astype(view.key.dtype)),
                (slot, 0, 0, 0),
            )

            self.views[idx] = self.views[idx].replace(
                key=quantizer(
                    apply_logical_sharding(
                        new_key,
                        axes=[BATCH, KV_LENGTH, KV_HEAD, KV_HEAD_DIM],
                        mode=MODE_PREFILL,
                        partition_manager=runtime_sharding_resolver,
                    )
                ),
                value=quantizer(
                    apply_logical_sharding(
                        new_val,
                        axes=[BATCH, KV_LENGTH, KV_HEAD, KV_HEAD_DIM],
                        mode=MODE_PREFILL,
                        partition_manager=runtime_sharding_resolver,
                    )
                ),
                indexes=apply_logical_sharding(
                    lax.dynamic_update_slice_in_dim(view.indexes, oview.indexes, slot, 0),
                    axes=[BATCH],
                    mode=MODE_PREFILL,
                    partition_manager=runtime_sharding_resolver,
                ),
                starts=apply_logical_sharding(
                    lax.dynamic_update_slice_in_dim(view.starts, oview.starts, slot, 0),
                    axes=[BATCH],
                    mode=MODE_PREFILL,
                    partition_manager=runtime_sharding_resolver,
                ),
                metadata=view.metadata,
            )
        return self

    @classmethod
    def init_empty(cls, num_hidden_layers: int) -> TransformerCache:
        """Initialize an empty transformer cache without allocated storage.

        Creates a cache structure with None views that can be populated later.

        Args:
            num_hidden_layers: Number of layers to create placeholders for.

        Returns:
            TransformerCache: Cache instance with uninitialized (None) views.
        """
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
