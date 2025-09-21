# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Lightning attention cache implementation for EasyDeL.

This module provides a specialized caching system for Lightning attention,
which uses a unified key-value representation for improved efficiency.
Lightning attention combines keys and values into a single tensor,
reducing memory bandwidth requirements.

Key Components:
    - LightningCacheMetaData: Configuration for Lightning cache
    - LightningCacheView: Per-layer Lightning cache storage
    - LightningCache: Multi-layer Lightning cache container
    - LightningMetadata: Runtime metadata (placeholder)

Features:
    - Unified KV tensor representation
    - Reduced memory bandwidth usage
    - Compatible with Lightning attention kernels
    - Supports standard transformer operations

Example:
    >>> metadata = LightningCacheMetaData.create(
    ...     partition_axis=partition_axis,
    ...     batch_size=2,
    ...     num_heads=16,
    ...     head_dim=64
    ... )
    >>> cache = LightningCache.init_cache(
    ...     num_hidden_layers=12,
    ...     metadata=metadata
    ... )
"""

from __future__ import annotations

import typing as tp

import jax
from eformer import escale as es
from eformer.jaximus import ImplicitArray
from eformer.pytree import auto_pytree
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array, Bool, Float, Int

from .._abstracts import BaseCache, BaseCacheMetadata, BaseCacheView, BaseRunTimeMetadata

if tp.TYPE_CHECKING:
    from easydel.layers.quantization.quantizers import EasyQuantizer
else:
    EasyQuantizer = object


@auto_pytree
class LightningCacheMetaData(BaseCacheMetadata):
    """Metadata configuration for Lightning attention cache.

    Stores configuration parameters specific to Lightning attention,
    which uses a unified key-value representation. Similar to standard
    transformer cache but optimized for Lightning's memory access patterns.

    Attributes:
        partition_axis (es.PartitionAxis): Axis configuration for tensor partitioning.
            Defines how tensors are sharded across devices.
        batch_size (int | None): Number of sequences in batch.
            None allows dynamic batch sizes.
        num_heads (int | None): Number of attention heads.
            Used for standard multi-head attention.
        head_dim (int | None): Dimension of each attention head.
            Defines the feature size per head.
        key_heads (int | None): Number of key heads.
            For multi-query or grouped-query attention.
        value_heads (int | None): Number of value heads.
            For multi-query or grouped-query attention.
        key_dim (int | None): Dimension of key projections.
            Can differ from head_dim for asymmetric attention.
        value_dim (int | None): Dimension of value projections.
            Can differ from head_dim for asymmetric attention.
    """

    partition_axis: es.PartitionAxis
    batch_size: int | None
    num_heads: int | None
    head_dim: int | None
    key_heads: int | None
    value_heads: int | None
    key_dim: int | None
    value_dim: int | None

    @classmethod
    def create(
        cls,
        partition_axis: es.PartitionAxis,
        batch_size: int | None = None,
        num_heads: int | None = None,
        head_dim: int | None = None,
        key_heads: int | None = None,
        value_heads: int | None = None,
        key_dim: int | None = None,
        value_dim: int | None = None,
    ) -> LightningCacheMetaData:
        """Create and validate Lightning cache metadata.

        Factory method for creating Lightning cache configuration.
        Unlike standard transformer cache, Lightning allows more
        flexibility in parameters as it handles unified KV tensors.

        Args:
            partition_axis (es.PartitionAxis): Tensor partitioning configuration.
            batch_size (int | None): Batch size for cache allocation.
                None for dynamic batching.
            num_heads (int | None): Number of attention heads.
                Defaults to None for flexible head configuration.
            head_dim (int | None): Dimension per attention head.
                Defaults to None for flexible dimensions.
            key_heads (int | None): Number of key heads for MQA/GQA.
                Defaults to None (same as num_heads).
            value_heads (int | None): Number of value heads for MQA/GQA.
                Defaults to None (same as num_heads).
            key_dim (int | None): Key projection dimension.
                Defaults to None (same as head_dim).
            value_dim (int | None): Value projection dimension.
                Defaults to None (same as head_dim).

        Returns:
            LightningCacheMetaData: Configured metadata instance.

        Note:
            Lightning attention's unified representation means some
            parameters may be handled differently than standard cache.
        """
        return cls(
            partition_axis=partition_axis,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            key_heads=key_heads,
            value_heads=value_heads,
            key_dim=key_dim,
            value_dim=value_dim,
        )


@auto_pytree
class LightningCacheView(BaseCacheView):
    """Single-layer cache view for Lightning attention.

    Manages the unified key-value cache for one layer using Lightning's
    optimized representation. Unlike standard caches that store keys and
    values separately, Lightning combines them for better memory efficiency.

    Attributes:
        key_value (cx.Array | ImplicitArray): Unified key-value tensor.
            Lightning's special representation combining K and V.
        metadata (LightningCacheMetaData): Static cache configuration.
        layer_index (int | None): Index of this layer in the model.

    Note:
        The unified representation requires special handling during
        concatenation and may not be compatible with standard attention.
    """

    key_value: Float[Array, "batch seq_len num_heads head_dim"] | ImplicitArray | None
    metadata: LightningCacheMetaData
    layer_index: int | None = None

    @classmethod
    def init(cls, metadata: LightningCacheMetaData, layer_index: int | None = None) -> LightningCacheView:
        """Initialize a Lightning cache view for a single layer.

        Creates a cache view with placeholder for unified KV tensor.
        Actual tensor allocation happens during first use to allow
        for dynamic shapes.

        Args:
            metadata (LightningCacheMetaData): Cache configuration.
            layer_index (int | None): Layer index in the model.

        Returns:
            LightningCacheView: Initialized view with None placeholder.
        """
        return cls(
            key_value=None,
            metadata=metadata,
            layer_index=layer_index,
        )

    @jax.named_scope("easydel-lightning-cacheview-concatenate-to-cache")
    def concatenate_to_cache(
        self,
        query: Float[Array, "batch query_len num_heads head_dim"],
        key: Float[Array, "batch query_len num_key_heads key_dim"],
        value: Float[Array, "batch query_len num_value_heads value_dim"],
        attention_mask: Bool[Array, "batch 1 query_len seq_len"] | Float[Array, "batch 1 query_len seq_len"],
        kv_sharding: PartitionSpec,
        quantizer: EasyQuantizer,
        causal_mask: Bool[Array, "batch 1 query_len seq_len"] | bool | None = None,
        token_type_ids: Int[Array, "batch query_len"] | None = None,
    ) -> tuple[
        Float[Array, "batch seq_len num_key_heads key_dim"],
        Float[Array, "batch seq_len num_value_heads value_dim"],
        Bool[Array, "batch 1 query_len seq_len"],
    ]:
        """Update cache with new key/value states for Lightning attention.

        Concatenates new key and value states to the cache using Lightning's
        unified KV representation. This method is called during each forward
        pass to update the cache with newly computed states.

        Note: This implementation appears to need refactoring as it references
        attributes (self.index, self.key, self.value) that don't exist in the
        current class definition. The actual Lightning implementation would
        use the unified self.key_value tensor.

        Args:
            query: Query tensor with shape [batch, query_len, num_heads, head_dim].
                Used to determine update dimensions.
            key: Key tensor with shape [batch, query_len, num_key_heads, key_dim].
                New keys to add to the cache.
            value: Value tensor with shape [batch, query_len, num_value_heads, value_dim].
                New values to add to the cache.
            attention_mask: Boolean or float mask with shape [batch, 1, query_len, seq_len].
                Defines which positions can attend to which.
            kv_sharding: JAX PartitionSpec for sharding the KV cache.
            quantizer: Quantization function for cache compression.
            causal_mask: Optional causal mask for autoregressive attention.
                Can be boolean array or boolean value.
            token_type_ids: Optional token type IDs for segment-level masking.
                Shape [batch, query_len].

        Returns:
            Tuple containing:
                - Updated key cache: Float[Array, "batch seq_len num_key_heads key_dim"]
                - Updated value cache: Float[Array, "batch seq_len num_value_heads value_dim"]
                - Updated attention mask: Bool[Array, "batch 1 query_len seq_len"]

        Raises:
            NotImplementedError: Current implementation needs refactoring to properly
                use Lightning's unified KV representation.
        """
        num_updated_cache_vectors = query.shape[1]
        end_index = self.index[0]

        *batch_dims, max_length, _num_heads, _depth_per_head = self.value.shape

        if attention_mask.ndim == 2:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        if causal_mask is not None:
            if hasattr(causal_mask, "value"):
                causal_mask = causal_mask.value
            causal_mask = jax.lax.dynamic_slice(
                causal_mask,
                (0, 0, end_index, 0),
                (1, 1, num_updated_cache_vectors, max_length),
            )
            if token_type_ids is not None and num_updated_cache_vectors != 1:
                token_type_mask = jnp.equal(
                    jnp.expand_dims(token_type_ids, 2),
                    jnp.expand_dims(token_type_ids, 1),
                )

                token_type_mask = jnp.where(token_type_ids == 0, False, token_type_mask)
                token_type_mask = jnp.expand_dims(token_type_mask, 1)
                sequence_length = token_type_ids.shape[1]
                masked_portion = jnp.logical_or(
                    token_type_mask[:, :, :num_updated_cache_vectors, :],
                    causal_mask[:, :, :, :sequence_length],
                )
                causal_mask = causal_mask.at[:, :, :, :sequence_length].set(masked_portion)

            causal_mask = jnp.broadcast_to(causal_mask, (query.shape[0], *causal_mask.shape[1:]))

            attention_mask = jnp.broadcast_to(attention_mask, causal_mask.shape)
            attention_mask = jnp.logical_and(attention_mask, causal_mask)

        slice_indices = (0, end_index % self.value.shape[1], 0, 0)

        value_cache = es.with_sharding_constraint(
            jax.lax.dynamic_update_slice(
                self.value,
                value.astype(self.value.dtype),
                slice_indices,
            ),
            sharding=kv_sharding,
        )
        key_cache = es.with_sharding_constraint(
            jax.lax.dynamic_update_slice(
                self.key,
                key.astype(self.key.dtype),
                slice_indices,
            ),
            sharding=kv_sharding,
        )
        pad_mask = jnp.broadcast_to(
            jnp.arange(max_length) < end_index + num_updated_cache_vectors,
            (*tuple(batch_dims), 1, num_updated_cache_vectors, max_length),
        )
        attention_mask = jnp.logical_and(pad_mask, attention_mask)

        self.key = quantizer(key_cache)
        self.value = quantizer(value_cache)

        self.index = self.index + num_updated_cache_vectors
        return key_cache, value_cache, attention_mask

    def __repr__(self) -> str:
        return (
            self.__class__.__name__ + f"(key={self.key.shape}, value={self.value.shape}, layer_index={self.layer_index})"
        )

    __str__ = __repr__


@auto_pytree
class LightningCache(BaseCache):
    """Multi-layer Lightning attention cache container.

    Orchestrates Lightning cache views across all model layers,
    providing unified management of the specialized Lightning
    attention cache format.

    Attributes:
        views (list[LightningCacheView | None]): Per-layer cache views.
            None for uninitialized layers.
    """

    views: list[LightningCacheView | None]

    @classmethod
    def init_cache(
        cls,
        num_hidden_layers: int,
        metadata: LightningCacheMetaData,
    ) -> LightningCache:
        """Initialize Lightning cache for all model layers.

        Creates cache views for each layer with consistent configuration.
        Views are initialized with placeholders; actual tensors are
        allocated on first use.

        Args:
            num_hidden_layers (int): Number of layers in the model.
            metadata (LightningCacheMetaData): Cache configuration.

        Returns:
            LightningCache: Initialized cache with views for all layers.
        """
        return cls(
            views=[
                LightningCacheView.init(metadata=metadata, layer_index=layer_index)
                for layer_index in range(num_hidden_layers)
            ]
        )

    @classmethod
    def init_empty(cls, num_hidden_layers: int) -> LightningCache:
        """Initialize empty Lightning cache structure.

        Creates cache container with None placeholders for all layers.
        Useful for gradual cache building or testing.

        Args:
            num_hidden_layers (int): Number of layer slots to create.

        Returns:
            LightningCache: Empty cache structure.
        """
        return cls(views=[None for _ in range(num_hidden_layers)])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n  " + "\n  ".join(str(view) for view in self.views) + "\n)"

    __str__ = __repr__


class LightningMetadata(BaseRunTimeMetadata):
    """Runtime metadata for Lightning attention cache operations.

    Placeholder class for future Lightning-specific runtime metadata.
    Currently empty but reserved for Lightning-specific runtime state.
    """

    ...
