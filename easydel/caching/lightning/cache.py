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

"""Lightning attention cache implementation for EasyDeL.

This module provides a specialized caching system for Lightning attention,
which uses a unified key-value representation for improved efficiency.
Lightning attention combines keys and values into a single tensor,
reducing memory bandwidth requirements.

Key Components:
    - LightningCacheConfig: Configuration for Lightning cache
    - LightningCacheView: Per-layer Lightning cache storage
    - LightningCache: Multi-layer Lightning cache container
    - LightningMetadata: Runtime metadata (placeholder)

Features:
    - Unified KV tensor representation
    - Reduced memory bandwidth usage
    - Compatible with Lightning attention kernels
    - Supports standard transformer operations

Example:
    >>> metadata = LightningCacheConfig.create(
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
from eformer.jaximus import ImplicitArray
from eformer.pytree import auto_pytree, field
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array, Bool, Float, Int
from spectrax import PartitionAxis

from .._abstracts import BaseCache, BaseCacheConfig, BaseCacheView, BaseRunTimeMetadata

if tp.TYPE_CHECKING:
    from easydel.layers.quantization._quants import EasyQuantizer
else:
    EasyQuantizer = object


@auto_pytree
class LightningCacheConfig(BaseCacheConfig):
    """Metadata configuration for Lightning attention cache.

    Stores configuration parameters specific to Lightning attention,
    which uses a unified key-value representation. Similar to standard
    transformer cache but optimized for Lightning's memory access patterns.

    Attributes:
        partition_axis (PartitionAxis): Axis configuration for tensor partitioning.
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

    partition_axis: PartitionAxis = field(pytree_node=False)
    batch_size: int | None = field(pytree_node=False)
    num_heads: int | None = field(pytree_node=False)
    head_dim: int | None = field(pytree_node=False)
    key_heads: int | None = field(pytree_node=False)
    value_heads: int | None = field(pytree_node=False)
    key_dim: int | None = field(pytree_node=False)
    value_dim: int | None = field(pytree_node=False)

    @classmethod
    def create(
        cls,
        partition_axis: PartitionAxis,
        batch_size: int | None = None,
        num_heads: int | None = None,
        head_dim: int | None = None,
        key_heads: int | None = None,
        value_heads: int | None = None,
        key_dim: int | None = None,
        value_dim: int | None = None,
    ) -> LightningCacheConfig:
        """Create and validate Lightning cache metadata.

        Factory method for creating Lightning cache configuration.
        Unlike standard transformer cache, Lightning allows more
        flexibility in parameters as it handles unified KV tensors.

        Args:
            partition_axis (PartitionAxis): Tensor partitioning configuration.
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
            LightningCacheConfig: Configured metadata instance.

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
        metadata (LightningCacheConfig): Static cache configuration.
        layer_index (int | None): Index of this layer in the model.

    Note:
        The unified representation requires special handling during
        concatenation and may not be compatible with standard attention.
    """

    key_value: Float[Array, "batch seq_len num_heads head_dim"] | ImplicitArray | None
    metadata: LightningCacheConfig
    layer_index: int | None = field(pytree_node=False, default=None)

    @classmethod
    def init(
        cls,
        config: LightningCacheConfig,
        layer_index: int | None = None,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        partition_specs: PartitionSpec | None = None,
    ) -> LightningCacheView:
        """Initialize a Lightning cache view for a single layer.

        Creates a cache view with placeholder for unified KV tensor.
        Actual tensor allocation happens during first use to allow
        for dynamic shapes.

        Args:
            config: Cache configuration.
            layer_index: Layer index in the model.
            dtype: Data type for cache tensors (unused, allocation is lazy).
            partition_specs: Sharding specification (unused, uses config.partition_axis).

        Returns:
            LightningCacheView: Initialized view with None placeholder.
        """
        return cls(
            key_value=None,
            metadata=config,
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
        """Not implemented for Lightning attention.

        Lightning attention uses a unified key_value tensor and requires a
        dedicated implementation that does not rely on separate key/value/index
        attributes.

        Raises:
            NotImplementedError: Always raised; this method is not yet implemented.
        """
        raise NotImplementedError(
            "LightningCacheView.concatenate_to_cache is not implemented. "
            "Lightning attention uses a unified key_value tensor and requires "
            "a dedicated implementation that does not rely on separate key/value/index attributes."
        )

    def __repr__(self) -> str:
        """Return a short ``repr`` showing the unified KV shape and layer index.

        Returns:
            A human-readable representation of the cache view.
        """
        kv_shape = self.key_value.shape if self.key_value is not None else None
        return self.__class__.__name__ + f"(key_value={kv_shape}, layer_index={self.layer_index})"

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
        config: LightningCacheConfig,
        dtype: jnp.dtype | None = None,
    ) -> LightningCache:
        """Initialize Lightning cache for all model layers.

        Creates cache views for each layer with consistent configuration.
        Views are initialized with placeholders; actual tensors are
        allocated on first use.

        Args:
            num_hidden_layers: Number of layers in the model.
            config: Cache configuration.
            dtype: Data type for cache tensors (unused, allocation is lazy).

        Returns:
            LightningCache: Initialized cache with views for all layers.
        """
        return cls(
            views=[
                LightningCacheView.init(config=config, layer_index=layer_index, dtype=dtype or jnp.bfloat16)
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
        """Return a multi-line ``repr`` listing each Lightning cache view.

        Returns:
            A string with one ``repr`` per layer, indented for readability.
        """
        return f"{self.__class__.__name__}(\n  " + "\n  ".join(str(view) for view in self.views) + "\n)"

    def to_pure(self) -> tuple[list[dict[str, tp.Any]], LightningCacheConfig | None]:
        """Convert cache to pure Python data for serialization.

        Returns:
            Tuple of (cache_data, metadata) where cache_data is a list of dicts
            containing serialized view data and metadata is the shared LightningCacheConfig.
        """
        cache_data: list[dict[str, tp.Any]] = []
        metadata: LightningCacheConfig | None = None

        for view in self.views:
            if view is None:
                cache_data.append({"is_none": True})
            else:
                if metadata is None:
                    metadata = view.metadata
                cache_data.append(
                    {
                        "is_none": False,
                        "key_value": view.key_value,
                        "layer_index": view.layer_index,
                    }
                )

        return cache_data, metadata

    @classmethod
    def from_pure(
        cls,
        cache_data: list[dict[str, tp.Any]],
        metadata: LightningCacheConfig | None = None,
    ) -> "LightningCache":
        """Reconstruct cache from pure Python data.

        Args:
            cache_data: List of dicts containing serialized view data.
            metadata: Shared LightningCacheConfig for reconstruction.

        Returns:
            Reconstructed LightningCache instance.
        """
        views: list[LightningCacheView | None] = []

        for layer_data in cache_data:
            if layer_data.get("is_none", False):
                views.append(None)
            else:
                view = LightningCacheView(
                    key_value=layer_data["key_value"],
                    metadata=metadata,
                    layer_index=layer_data.get("layer_index"),
                )
                views.append(view)

        return cls(views=views)

    def insert(
        self,
        other: "LightningCache",
        slot: int,
    ) -> "LightningCache":
        """Insert another cache's contents at a specific batch slot.

        Args:
            other: Source LightningCache to copy from (typically batch size 1).
            slot: Batch index to insert at.

        Returns:
            New LightningCache with the inserted content.
        """
        new_views: list[LightningCacheView | None] = []

        for self_view, other_view in zip(self.views, other.views, strict=False):
            if self_view is None or other_view is None:
                new_views.append(self_view)
            else:
                # Insert key_value at the specified slot
                if self_view.key_value is not None and other_view.key_value is not None:
                    new_key_value = self_view.key_value.at[slot].set(other_view.key_value[0])
                else:
                    new_key_value = self_view.key_value

                new_view = LightningCacheView(
                    key_value=new_key_value,
                    metadata=self_view.metadata,
                    layer_index=self_view.layer_index,
                )
                new_views.append(new_view)

        return LightningCache(views=new_views)

    __str__ = __repr__


class LightningMetadata(BaseRunTimeMetadata):
    """Runtime-side companion metadata for :class:`LightningCacheView`.

    Lightning's fused linear-recurrent kernel currently does not need extra
    per-step runtime fields beyond the cached ``key_value`` tensor itself —
    sequence positions advance implicitly with each kernel call and there
    is no padded KV table to mask. This class therefore exists as a typed
    placeholder so the unified :class:`OperationsMetadata` wrapper has a
    slot for Lightning runs and so future Lightning-specific bookkeeping
    (segment boundaries, layer-skip flags, …) can be added without
    breaking existing call sites.
    """

    ...
