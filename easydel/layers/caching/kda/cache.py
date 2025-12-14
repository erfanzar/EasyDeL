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

"""KDA (Key-Driven Attention) Cache implementation.

This module provides caching for KDA-style linear attention which uses
separate convolution states for Q, K, V projections (used in Kimi Linear).

The KDACache is designed to be composable - HybridCache uses KDACacheView
directly for its kda_linear_attention layers.
"""

from __future__ import annotations

import typing as tp

from eformer.escale import PartitionAxis, with_sharding_constraint
from eformer.jaximus import ImplicitArray
from eformer.pytree import auto_pytree, field
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array, Float, Int

from .._abstracts import BaseCache, BaseCacheConfig, BaseCacheView, BaseRunTimeMetadata


@auto_pytree
class KDACacheConfig(BaseCacheConfig):
    """Metadata for KDA cache configuration.

    Attributes:
        num_hidden_layers: Number of layers in the model.
        partition_axis: Configuration for tensor partitioning.
        batch_size: Number of sequences in batch.
        key_dim: Dimension of key (and query) projections.
        value_dim: Dimension of value projections.
        d_conv: Size of the convolution kernel.
        recurrent_state_shape: Shape of recurrent state (excluding batch).
    """

    num_hidden_layers: int = field(pytree_node=False)
    partition_axis: PartitionAxis = field(pytree_node=False)
    batch_size: int = field(pytree_node=False)
    key_dim: int = field(pytree_node=False)
    value_dim: int = field(pytree_node=False)
    d_conv: int = field(pytree_node=False)
    recurrent_state_shape: tuple[int, ...] = field(pytree_node=False)

    @classmethod
    def create(
        cls,
        num_hidden_layers: int,
        partition_axis: PartitionAxis,
        batch_size: int,
        key_dim: int,
        value_dim: int,
        d_conv: int,
        recurrent_state_shape: tuple[int, ...],
    ) -> KDACacheConfig:
        """Create and validate KDACacheConfig."""
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if key_dim <= 0:
            raise ValueError("key_dim must be positive")
        if value_dim <= 0:
            raise ValueError("value_dim must be positive")
        if d_conv <= 0:
            raise ValueError("d_conv must be positive")

        return cls(
            num_hidden_layers=num_hidden_layers,
            partition_axis=partition_axis,
            batch_size=batch_size,
            key_dim=key_dim,
            value_dim=value_dim,
            d_conv=d_conv,
            recurrent_state_shape=recurrent_state_shape,
        )


@auto_pytree
class KDACacheView(BaseCacheView):
    """Single-layer KDA cache view.

    Stores separate Q/K/V convolution states and recurrent state.

    Attributes:
        q_conv_state: Q convolution state [batch, key_dim, d_conv]
        k_conv_state: K convolution state [batch, key_dim, d_conv]
        v_conv_state: V convolution state [batch, value_dim, d_conv]
        recurrent_state: Recurrent state [batch, *recurrent_shape]
        positions: Current position per batch element
        metadata: Configuration metadata
        layer_index: Index of this layer
    """

    q_conv_state: Float[Array, "batch key_dim d_conv"] | ImplicitArray
    k_conv_state: Float[Array, "batch key_dim d_conv"] | ImplicitArray
    v_conv_state: Float[Array, "batch value_dim d_conv"] | ImplicitArray
    recurrent_state: Float[Array, "batch ..."] | ImplicitArray
    positions: Int[Array, "batch"]  # noqa:F821
    metadata: KDACacheConfig
    layer_index: int | None = field(pytree_node=False, default=None)

    @classmethod
    def init(
        cls,
        config: KDACacheConfig,
        layer_index: int | None = None,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        partition_specs: PartitionSpec | None = None,
    ) -> "KDACacheView":
        """Initialize a KDACacheView from a cache config.

        Creates cache tensors for KDA linear attention layers with
        separate Q/K/V conv states and recurrent state.

        Args:
            config: KDACacheConfig with cache dimensions.
            layer_index: Index of this layer in the model.
            dtype: Data type for cache tensors.
            partition_specs: Sharding specification (unused, uses config.partition_axis).

        Returns:
            KDACacheView: Initialized cache view.
        """
        paxis = config.partition_axis
        conv_specs = PartitionSpec(paxis.batch_axis, None, None)

        q_conv_state = with_sharding_constraint(
            arr=jnp.zeros(shape=(config.batch_size, config.key_dim, config.d_conv), dtype=dtype),
            sharding=conv_specs,
        )
        k_conv_state = with_sharding_constraint(
            arr=jnp.zeros(shape=(config.batch_size, config.key_dim, config.d_conv), dtype=dtype),
            sharding=conv_specs,
        )
        v_conv_state = with_sharding_constraint(
            arr=jnp.zeros(shape=(config.batch_size, config.value_dim, config.d_conv), dtype=dtype),
            sharding=conv_specs,
        )

        recurrent_state_shape = (config.batch_size, *config.recurrent_state_shape)
        recurrent_state = with_sharding_constraint(
            arr=jnp.zeros(shape=recurrent_state_shape, dtype=dtype),
            sharding=conv_specs,
        )

        return cls(
            q_conv_state=q_conv_state,
            k_conv_state=k_conv_state,
            v_conv_state=v_conv_state,
            recurrent_state=recurrent_state,
            positions=jnp.zeros((config.batch_size,), dtype=jnp.int32),
            metadata=config,
            layer_index=layer_index,
        )

    def concatenate_to_cache(
        self,
        q_conv_state: Float[Array, "batch key_dim d_conv"] | None = None,
        k_conv_state: Float[Array, "batch key_dim d_conv"] | None = None,
        v_conv_state: Float[Array, "batch value_dim d_conv"] | None = None,
        recurrent_state: Float[Array, "batch ..."] | None = None,
    ) -> tuple[None, None, KDACacheView]:
        """Update cache with new Q/K/V conv states and/or recurrent state."""
        new_q = q_conv_state if q_conv_state is not None else self.q_conv_state
        new_k = k_conv_state if k_conv_state is not None else self.k_conv_state
        new_v = v_conv_state if v_conv_state is not None else self.v_conv_state
        new_rec = recurrent_state if recurrent_state is not None else self.recurrent_state

        new_view = KDACacheView(
            q_conv_state=new_q,
            k_conv_state=new_k,
            v_conv_state=new_v,
            recurrent_state=new_rec,
            positions=self.positions,
            metadata=self.metadata,
            layer_index=self.layer_index,
        )

        return None, None, new_view

    def update_kda_states(
        self,
        new_q_conv_state: Float[Array, "batch key_dim d_conv"] | None = None,
        new_k_conv_state: Float[Array, "batch key_dim d_conv"] | None = None,
        new_v_conv_state: Float[Array, "batch value_dim d_conv"] | None = None,
        new_recurrent_state: Float[Array, "batch ..."] | None = None,
    ) -> KDACacheView:
        """Update KDA states."""
        _, _, new_view = self.concatenate_to_cache(
            q_conv_state=new_q_conv_state,
            k_conv_state=new_k_conv_state,
            v_conv_state=new_v_conv_state,
            recurrent_state=new_recurrent_state,
        )
        return new_view

    def reset(self) -> KDACacheView:
        """Reset cache to zeros."""
        return KDACacheView(
            q_conv_state=jnp.zeros_like(self.q_conv_state),
            k_conv_state=jnp.zeros_like(self.k_conv_state),
            v_conv_state=jnp.zeros_like(self.v_conv_state),
            recurrent_state=jnp.zeros_like(self.recurrent_state),
            positions=jnp.zeros_like(self.positions),
            metadata=self.metadata,
            layer_index=self.layer_index,
        )

    def __repr__(self) -> str:
        return (
            f"KDACacheView(q={self.q_conv_state.shape}, k={self.k_conv_state.shape}, "
            f"v={self.v_conv_state.shape}, rec={self.recurrent_state.shape}, "
            f"layer_index={self.layer_index})"
        )

    __str__ = __repr__


@auto_pytree
class KDACache(BaseCache):
    """Multi-layer KDA cache container.

    Attributes:
        views: List of KDACacheView, one per layer.
    """

    views: list[KDACacheView | None]

    @classmethod
    def init_cache(
        cls,
        config: KDACacheConfig,
        dtype: jnp.dtype | None = None,
        partition_specs: PartitionSpec | None = None,
    ) -> KDACache:
        """Initialize a complete KDA cache."""
        if dtype is None:
            dtype = jnp.bfloat16

        return cls(
            views=[
                KDACacheView.init(
                    config=config,
                    layer_index=layer_idx,
                    dtype=dtype,
                    partition_specs=partition_specs,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    @classmethod
    def init_empty(cls, num_hidden_layers: int) -> KDACache:
        """Initialize an empty KDA cache."""
        return cls(views=[None for _ in range(num_hidden_layers)])

    def reset(self) -> KDACache:
        """Reset all layers."""
        new_views = [view.reset() if view is not None else None for view in self.views]
        return KDACache(views=new_views)

    def __repr__(self) -> str:
        return f"KDACache(layers={len(self.views)})"

    def to_pure(self) -> tuple[list[dict[str, tp.Any]], KDACacheConfig | None]:
        """Convert cache to pure Python data for serialization.

        Returns:
            Tuple of (cache_data, metadata) where cache_data is a list of dicts
            containing serialized view data and metadata is the shared KDACacheConfig.
        """
        cache_data: list[dict[str, tp.Any]] = []
        metadata: KDACacheConfig | None = None

        for view in self.views:
            if view is None:
                cache_data.append({"is_none": True})
            else:
                if metadata is None:
                    metadata = view.metadata
                cache_data.append(
                    {
                        "is_none": False,
                        "q_conv_state": view.q_conv_state,
                        "k_conv_state": view.k_conv_state,
                        "v_conv_state": view.v_conv_state,
                        "recurrent_state": view.recurrent_state,
                        "positions": view.positions,
                        "layer_index": view.layer_index,
                    }
                )

        return cache_data, metadata

    @classmethod
    def from_pure(
        cls,
        cache_data: list[dict[str, tp.Any]],
        metadata: KDACacheConfig | None = None,
    ) -> "KDACache":
        """Reconstruct cache from pure Python data.

        Args:
            cache_data: List of dicts containing serialized view data.
            metadata: Shared KDACacheConfig for reconstruction.

        Returns:
            Reconstructed KDACache instance.
        """
        views: list[KDACacheView | None] = []

        for layer_data in cache_data:
            if layer_data.get("is_none", False):
                views.append(None)
            else:
                view = KDACacheView(
                    q_conv_state=layer_data["q_conv_state"],
                    k_conv_state=layer_data["k_conv_state"],
                    v_conv_state=layer_data["v_conv_state"],
                    recurrent_state=layer_data["recurrent_state"],
                    positions=layer_data["positions"],
                    metadata=metadata,
                    layer_index=layer_data.get("layer_index"),
                )
                views.append(view)

        return cls(views=views)

    def insert(
        self,
        other: "KDACache",
        slot: int,
    ) -> "KDACache":
        """Insert another cache's contents at a specific batch slot.

        Args:
            other: Source KDACache to copy from (typically batch size 1).
            slot: Batch index to insert at.

        Returns:
            New KDACache with the inserted content.
        """
        new_views: list[KDACacheView | None] = []

        for self_view, other_view in zip(self.views, other.views, strict=False):
            if self_view is None or other_view is None:
                new_views.append(self_view)
            else:
                # Insert states at the specified slot
                new_q_conv = self_view.q_conv_state.at[slot].set(other_view.q_conv_state[0])
                new_k_conv = self_view.k_conv_state.at[slot].set(other_view.k_conv_state[0])
                new_v_conv = self_view.v_conv_state.at[slot].set(other_view.v_conv_state[0])
                new_recurrent = self_view.recurrent_state.at[slot].set(other_view.recurrent_state[0])
                new_positions = self_view.positions.at[slot].set(other_view.positions[0])

                new_view = KDACacheView(
                    q_conv_state=new_q_conv,
                    k_conv_state=new_k_conv,
                    v_conv_state=new_v_conv,
                    recurrent_state=new_recurrent,
                    positions=new_positions,
                    metadata=self_view.metadata,
                    layer_index=self_view.layer_index,
                )
                new_views.append(new_view)

        return KDACache(views=new_views)

    __str__ = __repr__


class KDAMetadata(BaseRunTimeMetadata):
    """Runtime metadata for KDA cache operations."""

    ...
