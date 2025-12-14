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

"""Unified recurrent cache implementation for state-space and linear attention models.

This module provides a unified RecurrentCache that can handle:
- Mamba (single-head SSM): recurrent_state shape [batch, intermediate_size, state_size]
- Mamba2 (multi-head SSM): recurrent_state shape [batch, num_heads, head_dim, state_size]
- GatedDeltaNet: recurrent_state shape [batch, num_heads, head_dim, d_state]
- Other linear attention variants with arbitrary recurrent state shapes

The key insight is that all these models share the same pattern:
1. A convolutional state buffer (conv_state) for local context
2. A recurrent state (ssm_state/recurrent_state) for long-range dependencies

By parameterizing the shapes flexibly, we can support all variants with one implementation.
"""

from __future__ import annotations

import typing as tp

from eformer import escale as es
from eformer.escale import with_sharding_constraint
from eformer.jaximus import ImplicitArray
from eformer.pytree import auto_pytree, field
from jax import lax
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array, Float, Int

from .._abstracts import BaseCache, BaseCacheConfig, BaseCacheView, BaseRunTimeMetadata


@auto_pytree
class RecurrentCacheConfig(BaseCacheConfig):
    """Metadata for unified recurrent cache configuration.

    Stores static configuration for recurrent model caching, supporting both
    single-head (Mamba) and multi-head (Mamba2) state-space organizations,
    as well as linear attention variants like GatedDeltaNet.

    The key flexibility is in `recurrent_state_shape` which can be:
    - (intermediate_size, state_size) for Mamba
    - (num_heads, head_dim, state_size) for Mamba2
    - (num_heads, head_dim, d_state) for GatedDeltaNet
    - Any other shape needed by specific architectures

    Attributes:
        num_hidden_layers: Number of layers in the model.
        partition_axis: Configuration for tensor partitioning.
        batch_size: Number of sequences in batch.
        conv_dim: Dimension of the convolutional input (d_inner for Mamba).
        conv_kernel_size: Size of convolutional kernel (d_conv).
        recurrent_state_shape: Shape of recurrent state per batch element.
            Does NOT include batch dimension.
        seqlen_offset: Optional initial sequence offset for continuation.
    """

    num_hidden_layers: int = field(pytree_node=False)
    partition_axis: es.PartitionAxis = field(pytree_node=False)
    batch_size: int = field(pytree_node=False)
    conv_dim: int = field(pytree_node=False)
    conv_kernel_size: int = field(pytree_node=False)
    recurrent_state_shape: tuple[int, ...] = field(pytree_node=False)
    seqlen_offset: int = field(pytree_node=False, default=0)

    @classmethod
    def create(
        cls,
        num_hidden_layers: int,
        partition_axis: es.PartitionAxis,
        batch_size: int,
        conv_dim: int,
        conv_kernel_size: int,
        recurrent_state_shape: tuple[int, ...],
        seqlen_offset: int = 0,
    ) -> RecurrentCacheConfig:
        """Create a RecurrentCacheConfig instance with validation.

        Args:
            num_hidden_layers: Number of model layers.
            partition_axis: Partition axis configuration.
            batch_size: Size of the batch.
            conv_dim: Dimension for convolutional state (intermediate_size or d_inner).
            conv_kernel_size: Size of the convolution kernel.
            recurrent_state_shape: Shape of recurrent state excluding batch dim.
                Examples:
                - Mamba: (intermediate_size, state_size)
                - Mamba2: (num_heads, head_dim, state_size)
                - GatedDeltaNet: (num_heads, head_dim, d_state)
            seqlen_offset: Initial sequence offset for continuation.

        Returns:
            RecurrentCacheConfig instance.

        Raises:
            ValueError: If required parameters are invalid.
        """
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if conv_dim <= 0:
            raise ValueError("conv_dim must be positive")
        if conv_kernel_size <= 0:
            raise ValueError("conv_kernel_size must be positive")
        if not recurrent_state_shape or any(d <= 0 for d in recurrent_state_shape):
            raise ValueError("recurrent_state_shape must have positive dimensions")

        return cls(
            num_hidden_layers=num_hidden_layers,
            partition_axis=partition_axis,
            batch_size=batch_size,
            conv_dim=conv_dim,
            conv_kernel_size=conv_kernel_size,
            recurrent_state_shape=recurrent_state_shape,
            seqlen_offset=seqlen_offset,
        )

    @classmethod
    def create_for_mamba(
        cls,
        num_hidden_layers: int,
        partition_axis: es.PartitionAxis,
        batch_size: int,
        intermediate_size: int,
        ssm_state_size: int,
        conv_kernel_size: int,
    ) -> RecurrentCacheConfig:
        """Create metadata for Mamba (single-head SSM) models.

        Convenience factory for Mamba-style models with shape:
        - conv_state: [batch, intermediate_size, conv_kernel_size]
        - ssm_state: [batch, intermediate_size, ssm_state_size]

        Args:
            num_hidden_layers: Number of Mamba layers.
            partition_axis: Partition axis configuration.
            batch_size: Batch size.
            intermediate_size: Mamba d_inner dimension.
            ssm_state_size: Size of SSM state.
            conv_kernel_size: Size of conv kernel.

        Returns:
            RecurrentCacheConfig configured for Mamba.
        """
        return cls.create(
            num_hidden_layers=num_hidden_layers,
            partition_axis=partition_axis,
            batch_size=batch_size,
            conv_dim=intermediate_size,
            conv_kernel_size=conv_kernel_size,
            recurrent_state_shape=(intermediate_size, ssm_state_size),
        )

    @classmethod
    def create_for_mamba2(
        cls,
        num_hidden_layers: int,
        partition_axis: es.PartitionAxis,
        batch_size: int,
        intermediate_size: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
        conv_kernel_size: int,
        n_groups: int = 1,
    ) -> RecurrentCacheConfig:
        """Create metadata for Mamba2 (multi-head SSM) models.

        Convenience factory for Mamba2-style models with shape:
        - conv_state: [batch, intermediate_size + 2*n_groups*state_size, conv_kernel_size]
        - ssm_state: [batch, num_heads, head_dim, state_size]

        Args:
            num_hidden_layers: Number of Mamba2 layers.
            partition_axis: Partition axis configuration.
            batch_size: Batch size.
            intermediate_size: MLP hidden dimension.
            num_heads: Number of SSM heads.
            head_dim: Dimension per head.
            state_size: Size of SSM state.
            conv_kernel_size: Size of conv kernel.
            n_groups: Number of groups for normalization.

        Returns:
            RecurrentCacheConfig configured for Mamba2.
        """
        # Mamba2 uses extended conv_dim
        conv_dim = intermediate_size + 2 * n_groups * state_size
        return cls.create(
            num_hidden_layers=num_hidden_layers,
            partition_axis=partition_axis,
            batch_size=batch_size,
            conv_dim=conv_dim,
            conv_kernel_size=conv_kernel_size,
            recurrent_state_shape=(num_heads, head_dim, state_size),
        )


@auto_pytree
class RecurrentCacheView(BaseCacheView):
    """Single-layer cache view for recurrent state management.

    Manages both convolutional and recurrent states for one layer.
    The convolutional states provide local context through a sliding
    window, while recurrent states encode the full sequence history.

    This unified view supports:
    - Mamba: 2D recurrent state [intermediate_size, state_size]
    - Mamba2: 3D recurrent state [num_heads, head_dim, state_size]
    - GatedDeltaNet: 3D recurrent state [num_heads, head_dim, d_state]

    Attributes:
        conv_state: Rolling buffer of convolutional states.
            Shape: [batch_size, conv_dim, conv_kernel_size]
        recurrent_state: Recurrent hidden state.
            Shape: [batch_size, *recurrent_state_shape]
        positions: Current position index per batch element.
        seqlen_offset: Sequence length offset for continuation.
        metadata: Static configuration metadata.
        layer_index: Index of this layer in the model.
    """

    conv_state: Float[Array, "batch conv_dim conv_kernel_size"] | ImplicitArray | None
    recurrent_state: Float[Array, "batch ..."] | ImplicitArray | None
    positions: Int[Array, "batch"]  # noqa:F821
    # NOTE: this must be a JAX value (not static pytree metadata) so it can
    # change inside `lax.while_loop` during generation without breaking the
    # carry pytree structure.
    seqlen_offset: Int[Array, ""] = field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))
    metadata: RecurrentCacheConfig = field(default=None)
    layer_index: int | None = field(pytree_node=False, default=None)

    @classmethod
    def init(
        cls,
        config: RecurrentCacheConfig,
        layer_index: int | None = None,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        partition_specs: PartitionSpec | None = None,
    ) -> "RecurrentCacheView":
        """Initialize a RecurrentCacheView from a cache config.

        Creates and allocates cache tensors for both convolutional and
        recurrent states for linear attention layers.

        Args:
            config: RecurrentCacheConfig with cache dimensions.
            layer_index: Index of this layer in the model.
            dtype: Data type for cache tensors.
            partition_specs: Sharding specification for distributed execution.

        Returns:
            RecurrentCacheView: Initialized cache view.
        """
        paxis = config.partition_axis
        if partition_specs is None:
            partition_specs = PartitionSpec(
                paxis.batch_axis,
                paxis.head_axis,
                paxis.sequence_axis,
            )

        # Conv state shape: [batch, conv_dim, conv_kernel_size]
        conv_state_shape = (
            config.batch_size,
            config.conv_dim,
            config.conv_kernel_size,
        )

        # Recurrent state shape: [batch, *recurrent_state_shape]
        recurrent_state_shape = (config.batch_size, *config.recurrent_state_shape)

        return cls(
            conv_state=with_sharding_constraint(
                arr=jnp.zeros(conv_state_shape, dtype=dtype),
                sharding=partition_specs,
            ),
            recurrent_state=with_sharding_constraint(
                arr=jnp.zeros(recurrent_state_shape, dtype=dtype),
                sharding=partition_specs,
            ),
            positions=jnp.zeros((config.batch_size,), dtype=jnp.int32),
            seqlen_offset=jnp.array(config.seqlen_offset, dtype=jnp.int32),
            metadata=config,
            layer_index=layer_index,
        )

    def concatenate_to_cache(
        self,
        conv_state: Float[Array, "batch conv_dim"] | None = None,
        recurrent_state: Float[Array, "batch ..."] | None = None,
        cache_position: Int[Array, "..."] | None = None,
    ) -> tuple[
        Float[Array, "batch conv_dim conv_kernel_size"] | None,
        Float[Array, "batch ..."] | None,
        RecurrentCacheView,
    ]:
        """Update cache state with new conv and/or recurrent states.

        Unified update method that handles both conv_state (rolling buffer)
        and recurrent_state (direct replacement). Updates are only applied
        for non-None inputs.

        Args:
            conv_state: New convolutional state to insert.
                Shape: [batch, conv_dim]
                If provided, will be inserted into rolling buffer.
            recurrent_state: New recurrent state.
                Shape: [batch, *recurrent_state_shape]
                If provided, will replace current recurrent state.
            cache_position: Position index for conv state insertion.
                Clamped to valid range [0, conv_kernel_size-1].

        Returns:
            Tuple of (updated_conv_state, updated_recurrent_state, updated_view).
        """
        new_conv_state = self.conv_state
        new_recurrent_state = self.recurrent_state

        # Update conv_state with rolling buffer
        if conv_state is not None and self.conv_state is not None:
            if cache_position is not None:
                cache_position = jnp.clip(cache_position, 0, self.metadata.conv_kernel_size - 1)
            # Roll and insert new state
            rolled = jnp.roll(self.conv_state, shift=-1, axis=-1)
            if cache_position is not None:
                new_conv_state = rolled.at[:, :, cache_position].set(conv_state)
            else:
                # Default: insert at last position
                new_conv_state = rolled.at[:, :, -1].set(conv_state)

        # Update recurrent_state (direct replacement)
        if recurrent_state is not None:
            new_recurrent_state = recurrent_state

        new_view = RecurrentCacheView(
            conv_state=new_conv_state,
            recurrent_state=new_recurrent_state,
            positions=self.positions,
            seqlen_offset=self.seqlen_offset,
            metadata=self.metadata,
            layer_index=self.layer_index,
        )

        return new_conv_state, new_recurrent_state, new_view

    def update_conv_state(
        self,
        new_conv_state: Float[Array, "batch conv_dim"],
        cache_position: Int[Array, "..."] | None = None,
    ) -> RecurrentCacheView:
        """Update the convolutional state with new values.

        Implements a rolling buffer for convolutional states.

        Args:
            new_conv_state: New convolutional state to insert.
                Shape: [batch, conv_dim]
            cache_position: Position index for insertion.

        Returns:
            RecurrentCacheView: Updated view with new conv state.
        """
        _, _, new_view = self.concatenate_to_cache(
            conv_state=new_conv_state,
            cache_position=cache_position,
        )
        return new_view

    def update_recurrent_state(
        self,
        new_recurrent_state: Float[Array, "batch ..."],
    ) -> RecurrentCacheView:
        """Update the recurrent (SSM) state.

        Replaces the entire recurrent state with new values.

        Args:
            new_recurrent_state: New recurrent state tensor.
                Shape: [batch, *recurrent_state_shape]

        Returns:
            RecurrentCacheView: Updated view with new recurrent state.
        """
        _, _, new_view = self.concatenate_to_cache(recurrent_state=new_recurrent_state)
        return new_view

    # Aliases for backward compatibility with Mamba/Mamba2
    def update_ssm_state(
        self,
        new_ssm_state: Float[Array, "batch ..."],
    ) -> RecurrentCacheView:
        """Alias for update_recurrent_state (backward compatibility)."""
        return self.update_recurrent_state(new_ssm_state)

    @property
    def ssm_states(self) -> Float[Array, "batch ..."] | ImplicitArray | None:
        """Alias for recurrent_state (backward compatibility with MambaCache)."""
        return self.recurrent_state

    @property
    def conv_states(self) -> Float[Array, "batch conv_dim conv_kernel_size"] | ImplicitArray | None:
        """Alias for conv_state (backward compatibility with MambaCache)."""
        return self.conv_state

    def reset(self) -> RecurrentCacheView:
        """Reset all cache states to zeros.

        Returns:
            RecurrentCacheView: Reset view with zeroed states.
        """
        return RecurrentCacheView(
            conv_state=jnp.zeros_like(self.conv_state) if self.conv_state is not None else None,
            recurrent_state=jnp.zeros_like(self.recurrent_state) if self.recurrent_state is not None else None,
            positions=jnp.zeros_like(self.positions),
            seqlen_offset=jnp.array(0, dtype=jnp.int32),
            metadata=self.metadata,
            layer_index=self.layer_index,
        )

    def __repr__(self) -> str:
        conv_shape = self.conv_state.shape if self.conv_state is not None else None
        rec_shape = self.recurrent_state.shape if self.recurrent_state is not None else None
        return (
            f"{self.__class__.__name__}("
            f"conv_state={conv_shape}, "
            f"recurrent_state={rec_shape}, "
            f"layer_index={self.layer_index})"
        )

    __str__ = __repr__


@auto_pytree
class RecurrentCache(BaseCache):
    """Multi-layer cache container for recurrent models.

    Orchestrates cache views across all layers, providing a unified
    interface for state management during inference.

    Attributes:
        views: Ordered list of cache views, one per model layer.
    """

    views: list[RecurrentCacheView | None]

    @classmethod
    def init_cache(
        cls,
        config: RecurrentCacheConfig,
        dtype: jnp.dtype | None = None,
        partition_specs: PartitionSpec | None = None,
    ) -> RecurrentCache:
        """Initialize a complete recurrent cache with views for all layers.

        Args:
            config: Configuration defining cache dimensions and layers.
            dtype: Data type for cache tensors. Defaults to bfloat16.
            partition_specs: Sharding specification for distributed execution.

        Returns:
            RecurrentCache: Fully initialized cache ready for inference.
        """
        if dtype is None:
            dtype = jnp.bfloat16

        return cls(
            views=[
                RecurrentCacheView.init(
                    config=config,
                    layer_index=layer_index,
                    dtype=dtype,
                    partition_specs=partition_specs,
                )
                for layer_index in range(config.num_hidden_layers)
            ]
        )

    def update_conv_state(
        self,
        layer_idx: int,
        new_conv_state: Float[Array, "batch conv_dim"],
        cache_position: Int[Array, "..."] | None = None,
    ) -> RecurrentCache:
        """Update convolutional state for a specific layer.

        Args:
            layer_idx: Index of the layer to update.
            new_conv_state: New convolutional state.
            cache_position: Position for insertion.

        Returns:
            RecurrentCache: New cache instance with updated layer.
        """
        if self.views[layer_idx] is None:
            raise ValueError(f"Cache view for layer {layer_idx} is None")

        updated_view = self.views[layer_idx].update_conv_state(
            new_conv_state=new_conv_state,
            cache_position=cache_position,
        )

        new_views = list(self.views)
        new_views[layer_idx] = updated_view
        return RecurrentCache(views=new_views)

    def update_recurrent_state(
        self,
        layer_idx: int,
        new_recurrent_state: Float[Array, "batch ..."],
    ) -> RecurrentCache:
        """Update recurrent state for a specific layer.

        Args:
            layer_idx: Index of the layer to update.
            new_recurrent_state: New recurrent state tensor.

        Returns:
            RecurrentCache: New cache instance with updated layer.
        """
        if self.views[layer_idx] is None:
            raise ValueError(f"Cache view for layer {layer_idx} is None")

        updated_view = self.views[layer_idx].update_recurrent_state(
            new_recurrent_state=new_recurrent_state,
        )

        new_views = list(self.views)
        new_views[layer_idx] = updated_view
        return RecurrentCache(views=new_views)

    # Alias for backward compatibility
    def update_ssm_state(
        self,
        layer_idx: int,
        new_ssm_state: Float[Array, "batch ..."],
    ) -> RecurrentCache:
        """Alias for update_recurrent_state (backward compatibility)."""
        return self.update_recurrent_state(layer_idx, new_ssm_state)

    def reset(self) -> RecurrentCache:
        """Reset all cache layers to zero states.

        Returns:
            RecurrentCache: New cache instance with all states zeroed.
        """
        new_views = [view.reset() if view is not None else None for view in self.views]
        return RecurrentCache(views=new_views)

    @classmethod
    def init_empty(cls, num_hidden_layers: int) -> RecurrentCache:
        """Initialize an empty cache without allocated storage.

        Args:
            num_hidden_layers: Number of layers to create placeholders for.

        Returns:
            RecurrentCache: Cache instance with uninitialized (None) views.
        """
        return cls(views=[None for _ in range(num_hidden_layers)])

    def update_seq(self, num: int) -> None:
        """Update sequence positions across all layers.

        Args:
            num: Number of positions to advance.
        """
        for view in self.views:
            if view is not None:
                view.positions = view.positions + num
                view.seqlen_offset = view.seqlen_offset + num

    def to_pure(self) -> tuple[list[dict[str, tp.Any]], RecurrentCacheConfig | None]:
        """Convert cache to pure Python data structure for serialization.

        Extracts raw tensors and metadata for checkpointing or transfer.

        Returns:
            tuple: Pair of (cache_data, metadata) where:
                - cache_data: List of dicts with conv_state, recurrent_state, positions
                - metadata: Cache configuration metadata (from first non-None view)
        """
        cache_data = []
        metadata = None

        for view in self.views:
            if view is None:
                cache_data.append(None)
            else:
                cache_data.append(
                    {
                        "conv_state": view.conv_state,
                        "recurrent_state": view.recurrent_state,
                        "positions": view.positions,
                    }
                )
                if metadata is None:
                    metadata = view.metadata

        return cache_data, metadata

    @classmethod
    def from_pure(
        cls,
        cache_data: list[dict[str, tp.Any] | None],
        metadata: RecurrentCacheConfig | None = None,
    ) -> "RecurrentCache":
        """Reconstruct cache from pure Python data structure.

        Restores a cache from serialized tensors and metadata,
        typically after loading from disk or receiving from transfer.

        Args:
            cache_data: List of dicts with conv_state, recurrent_state, positions per layer.
            metadata: Cache configuration metadata.

        Returns:
            RecurrentCache: Reconstructed cache instance.
        """
        views = []

        for idx, data in enumerate(cache_data):
            if data is None:
                views.append(None)
            else:
                views.append(
                    RecurrentCacheView(
                        conv_state=data["conv_state"],
                        recurrent_state=data["recurrent_state"],
                        positions=data.get("positions", jnp.zeros((data["conv_state"].shape[0],), dtype=jnp.int32)),
                        metadata=metadata,
                        layer_index=idx,
                    )
                )

        return cls(views=views)

    def insert(
        self,
        other: "RecurrentCache",
        slot: int,
    ) -> "RecurrentCache":
        """Insert another cache's contents at specified batch slot.

        Copies conv_state and recurrent_state from another cache into
        this cache at the specified batch position.

        Args:
            other: Source cache to copy from.
            slot: Batch slot index to insert into.

        Returns:
            RecurrentCache: Updated cache instance.
        """
        new_views = list(self.views)

        for idx in range(len(self.views)):
            view = self.views[idx]
            oview = other.views[idx]

            if view is None or oview is None:
                continue

            update_dict = {}

            if view.conv_state is not None and oview.conv_state is not None:
                update_dict["conv_state"] = lax.dynamic_update_slice_in_dim(view.conv_state, oview.conv_state, slot, 0)

            if view.recurrent_state is not None and oview.recurrent_state is not None:
                update_dict["recurrent_state"] = lax.dynamic_update_slice_in_dim(
                    view.recurrent_state, oview.recurrent_state, slot, 0
                )

            if view.positions is not None and oview.positions is not None:
                update_dict["positions"] = lax.dynamic_update_slice_in_dim(view.positions, oview.positions, slot, 0)

            new_views[idx] = view.replace(**update_dict)

        return self.replace(views=new_views)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n  " + "\n  ".join(str(view) for view in self.views) + "\n)"

    __str__ = __repr__


class RecurrentMetadata(BaseRunTimeMetadata):
    """Runtime metadata for recurrent cache operations.

    Placeholder for future recurrent-model-specific runtime metadata.
    May include sequence positions, segment boundaries, etc.
    """

    ...


# Convenience aliases for backward compatibility
@auto_pytree
class LinearCache(RecurrentCache): ...


@auto_pytree
class LinearCacheConfig(RecurrentCacheConfig): ...


@auto_pytree
class LinearCacheView(RecurrentCacheView): ...


@auto_pytree
class LinearMetadata(RecurrentMetadata): ...
