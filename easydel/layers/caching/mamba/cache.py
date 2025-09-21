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

"""Mamba state-space model cache implementation for EasyDeL.

This module provides caching for Mamba models, which use state-space
formulations instead of attention mechanisms. Mamba caches maintain
convolutional and SSM (State Space Model) states rather than key-value pairs.

Mamba models process sequences using:
- Convolutional states for local context
- SSM states for long-range dependencies
- Efficient linear-time complexity

Key Components:
    - MambaCacheMetaData: Configuration for Mamba cache dimensions
    - MambaCacheView: Per-layer state storage for conv and SSM
    - MambaCache: Multi-layer Mamba cache orchestration
    - MambaMetadata: Runtime metadata (placeholder)

Features:
    - Separate convolutional and SSM state management
    - Rolling buffer for convolutional states
    - Direct SSM state updates
    - Memory-efficient state representation

Example:
    >>> metadata = MambaCacheMetaData.create(
    ...     num_hidden_layers=24,
    ...     partition_axis=partition_axis,
    ...     batch_size=2,
    ...     intermediate_size=2048,
    ...     ssm_state_size=16,
    ...     conv_kernel_size=4
    ... )
    >>> cache = MambaCache.init_cache(
    ...     metadata=metadata,
    ...     dtype=jnp.float32
    ... )
    >>> # Update conv state for layer 0
    >>> cache = cache.update_conv_state(
    ...     layer_idx=0,
    ...     new_conv_state=conv_state,
    ...     cache_position=position
    ... )
"""

from __future__ import annotations

import chex as cx
from eformer import escale as es
from eformer.escale import PartitionAxis, with_sharding_constraint
from eformer.jaximus import ImplicitArray
from eformer.pytree import auto_pytree
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array, Bool, Float, Int

from .._abstracts import BaseCache, BaseCacheMetadata, BaseCacheView, BaseRunTimeMetadata


@auto_pytree
class MambaCacheMetaData(BaseCacheMetadata):
    """Metadata for Mamba cache configuration.

    Stores static configuration for Mamba model caching, including dimensions
    for state-space model states and convolutional buffers. Mamba models
    use a combination of SSM states for long-range modeling and convolutional
    states for local context.

    Attributes:
        num_hidden_layers (int): Number of Mamba layers in the model.
        partition_axis (PartitionAxis): Configuration for tensor partitioning
            in distributed settings.
        batch_size (int): Number of sequences in batch.
        intermediate_size (int): Dimension of intermediate representations
            in Mamba blocks (typically expansion of model dimension).
        ssm_state_size (int): Dimension of the SSM state vectors.
            Controls model's memory capacity.
        conv_kernel_size (int): Size of convolutional kernel for local mixing.
            Typically 3-7 for short-range dependencies.
    """

    # Required fields
    num_hidden_layers: int
    partition_axis: es.PartitionAxis
    batch_size: int
    intermediate_size: int
    ssm_state_size: int
    conv_kernel_size: int

    @classmethod
    def create(
        cls,
        num_hidden_layers: int,
        partition_axis: es.PartitionAxis,
        batch_size: int,
        intermediate_size: int,
        ssm_state_size: int,
        conv_kernel_size: int,
    ) -> "MambaCacheMetaData":
        """
        Create a MambaCacheMetaData instance with validation.

        Arguments:
                        partition_axis: Partition Axis.
            batch_size: Size of the batch
            intermediate_size: Model's intermediate size
            ssm_state_size: Model's state size
            conv_kernel_size: Model's convolution kernel size

        Returns:
            MambaCacheMetaData instance

        Raises:
            ValueError: If required parameters are invalid
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if ssm_state_size <= 0:
            raise ValueError("ssm_state_size must be positive")
        if conv_kernel_size <= 0:
            raise ValueError("conv_kernel_size must be positive")

        return cls(
            num_hidden_layers=num_hidden_layers,
            partition_axis=partition_axis,
            batch_size=batch_size,
            intermediate_size=intermediate_size,
            ssm_state_size=ssm_state_size,
            conv_kernel_size=conv_kernel_size,
        )


@auto_pytree
class MambaCacheView(BaseCacheView):
    """Single-layer cache view for Mamba state management.

    Manages both convolutional and SSM states for one Mamba layer.
    The convolutional states provide local context through a sliding
    window, while SSM states encode the full sequence history.

    Attributes:
        conv_states (Array | ImplicitArray): Rolling buffer of convolutional states.
            Shape: [batch_size, intermediate_size, conv_kernel_size]
            Stores the last `conv_kernel_size` timesteps for convolution.
        ssm_states (Array | ImplicitArray): State-space model hidden states.
            Shape: [batch_size, intermediate_size, ssm_state_size]
            Encodes full sequence history in compressed form.
        positions (Array): Current position index per batch element.
            Shape: [batch_size]
            Tracks where each sequence is in generation.
        metadata (MambaCacheMetaData): Static configuration metadata.
        layer_index (int | None): Index of this layer in the model.
    """

    conv_states: Float[Array, "batch intermediate_size conv_kernel_size"] | ImplicitArray
    ssm_states: Float[Array, "batch intermediate_size ssm_state_size"] | ImplicitArray
    positions: Int[Array, "batch"]  # noqa: F821
    metadata: MambaCacheMetaData
    layer_index: int | None = None

    @classmethod
    def init(
        cls,
        metadata: MambaCacheMetaData,
        partition_specs: PartitionSpec,
        dtype: jnp.dtype,
        layer_index: int | None = None,
    ) -> MambaCacheView:
        """Initialize a Mamba cache view with zero states.

        Creates and allocates cache tensors for both convolutional and SSM
        states. All states are initialized to zeros, representing a fresh
        start with no prior context.

        Args:
            metadata (MambaCacheMetaData): Configuration for cache dimensions.
            partition_specs (PartitionSpec): Sharding specification for distributed
                execution. Applied to both conv and SSM states.
            dtype (jnp.dtype): Data type for state tensors (e.g., float32, bfloat16).
            layer_index (int | None): Optional index of this layer in the model.

        Returns:
            MambaCacheView: Initialized cache view with allocated zero tensors.

        Example:
            >>> view = MambaCacheView.init(
            ...     metadata=metadata,
            ...     partition_specs=PartitionSpec('dp', None, None),
            ...     dtype=jnp.float32,
            ...     layer_index=0
            ... )
        """
        return cls(
            conv_states=with_sharding_constraint(
                arr=jnp.zeros(
                    shape=(
                        metadata.batch_size,
                        metadata.intermediate_size,
                        metadata.conv_kernel_size,
                    ),
                    dtype=dtype,
                ),
                sharding=partition_specs,
            ),
            ssm_states=with_sharding_constraint(
                arr=jnp.zeros(
                    shape=(
                        metadata.batch_size,
                        metadata.intermediate_size,
                        metadata.ssm_state_size,
                    ),
                    dtype=dtype,
                ),
                sharding=partition_specs,
            ),
            positions=jnp.zeros((metadata.batch_size,), "i4"),
            metadata=metadata,
            layer_index=layer_index,
        )

    def concatenate_to_cache(self, *args, **kwargs) -> tuple:
        """Not implemented for Mamba cache.

        Mamba uses separate update methods for conv and SSM states
        rather than a unified concatenation interface.

        Raises:
            NotImplementedError: Always raised as this method is not
                applicable to Mamba caching strategy.

        Note:
            Use `update_conv_state()` and `update_ssm_state()` instead.
        """
        raise NotImplementedError()

    def update_conv_state(
        self,
        new_conv_state: Float[Array, "batch intermediate_size"],
        cache_position: Int[Array, "..."],
    ) -> MambaCacheView:
        """Update the convolutional state with new values.

        Implements a rolling buffer for convolutional states, where
        new states replace old ones in a circular fashion. This
        maintains a fixed-size window of recent states.

        The update process:
        1. Roll existing states to make room
        2. Insert new state at specified position
        3. Return updated view (functional update)

        Args:
            new_conv_state (cx.Array): New convolutional state to insert.
                Shape: [batch_size, intermediate_size]
            cache_position (cx.Array): Position index for insertion.
                Clamped to valid range [0, conv_kernel_size-1].

        Returns:
            MambaCacheView: Updated view with new conv state.

        Note:
            Position is automatically clamped to prevent out-of-bounds access.
        """
        # Clamp cache position to valid range
        cache_position = jnp.clip(cache_position, 0, self.metadata.conv_kernel_size - 1)

        # Roll the conv states and update with new state
        conv_state = jnp.roll(self.conv_states, shift=-1, axis=-1)
        updated_conv_states = conv_state.at[:, :, cache_position].set(new_conv_state)

        self.conv_states = updated_conv_states

    def update_ssm_state(
        self,
        new_ssm_state: Float[Array, "batch intermediate_size ssm_state_size"],
    ) -> MambaCacheView:
        """Update the SSM (State Space Model) state.

        Replaces the entire SSM state with new values. Unlike conv states
        which use a rolling buffer, SSM states are completely replaced
        as they represent the full model state at each timestep.

        Args:
            new_ssm_state (cx.Array): New SSM state tensor.
                Shape: [batch_size, intermediate_size, ssm_state_size]

        Returns:
            MambaCacheView: Updated view with new SSM state.

        Note:
            SSM states encode the full history up to current position,
            so replacement (not accumulation) is the correct operation.
        """
        self.ssm_states = new_ssm_state

    def reset(self) -> MambaCacheView:
        """Reset all cache states to zeros.

        Clears both convolutional and SSM states, effectively
        resetting the model's memory. Useful for:
        - Starting new sequences
        - Clearing context between batches
        - Debugging and testing

        Returns:
            MambaCacheView: Reset view with zeroed states.
        """
        self.conv_states = jnp.zeros_like(self.conv_states)
        self.ssm_states = jnp.zeros_like(self.ssm_states)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__ + f"(conv_states={self.conv_states.shape}, ssm_states={self.ssm_states.shape},"
            f" layer_index={self.layer_index})"
        )

    __str__ = __repr__


@auto_pytree
class MambaCache(BaseCache):
    """Multi-layer cache container for Mamba models.

    Orchestrates cache views across all Mamba layers, providing a unified
    interface for state management during inference. Each layer maintains
    independent conv and SSM states.

    Attributes:
        views (list[MambaCacheView | None]): Ordered list of cache views,
            one per model layer. None values indicate uninitialized layers.
    """

    views: list[MambaCacheView | None]

    @classmethod
    def init_cache(
        cls,
        metadata: MambaCacheMetaData,
        dtype: jnp.dtype | None = None,
        partition_specs: PartitionSpec | None = None,
    ) -> MambaCache:
        """Initialize a complete Mamba cache with views for all layers.

        Creates a fully initialized cache with allocated storage for all
        layers specified in the metadata. Each layer gets its own view
        with independent state tensors.

        Args:
            metadata (MambaCacheMetaData): Configuration defining cache
                dimensions and number of layers.
            dtype (jnp.dtype | None): Data type for cache tensors.
                Defaults to bfloat16 if not specified.
            partition_specs (PartitionSpec | None): Sharding specification
                for distributed execution. If None, creates default spec
                with batch, head, and sequence axes.

        Returns:
            MambaCache: Fully initialized cache ready for inference.

        Example:
            >>> cache = MambaCache.init_cache(
            ...     metadata=metadata,
            ...     dtype=jnp.float32,
            ...     partition_specs=PartitionSpec('dp', None, None)
            ... )
        """
        paxis = PartitionAxis()
        partition_specs = partition_specs or PartitionSpec(
            paxis.batch_axis,
            paxis.head_axis,
            paxis.sequence_axis,
        )
        if dtype is None:
            dtype = jnp.bfloat16

        return cls(
            views=[
                MambaCacheView.init(
                    metadata=metadata,
                    partition_specs=partition_specs,
                    dtype=dtype,
                    layer_index=layer_index,
                )
                for layer_index in range(metadata.num_hidden_layers)
            ]
        )

    def update_conv_state(
        self,
        layer_idx: int,
        new_conv_state: Float[Array, "batch intermediate_size"],
        cache_position: Int[Array, "..."],
    ) -> MambaCache:
        """Update convolutional state for a specific layer.

        Delegates to the specified layer's view to update its conv state,
        then returns a new cache instance with the updated view.

        Args:
            layer_idx (int): Index of the layer to update.
            new_conv_state (cx.Array): New convolutional state.
                Shape: [batch_size, intermediate_size]
            cache_position (cx.Array): Position for insertion.

        Returns:
            MambaCache: New cache instance with updated layer.

        Raises:
            ValueError: If specified layer view is None.

        Example:
            >>> cache = cache.update_conv_state(
            ...     layer_idx=5,
            ...     new_conv_state=hidden_states,
            ...     cache_position=jnp.array([2])
            ... )
        """
        if self.views[layer_idx] is None:
            raise ValueError(f"Cache view for layer {layer_idx} is None")

        updated_view = self.views[layer_idx].update_conv_state(
            new_conv_state=new_conv_state,
            cache_position=cache_position,
        )

        new_views = list(self.views)
        new_views[layer_idx] = updated_view
        return self.replace(views=new_views)

    def update_ssm_state(
        self,
        layer_idx: int,
        new_ssm_state: Float[Array, "batch intermediate_size ssm_state_size"],
    ) -> MambaCache:
        """Update SSM state for a specific layer.

        Replaces the SSM state for the specified layer with new values,
        returning a new cache instance with the update.

        Args:
            layer_idx (int): Index of the layer to update.
            new_ssm_state (cx.Array): New SSM state tensor.
                Shape: [batch_size, intermediate_size, ssm_state_size]

        Returns:
            MambaCache: New cache instance with updated layer.

        Raises:
            ValueError: If specified layer view is None.

        Example:
            >>> cache = cache.update_ssm_state(
            ...     layer_idx=5,
            ...     new_ssm_state=ssm_output
            ... )
        """
        if self.views[layer_idx] is None:
            raise ValueError(f"Cache view for layer {layer_idx} is None")

        updated_view = self.views[layer_idx].update_ssm_state(
            new_ssm_state=new_ssm_state,
        )

        new_views = list(self.views)
        new_views[layer_idx] = updated_view
        return self.replace(views=new_views)

    def reset(self) -> MambaCache:
        """Reset all cache layers to zero states.

        Clears the entire cache by resetting each layer's conv and SSM
        states to zeros. Useful for sequence boundaries or reinitialization.

        Returns:
            MambaCache: New cache instance with all states zeroed.

        Note:
            Preserves cache structure; only clears state values.
        """
        new_views = [view.reset() if view is not None else None for view in self.views]
        return self.replace(views=new_views)

    @classmethod
    def init_empty(cls, num_hidden_layers: int) -> MambaCache:
        """Initialize an empty Mamba cache without allocated storage.

        Creates a cache structure with None placeholders for each layer.
        Useful for gradual initialization or when cache allocation is
        deferred.

        Args:
            num_hidden_layers (int): Number of layers to create placeholders for.

        Returns:
            MambaCache: Cache instance with uninitialized (None) views.

        Example:
            >>> cache = MambaCache.init_empty(num_hidden_layers=24)
            >>> # Populate individual layers later
            >>> cache.views[0] = MambaCacheView.init(...)
        """
        return cls(views=[None for _ in range(num_hidden_layers)])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n  " + "\n  ".join(str(view) for view in self.views) + "\n)"

    __str__ = __repr__


class MambaMetadata(BaseRunTimeMetadata):
    """Runtime metadata for Mamba cache operations.

    Placeholder for future Mamba-specific runtime metadata.
    May include sequence positions, segment boundaries, etc.
    """

    ...
