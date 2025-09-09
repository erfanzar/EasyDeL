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

import chex as cx
from eformer import escale as es
from eformer.escale import PartitionAxis, with_sharding_constraint
from eformer.jaximus import ImplicitArray
from eformer.pytree import auto_pytree
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from .._abstracts import BaseCache, BaseCacheMetadata, BaseCacheView, BaseRunTimeMetadata


@auto_pytree
class MambaCacheMetaData(BaseCacheMetadata):
    """Metadata for Mamba cache configuration."""

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
    conv_states: cx.Array | ImplicitArray
    ssm_states: cx.Array | ImplicitArray
    positions: cx.Array
    metadata: MambaCacheMetaData
    layer_index: int | None = None

    @classmethod
    def init(
        cls,
        metadata: MambaCacheMetaData,
        partition_specs: PartitionSpec,
        dtype: jnp.dtype,
        layer_index: int | None = None,
    ):
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

    def concatenate_to_cache(self, *args, **kwargs):
        raise NotImplementedError()

    def update_conv_state(
        self,
        new_conv_state: cx.Array,
        cache_position: cx.Array,
    ) -> "MambaCacheView":
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
        new_ssm_state: cx.Array,
    ) -> "MambaCacheView":
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

    def reset(self) -> "MambaCacheView":
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

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(conv_states={self.conv_states.shape}, ssm_states={self.ssm_states.shape},"
            f" layer_index={self.layer_index})"
        )

    __str__ = __repr__


@auto_pytree
class MambaCache(BaseCache):
    views: list[MambaCacheView | None]

    @classmethod
    def init_cache(
        cls,
        metadata: MambaCacheMetaData,
        dtype: jnp.dtype | None = None,
        partition_specs: PartitionSpec | None = None,
    ):
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
        new_conv_state: cx.Array,
        cache_position: cx.Array,
    ) -> "MambaCache":
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
        new_ssm_state: cx.Array,
    ) -> "MambaCache":
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

    def reset(self) -> "MambaCache":
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
    def init_empty(cls, num_hidden_layers):
        return cls(views=[None for _ in range(num_hidden_layers)])

    def __repr__(self):
        return f"{self.__class__.__name__}(\n  " + "\n  ".join(str(view) for view in self.views) + "\n)"

    __str__ = __repr__


class MambaMetadata(BaseRunTimeMetadata):
    """Runtime metadata for Mamba cache operations.

    Placeholder for future Mamba-specific runtime metadata.
    May include sequence positions, segment boundaries, etc.
    """

    ...
