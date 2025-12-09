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

"""Mamba2 enhanced state-space model cache implementation.

This module provides caching for Mamba2 models, which extend the original
Mamba architecture with additional features including:
- Multi-head state space models
- Group normalization capabilities
- Enhanced convolutional processing
- Improved sequence modeling

Mamba2 introduces structured state spaces with head-based organization,
similar to multi-head attention but for state-space models.

Key Components:
    - Mamba2CacheMetaData: Enhanced configuration with head/group support
    - Mamba2CacheView: Per-layer state storage with sequence tracking
    - Mamba2Cache: Multi-layer orchestration with sequence updates
    - Mamba2Metadata: Runtime metadata (placeholder)

Features:
    - Head-based SSM organization
    - Group normalization support
    - Sequence position tracking
    - Extended convolutional states

Example:
    >>> metadata = Mamba2CacheMetaData.create(
    ...     partition_axis=partition_axis,
    ...     num_hidden_layers=32,
    ...     batch_size=2,
    ...     intermediate_size=2816,
    ...     num_heads=16,
    ...     head_dim=64,
    ...     state_size=128,
    ...     conv_kernel_size=4,
    ...     n_groups=8
    ... )
    >>> cache = Mamba2Cache.init_cache(
    ...     num_hidden_layers=32,
    ...     metadata=metadata,
    ...     dtype=jnp.float32
    ... )
"""

from __future__ import annotations

from eformer.escale import PartitionAxis, with_sharding_constraint
from eformer.jaximus import ImplicitArray
from eformer.pytree import auto_pytree, field
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array, Float, Int

from .._abstracts import BaseCache, BaseCacheMetadata, BaseCacheView, BaseRunTimeMetadata


@auto_pytree
class Mamba2CacheMetaData(BaseCacheMetadata):
    """Metadata configuration for Mamba2 state-space cache.

    Extends the original Mamba cache with support for multi-head
    state-space models and group normalization. The head-based
    organization allows for more expressive state representations.

    Attributes:
        partition_axis (PartitionAxis): Tensor partitioning configuration.
        num_hidden_layers (int): Number of Mamba2 layers in the model.
        batch_size (int): Number of sequences in batch.
        intermediate_size (int): Hidden dimension of MLP layers.
        num_heads (int): Number of SSM heads (similar to attention heads).
        head_dim (int): Dimension per SSM head.
        state_size (int): Size of the state-space representation.
        conv_kernel_size (int): Size of convolutional kernel.
        n_groups (int): Number of groups for group operations.
    """

    partition_axis: PartitionAxis = field(pytree_node=False)
    num_hidden_layers: int = field(pytree_node=False)
    batch_size: int = field(pytree_node=False)
    intermediate_size: int = field(pytree_node=False)
    num_heads: int = field(pytree_node=False)
    head_dim: int = field(pytree_node=False)
    state_size: int = field(pytree_node=False)
    conv_kernel_size: int = field(pytree_node=False)
    n_groups: int = field(pytree_node=False)

    @classmethod
    def create(
        cls,
        partition_axis: PartitionAxis,
        num_hidden_layers: int,
        batch_size: int,
        intermediate_size: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
        conv_kernel_size: int,
        n_groups: int,
    ) -> "Mamba2CacheMetaData":
        """Create and validate Mamba2 cache metadata.

        Factory method that validates all parameters before creating
        the metadata instance. Ensures all dimensions are positive
        and compatible.

        Args:
            partition_axis (PartitionAxis): Sharding configuration.
            num_hidden_layers (int): Number of model layers.
            batch_size (int): Batch size for cache allocation.
            intermediate_size (int): MLP hidden dimension.
            num_heads (int): Number of SSM heads.
            head_dim (int): Dimension per head.
            state_size (int): State-space size per head.
            conv_kernel_size (int): Convolution kernel size.
            n_groups (int): Number of normalization groups.

        Returns:
            Mamba2CacheMetaData: Validated metadata instance.

        Raises:
            ValueError: If any parameter is non-positive.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if state_size <= 0:
            raise ValueError("state_size must be positive")
        if conv_kernel_size <= 0:
            raise ValueError("conv_kernel_size must be positive")
        if n_groups <= 0:
            raise ValueError("n_groups must be positive")

        return cls(
            num_hidden_layers=num_hidden_layers,
            partition_axis=partition_axis,
            batch_size=batch_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            head_dim=head_dim,
            state_size=state_size,
            conv_kernel_size=conv_kernel_size,
            n_groups=n_groups,
        )


@auto_pytree
class Mamba2CacheView(BaseCacheView):
    """Single-layer cache view for Mamba2 state-space model.

    Manages both convolutional and SSM states for one Mamba2 layer,
    with additional tracking for sequence positions and offsets.
    The multi-head organization allows for richer state representations.

    Attributes:
        conv_states (cx.Array | ImplicitArray): Convolutional states buffer.
            Shape: [batch, intermediate_size + 2*n_groups*state_size, kernel_size]
        ssm_states (cx.Array | ImplicitArray): State-space model states.
            Shape: [batch, num_heads, head_dim, state_size]
        positions (cx.Array): Current position per sequence.
            Shape: [batch_size]
        seqlen_offset (int): Global sequence offset for continuation.
        metadata (Mamba2CacheMetaData): Static configuration.
        layer_index (int | None): Layer index in model.
    """

    conv_states: Float[Array, "batch extended_size conv_kernel_size"] | ImplicitArray
    ssm_states: Float[Array, "batch num_heads head_dim state_size"] | ImplicitArray
    positions: Int[Array, "batch"]  # noqa: F821
    seqlen_offset: int = field(pytree_node=False)
    metadata: Mamba2CacheMetaData
    layer_index: int | None = field(pytree_node=False, default=None)

    @classmethod
    def init(
        cls,
        metadata: Mamba2CacheMetaData,
        partition_specs: PartitionSpec,
        dtype: jnp.dtype,
        layer_index: int | None = None,
    ) -> Mamba2CacheView:
        """Initialize a Mamba2 cache view with zero states.

        Creates cache tensors for the extended Mamba2 architecture,
        including multi-head SSM states and expanded convolutional
        buffers that incorporate group normalization dimensions.

        Args:
            metadata (Mamba2CacheMetaData): Configuration for cache dimensions.
            partition_specs (PartitionSpec): Sharding specification for
                distributed execution.
            dtype (jnp.dtype): Data type for state tensors.
            layer_index (int | None): Optional layer index in model.

        Returns:
            Mamba2CacheView: Initialized cache view with allocated tensors.

        Note:
            Conv state size is extended by 2*n_groups*state_size to
            accommodate group normalization features.
        """
        return cls(
            conv_states=with_sharding_constraint(
                arr=jnp.zeros(
                    shape=(
                        metadata.batch_size,
                        metadata.intermediate_size + 2 * metadata.n_groups * metadata.state_size,
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
                        metadata.num_heads,
                        metadata.head_dim,
                        metadata.state_size,
                    ),
                    dtype=dtype,
                ),
                sharding=partition_specs,
            ),
            positions=jnp.zeros((metadata.batch_size,), "i4"),
            metadata=metadata,
            layer_index=layer_index,
            seqlen_offset=0,
        )

    def concatenate_to_cache(self, *args, **kwargs) -> tuple:
        """Not implemented for Mamba2 cache.

        Mamba2 uses separate update methods for conv and SSM states.

        Raises:
            NotImplementedError: Always raised.

        Note:
            Use `update_conv_state()` and `update_ssm_state()` instead.
        """
        raise NotImplementedError()

    def update_conv_state(
        self,
        new_conv_state: Float[Array, "batch extended_size"],
        cache_position: Int[Array, "..."],
    ) -> Mamba2CacheView:
        """Update convolutional state with new values.

        Maintains a rolling buffer of convolutional states, with
        support for the extended state size used in Mamba2.

        Args:
            new_conv_state (cx.Array): New conv state to insert.
                Shape: [batch, intermediate_size + 2*n_groups*state_size]
            cache_position (cx.Array): Position index for insertion.

        Returns:
            Mamba2CacheView: Updated view with new conv state.
        """
        cache_position = jnp.clip(cache_position, 0, self.metadata.conv_kernel_size - 1)
        conv_state = jnp.roll(self.conv_states, shift=-1, axis=-1)
        updated_conv_states = conv_state.at[:, :, cache_position].set(new_conv_state)
        self.conv_states = updated_conv_states
        return self

    def update_ssm_state(
        self,
        new_ssm_state: Float[Array, "batch num_heads head_dim state_size"],
    ) -> Mamba2CacheView:
        """Update SSM state with head-structured representation.

        Replaces the multi-head SSM state with new values,
        maintaining the head-based organization.

        Args:
            new_ssm_state (cx.Array): New SSM state.
                Shape: [batch, num_heads, head_dim, state_size]

        Returns:
            Mamba2CacheView: Updated view with new SSM state.
        """
        self.ssm_states = new_ssm_state
        return self

    def reset(self) -> Mamba2CacheView:
        """Reset all cache states to zeros.

        Clears both convolutional and SSM states while preserving
        structure and shape. Position tracking is maintained.

        Returns:
            Mamba2CacheView: Reset view with zeroed states.
        """
        self.conv_states = jnp.zeros_like(self.conv_states)
        self.ssm_states = jnp.zeros_like(self.ssm_states)
        return self


@auto_pytree
class Mamba2Cache(BaseCache):
    """Multi-layer Mamba2 cache container.

    Orchestrates Mamba2 cache views across all model layers,
    with additional support for sequence position tracking
    and batch sequence updates.

    Attributes:
        views (list[Mamba2CacheView | None]): Per-layer cache views.
    """

    views: list[Mamba2CacheView | None]

    @classmethod
    def init_cache(
        cls,
        num_hidden_layers: int,
        metadata: Mamba2CacheMetaData,
        dtype: jnp.dtype | None = None,
        partition_specs: PartitionSpec | None = None,
    ) -> Mamba2Cache:
        """Initialize a complete Mamba2 cache for all layers.

        Creates a fully initialized cache with allocated storage for
        the specified number of layers. Each layer gets independent
        multi-head SSM states and convolutional buffers.

        Args:
            num_hidden_layers (int): Number of Mamba2 layers to initialize.
            metadata (Mamba2CacheMetaData): Configuration for cache dimensions.
            dtype (jnp.dtype | None): Data type for tensors. Defaults to bfloat16.
            partition_specs (PartitionSpec | None): Sharding specification.
                If None, creates default spec with batch, head, and sequence axes.

        Returns:
            Mamba2Cache: Fully initialized multi-layer cache.

        Example:
            >>> cache = Mamba2Cache.init_cache(
            ...     num_hidden_layers=32,
            ...     metadata=metadata,
            ...     dtype=jnp.float32
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
                Mamba2CacheView.init(
                    metadata=metadata,
                    partition_specs=partition_specs,
                    dtype=dtype,
                    layer_index=layer_index,
                )
                for layer_index in range(num_hidden_layers)
            ]
        )

    def update_conv_state(
        self,
        layer_idx: int,
        new_conv_state: Float[Array, "batch extended_size"],
        cache_position: Int[Array, "..."],
    ) -> Mamba2Cache:
        """
        Update the convolutional state for a specific layer.

        Arguments:
            layer_idx: Index of the layer to update
            new_conv_state: New state to be inserted
            cache_position: Position in the cache to update

        Returns:
            Updated MambaCache
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
        new_ssm_state: Float[Array, "batch num_heads head_dim state_size"],
    ) -> Mamba2Cache:
        """
        Update the SSM state for a specific layer.

        Arguments:
            layer_idx: Index of the layer to update
            new_ssm_state: New SSM state to replace the current one

        Returns:
            Updated MambaCache
        """
        if self.views[layer_idx] is None:
            raise ValueError(f"Cache view for layer {layer_idx} is None")

        updated_view = self.views[layer_idx].update_ssm_state(
            new_ssm_state=new_ssm_state,
        )

        new_views = list(self.views)
        new_views[layer_idx] = updated_view
        return self.replace(views=new_views)

    def reset(self) -> Mamba2Cache:
        """
        Reset all cache views to their initial state.

        Returns:
            Reset MambaCache
        """
        new_views = [view.reset() if view is not None else None for view in self.views]
        return self.replace(views=new_views)

    @classmethod
    def init_empty(cls, num_hidden_layers: int) -> Mamba2Cache:
        """Initialize an empty Mamba2 cache structure.

        Creates a cache with None placeholders for gradual initialization.

        Args:
            num_hidden_layers (int): Number of layer placeholders.

        Returns:
            Mamba2Cache: Cache with uninitialized views.

        Example:
            >>> cache = Mamba2Cache.init_empty(32)
            >>> # Initialize layers individually later
        """
        return cls(views=[None for _ in range(num_hidden_layers)])

    def update_seq(self, num: int) -> None:
        """Update sequence positions across all layers.

        Increments position tracking for sequence continuation,
        useful when processing long sequences in chunks.

        Args:
            num (int): Number of positions to advance.

        Note:
            Updates both positions and seqlen_offset for each layer.
        """
        for view in self.views:
            if view is not None:
                view.positions += num
                view.seqlen_offset += num

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n  " + "\n  ".join(str(view) for view in self.views) + "\n)"

    __str__ = __repr__


class Mamba2Metadata(BaseRunTimeMetadata):
    """Runtime metadata for Mamba2 cache operations.

    Placeholder for future Mamba2-specific runtime state.
    May include head masks, group indices, etc.
    """
