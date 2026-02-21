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

"""Hybrid cache implementation for models with mixed attention types.

This module provides caching for hybrid transformer models that combine
different attention mechanisms, such as Qwen3Next which uses:
- Full attention layers: Standard multi-head attention (needs KV cache)
- Linear attention layers: GatedDeltaNet (needs conv state + recurrent state)

The hybrid cache provides a unified interface for both cache types,
automatically routing operations to the appropriate state storage
based on the layer's attention type.

Key Features:
- Per-layer attention type configuration
- KV cache for full attention layers
- Conv state + recurrent state for linear attention layers
- Functional updates for JAX compatibility
- Memory-efficient storage (only allocates what each layer needs)

Architecture:
    HybridCacheConfig
    ├── Configuration for all layers
    ├── Layer type mapping (full_attention vs linear_attention)
    └── Dimensions for both KV and recurrent state

    HybridCacheView
    ├── KV states (for full_attention)
    │   ├── key: [batch, seq, num_kv_heads, head_dim]
    │   └── value: [batch, seq, num_kv_heads, head_dim]
    └── Recurrent states (for linear_attention)
        ├── conv_state: [batch, d_inner, d_conv]
        └── recurrent_state: [batch, num_heads, head_dim, d_state]

Example:
    >>> # Create metadata with layer types
    >>> layer_types = tuple(
    ...     "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
    ...     for i in range(48)
    ... )
    >>> metadata = HybridCacheConfig.create(
    ...     num_hidden_layers=48,
    ...     partition_axis=PartitionAxis(),
    ...     batch_size=2,
    ...     sequence_length=2048,
    ...     num_key_value_heads=8,
    ...     head_dim=128,
    ...     d_inner=2048,
    ...     d_conv=4,
    ...     d_state=64,
    ...     layer_types=layer_types,
    ... )
    >>> cache = HybridCache.init_cache(metadata, dtype=jnp.bfloat16)
"""

from __future__ import annotations

import typing as tp

from eformer import escale as es
from eformer.escale import PartitionAxis, with_sharding_constraint
from eformer.jaximus import ImplicitArray
from eformer.pytree import auto_pytree, field, xTree
from jax import lax
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array, Float, Int

from .._abstracts import BaseCache, BaseCacheConfig, BaseCacheView, BaseRunTimeMetadata

if tp.TYPE_CHECKING:
    from eformer.escale import PartitionManager

    from ..kda.cache import KDACacheView
    from ..recurrent.cache import RecurrentCacheView
    from ..transformer.cache import TransformerCacheView
else:
    # AutoPytree need these informations for optimizations
    KDACacheView, RecurrentCacheView, TransformerCacheView = [xTree] * 3

FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"
KDA_LINEAR_ATTENTION = "kda_linear_attention"
PARALLEL_HYBRID = "parallel_hybrid"
LayerType = tp.Literal["full_attention", "linear_attention", "kda_linear_attention", "parallel_hybrid"]


@auto_pytree
class HybridCacheConfig(BaseCacheConfig):
    """Metadata for hybrid cache configuration.

    Stores static configuration for hybrid attention model caching, supporting
    both standard KV caching (for full attention layers) and recurrent state
    caching (for linear attention layers like GatedDeltaNet).

    The metadata defines dimensions for both cache types, with storage
    allocated per-layer based on the layer_types configuration.

    Attributes:
        num_hidden_layers (int): Number of transformer layers in the model.
        partition_axis (PartitionAxis): Configuration for tensor partitioning.
        batch_size (int): Number of sequences in batch.
        sequence_length (int): Maximum sequence length for KV cache.
        num_key_value_heads (int): Number of KV heads for full attention.
        head_dim (int): Dimension of each attention head.
        d_inner (int): Intermediate dimension for linear attention.
        d_conv (int): Convolution kernel size for linear attention.
        d_state (int): Recurrent state dimension for linear attention.
        layer_types (tuple[str, ...]): Per-layer attention type specification.
            Each element is either "full_attention" or "linear_attention".
    """

    # Required fields
    num_hidden_layers: int = field(pytree_node=False)
    partition_axis: es.PartitionAxis = field(pytree_node=False)
    batch_size: int = field(pytree_node=False)
    sequence_length: int = field(pytree_node=False)

    # KV cache params (for full_attention layers)
    num_key_value_heads: int = field(pytree_node=False)
    head_dim: int = field(pytree_node=False)

    # Recurrent state params (for linear_attention layers)
    d_inner: int = field(pytree_node=False)
    d_conv: int = field(pytree_node=False)
    d_state: int = field(pytree_node=False)
    num_attention_heads: int = field(pytree_node=False)

    # Layer type configuration
    layer_types: tuple[str, ...] = field(pytree_node=False)

    @classmethod
    def create(
        cls,
        num_hidden_layers: int,
        partition_axis: es.PartitionAxis,
        batch_size: int,
        sequence_length: int,
        num_key_value_heads: int,
        head_dim: int,
        d_inner: int,
        d_conv: int,
        d_state: int,
        layer_types: tuple[str, ...] | list[str],
        num_attention_heads: int | None = None,
    ) -> "HybridCacheConfig":
        """Create a HybridCacheConfig instance with validation.

        Args:
            num_hidden_layers: Number of transformer layers.
            partition_axis: Partition axis configuration.
            batch_size: Batch size.
            sequence_length: Maximum sequence length for KV cache.
            num_key_value_heads: Number of KV heads for full attention.
            head_dim: Dimension of each attention head.
            d_inner: Intermediate dimension for linear attention.
            d_conv: Convolution kernel size for linear attention.
            d_state: Recurrent state dimension for linear attention.
            layer_types: Per-layer attention type ("full_attention" or "linear_attention").
            num_attention_heads: Number of attention heads (defaults to num_key_value_heads).

        Returns:
            HybridCacheConfig instance.

        Raises:
            ValueError: If parameters are invalid.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if d_inner <= 0:
            raise ValueError("d_inner must be positive")
        if d_conv <= 0:
            raise ValueError("d_conv must be positive")
        if d_state <= 0:
            raise ValueError("d_state must be positive")
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                f"layer_types length ({len(layer_types)}) must match num_hidden_layers ({num_hidden_layers})"
            )

        # Validate layer types
        valid_types = {FULL_ATTENTION, LINEAR_ATTENTION, KDA_LINEAR_ATTENTION, PARALLEL_HYBRID}
        for i, lt in enumerate(layer_types):
            if lt not in valid_types:
                raise ValueError(f"Invalid layer_type at index {i}: {lt}. Must be one of {valid_types}")

        if num_attention_heads is None:
            num_attention_heads = num_key_value_heads

        return cls(
            num_hidden_layers=num_hidden_layers,
            partition_axis=partition_axis,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            d_inner=d_inner,
            d_conv=d_conv,
            d_state=d_state,
            num_attention_heads=num_attention_heads,
            layer_types=tuple(layer_types),
        )

    def is_full_attention(self, layer_idx: int) -> bool:
        """Check if a layer uses full attention."""
        return self.layer_types[layer_idx] == FULL_ATTENTION

    def is_linear_attention(self, layer_idx: int) -> bool:
        """Check if a layer uses linear attention (GatedDeltaNet)."""
        return self.layer_types[layer_idx] == LINEAR_ATTENTION

    def is_kda_attention(self, layer_idx: int) -> bool:
        """Check if a layer uses KDA linear attention (Kimi Linear)."""
        return self.layer_types[layer_idx] == KDA_LINEAR_ATTENTION

    def is_parallel_hybrid(self, layer_idx: int) -> bool:
        """Check if a layer uses parallel hybrid (attention + SSM in parallel)."""
        return self.layer_types[layer_idx] == PARALLEL_HYBRID


@auto_pytree
class HybridCacheView(BaseCacheView):
    """Single-layer cache view for hybrid attention models.

    Manages cache state for one layer, supporting either:
    - KV cache (for full_attention layers)
    - Conv state + recurrent state (for linear_attention layers)

    The view stores only the state type needed for its layer type,
    with the other fields set to None.

    Attributes:
        key (Array | None): Key cache for full attention.
            Shape: [batch_size, sequence_length, num_kv_heads, head_dim]
        value (Array | None): Value cache for full attention.
            Shape: [batch_size, sequence_length, num_kv_heads, head_dim]
        conv_state (Array | None): Convolution state for linear attention.
            Shape: [batch_size, d_inner, d_conv]
        recurrent_state (Array | None): Recurrent state for linear attention.
            Shape: [batch_size, num_heads, head_dim, d_state]
        positions (Array): Current position index per batch element.
            Shape: [batch_size]
        metadata (HybridCacheConfig): Static configuration metadata.
        layer_index (int | None): Index of this layer in the model.
        layer_type (str): Type of attention for this layer.
    """

    # KV cache (for full_attention)
    key: Float[Array, "batch seq num_kv_heads head_dim"] | ImplicitArray | None
    value: Float[Array, "batch seq num_kv_heads head_dim"] | ImplicitArray | None

    # Recurrent state (for linear_attention - GatedDeltaNet)
    conv_state: Float[Array, "batch d_inner d_conv"] | ImplicitArray | None
    recurrent_state: Float[Array, "batch num_heads head_dim d_state"] | ImplicitArray | None

    # Common (no defaults - must come before fields with defaults)
    positions: Int[Array, "batch"]  # noqa
    metadata: HybridCacheConfig

    # Separate conv states (for kda_linear_attention - Kimi Linear)
    # These have defaults and must come after fields without defaults
    q_conv_state: Float[Array, "batch key_dim d_conv"] | ImplicitArray | None = None
    k_conv_state: Float[Array, "batch key_dim d_conv"] | ImplicitArray | None = None
    v_conv_state: Float[Array, "batch value_dim d_conv"] | ImplicitArray | None = None

    layer_index: int | None = field(pytree_node=False, default=None)
    layer_type: str = field(pytree_node=False, default=FULL_ATTENTION)

    @classmethod
    def init(
        cls,
        config: HybridCacheConfig,
        layer_index: int | None = None,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        partition_specs: PartitionSpec | None = None,
    ) -> HybridCacheView:
        """Initialize a hybrid cache view for a single layer.

        Creates and allocates cache tensors based on the layer's attention type:
        - Full attention: Allocates KV cache
        - Linear attention (GatedDeltaNet): Allocates conv state and recurrent state
        - KDA linear attention (Kimi): Allocates separate Q/K/V conv states and recurrent state

        Args:
            config: Configuration for cache dimensions.
            layer_index: Index of this layer in the model.
            dtype: Data type for cache tensors.
            partition_specs: Sharding specification for distributed execution.

        Returns:
            HybridCacheView: Initialized cache view with allocated tensors.
        """
        metadata = config
        layer_type = metadata.layer_types[layer_index] if layer_index is not None else FULL_ATTENTION

        paxis = metadata.partition_axis
        if partition_specs is None:
            partition_specs = PartitionSpec(
                paxis.batch_axis,
                paxis.sequence_axis,
                paxis.head_axis,
                None,
            )

        # Initialize all fields to None
        key = None
        value = None
        conv_state = None
        recurrent_state = None
        q_conv_state = None
        k_conv_state = None
        v_conv_state = None

        if layer_type == FULL_ATTENTION:
            # Allocate KV cache
            key = with_sharding_constraint(
                arr=jnp.zeros(
                    shape=(
                        metadata.batch_size,
                        metadata.sequence_length,
                        metadata.num_key_value_heads,
                        metadata.head_dim,
                    ),
                    dtype=dtype,
                ),
                sharding=partition_specs,
            )
            value = with_sharding_constraint(
                arr=jnp.zeros(
                    shape=(
                        metadata.batch_size,
                        metadata.sequence_length,
                        metadata.num_key_value_heads,
                        metadata.head_dim,
                    ),
                    dtype=dtype,
                ),
                sharding=partition_specs,
            )
        elif layer_type == LINEAR_ATTENTION:
            # Allocate combined conv state + recurrent state for GatedDeltaNet
            conv_state = with_sharding_constraint(
                arr=jnp.zeros(
                    shape=(
                        metadata.batch_size,
                        metadata.d_inner,
                        metadata.d_conv,
                    ),
                    dtype=dtype,
                ),
                sharding=PartitionSpec(
                    metadata.partition_axis.batch_axis,
                    None,
                    None,
                ),
            )
            recurrent_state = with_sharding_constraint(
                arr=jnp.zeros(
                    shape=(
                        metadata.batch_size,
                        metadata.num_attention_heads,
                        metadata.head_dim,
                        metadata.d_state,
                    ),
                    dtype=dtype,
                ),
                sharding=PartitionSpec(
                    metadata.partition_axis.batch_axis,
                    metadata.partition_axis.head_axis,
                    None,
                    None,
                ),
            )
        elif layer_type == KDA_LINEAR_ATTENTION:
            # Allocate separate Q/K/V conv states + recurrent state for KDA (Kimi Linear)
            # For KDA, d_inner is split into key_dim (Q/K) and value_dim (V)
            # We use d_inner for key dimension and d_state for value dimension
            key_dim = metadata.num_attention_heads * metadata.head_dim
            value_dim = metadata.d_state  # In KDA, value has different dimension

            conv_sharding = PartitionSpec(
                metadata.partition_axis.batch_axis,
                None,
                None,
            )
            q_conv_state = with_sharding_constraint(
                arr=jnp.zeros(
                    shape=(metadata.batch_size, key_dim, metadata.d_conv),
                    dtype=dtype,
                ),
                sharding=conv_sharding,
            )
            k_conv_state = with_sharding_constraint(
                arr=jnp.zeros(
                    shape=(metadata.batch_size, key_dim, metadata.d_conv),
                    dtype=dtype,
                ),
                sharding=conv_sharding,
            )
            v_conv_state = with_sharding_constraint(
                arr=jnp.zeros(
                    shape=(metadata.batch_size, value_dim, metadata.d_conv),
                    dtype=dtype,
                ),
                sharding=conv_sharding,
            )
            recurrent_state = with_sharding_constraint(
                arr=jnp.zeros(
                    shape=(
                        metadata.batch_size,
                        metadata.num_attention_heads,
                        metadata.head_dim,
                        metadata.d_state,
                    ),
                    dtype=dtype,
                ),
                sharding=PartitionSpec(
                    metadata.partition_axis.batch_axis,
                    metadata.partition_axis.head_axis,
                    None,
                    None,
                ),
            )

        elif layer_type == PARALLEL_HYBRID:
            # Allocate BOTH KV cache AND recurrent state for parallel hybrid layers
            # (e.g., FalconH1 where attention and SSM run in parallel)

            # KV cache (same as FULL_ATTENTION)
            key = with_sharding_constraint(
                arr=jnp.zeros(
                    shape=(
                        metadata.batch_size,
                        metadata.sequence_length,
                        metadata.num_key_value_heads,
                        metadata.head_dim,
                    ),
                    dtype=dtype,
                ),
                sharding=partition_specs,
            )
            value = with_sharding_constraint(
                arr=jnp.zeros(
                    shape=(
                        metadata.batch_size,
                        metadata.sequence_length,
                        metadata.num_key_value_heads,
                        metadata.head_dim,
                    ),
                    dtype=dtype,
                ),
                sharding=partition_specs,
            )

            # Recurrent state (same as LINEAR_ATTENTION)
            conv_state = with_sharding_constraint(
                arr=jnp.zeros(
                    shape=(
                        metadata.batch_size,
                        metadata.d_inner,
                        metadata.d_conv,
                    ),
                    dtype=dtype,
                ),
                sharding=PartitionSpec(
                    metadata.partition_axis.batch_axis,
                    None,
                    None,
                ),
            )
            recurrent_state = with_sharding_constraint(
                arr=jnp.zeros(
                    shape=(
                        metadata.batch_size,
                        metadata.num_attention_heads,
                        metadata.head_dim,
                        metadata.d_state,
                    ),
                    dtype=dtype,
                ),
                sharding=PartitionSpec(
                    metadata.partition_axis.batch_axis,
                    metadata.partition_axis.head_axis,
                    None,
                    None,
                ),
            )

        return cls(
            key=key,
            value=value,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            q_conv_state=q_conv_state,
            k_conv_state=k_conv_state,
            v_conv_state=v_conv_state,
            positions=jnp.zeros((metadata.batch_size,), dtype="i4"),
            metadata=metadata,
            layer_index=layer_index,
            layer_type=layer_type,
        )

    def concatenate_to_cache(
        self,
        key_states: Float[Array, "batch seq_len num_kv_heads head_dim"] | None = None,
        value_states: Float[Array, "batch seq_len num_kv_heads head_dim"] | None = None,
        conv_state: Float[Array, "batch d_inner d_conv"] | None = None,
        recurrent_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
        q_conv_state: Float[Array, "batch key_dim d_conv"] | None = None,
        k_conv_state: Float[Array, "batch key_dim d_conv"] | None = None,
        v_conv_state: Float[Array, "batch value_dim d_conv"] | None = None,
        cache_position: Int[Array, "batch"] | None = None,  # noqa
    ) -> tuple[
        Float[Array, "batch seq num_kv_heads head_dim"] | None,
        Float[Array, "batch seq num_kv_heads head_dim"] | None,
        HybridCacheView,
    ]:
        """Update cache state based on layer type.

        This unified method handles all layer types:
        - FULL_ATTENTION: Updates KV cache with key_states and value_states
        - LINEAR_ATTENTION: Updates conv_state and recurrent_state
        - KDA_LINEAR_ATTENTION: Updates q/k/v conv states and recurrent_state

        Args:
            key_states: New key states for full attention layers.
            value_states: New value states for full attention layers.
            conv_state: New convolution state for linear attention layers.
            recurrent_state: New recurrent state for linear/KDA attention layers.
            q_conv_state: New Q convolution state for KDA layers.
            k_conv_state: New K convolution state for KDA layers.
            v_conv_state: New V convolution state for KDA layers.
            cache_position: Position in cache to update (for full attention).

        Returns:
            Tuple of (key_cache, value_cache, updated_view).
            For non-full-attention layers, key_cache and value_cache are None.
        """
        if self.layer_type == FULL_ATTENTION:
            # Update KV cache for full attention layers
            if key_states is None or value_states is None:
                return self.key, self.value, self

            _batch_size, seq_len = key_states.shape[:2]

            if cache_position is None:
                cache_position = self.positions

            new_key = self.key
            new_value = self.value

            # Simple update: assume cache_position is a scalar for all batches
            start_indices = (0, int(cache_position[0]), 0, 0)
            new_key = lax.dynamic_update_slice(new_key, key_states, start_indices)
            new_value = lax.dynamic_update_slice(new_value, value_states, start_indices)

            # Update positions
            new_positions = cache_position + seq_len

            return (
                new_key,
                new_value,
                self.replace(
                    key=new_key,
                    value=new_value,
                    positions=new_positions,
                ),
            )

        elif self.layer_type == LINEAR_ATTENTION:
            # Update conv state and recurrent state for linear attention
            new_conv_state = conv_state if conv_state is not None else self.conv_state
            new_recurrent_state = recurrent_state if recurrent_state is not None else self.recurrent_state

            return (
                None,
                None,
                self.replace(
                    conv_state=new_conv_state,
                    recurrent_state=new_recurrent_state,
                ),
            )

        elif self.layer_type == KDA_LINEAR_ATTENTION:
            # Update KDA states for kda_linear_attention layers
            new_q_conv_state = q_conv_state if q_conv_state is not None else self.q_conv_state
            new_k_conv_state = k_conv_state if k_conv_state is not None else self.k_conv_state
            new_v_conv_state = v_conv_state if v_conv_state is not None else self.v_conv_state
            new_recurrent_state = recurrent_state if recurrent_state is not None else self.recurrent_state

            return (
                None,
                None,
                self.replace(
                    q_conv_state=new_q_conv_state,
                    k_conv_state=new_k_conv_state,
                    v_conv_state=new_v_conv_state,
                    recurrent_state=new_recurrent_state,
                ),
            )

        elif self.layer_type == PARALLEL_HYBRID:
            # Update BOTH KV cache AND recurrent state for parallel hybrid layers
            # This is used by models like FalconH1 where attention and SSM run in parallel

            # Update KV cache (like FULL_ATTENTION)
            new_key = self.key
            new_value = self.value
            new_positions = self.positions

            if key_states is not None and value_states is not None:
                _batch_size, seq_len = key_states.shape[:2]

                if cache_position is None:
                    cache_position = self.positions

                start_indices = (0, int(cache_position[0]), 0, 0)
                new_key = lax.dynamic_update_slice(self.key, key_states, start_indices)
                new_value = lax.dynamic_update_slice(self.value, value_states, start_indices)
                new_positions = cache_position + seq_len

            # Update recurrent state (like LINEAR_ATTENTION)
            new_conv_state = conv_state if conv_state is not None else self.conv_state
            new_recurrent_state = recurrent_state if recurrent_state is not None else self.recurrent_state

            return (
                new_key,
                new_value,
                self.replace(
                    key=new_key,
                    value=new_value,
                    conv_state=new_conv_state,
                    recurrent_state=new_recurrent_state,
                    positions=new_positions,
                ),
            )

        else:
            raise ValueError(f"Unknown layer_type: {self.layer_type}")

    def update_recurrent_state(
        self,
        new_conv_state: Float[Array, "batch d_inner d_conv"] | None = None,
        new_recurrent_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
    ) -> HybridCacheView:
        """Update recurrent state for linear attention layers.

        This is a convenience wrapper around concatenate_to_cache for
        linear attention layers.

        Args:
            new_conv_state: New convolution state.
            new_recurrent_state: New recurrent state.

        Returns:
            Updated HybridCacheView.
        """
        _, _, updated_view = self.concatenate_to_cache(
            conv_state=new_conv_state,
            recurrent_state=new_recurrent_state,
        )
        return updated_view

    def update_conv_state(
        self,
        new_hidden_state: Float[Array, "batch d_inner"],
        cache_position: Int[Array, "..."] | None = None,
    ) -> HybridCacheView:
        """Update convolution state with rolling buffer.

        Implements a rolling buffer for convolutional states, where
        new states replace old ones in a circular fashion.

        Args:
            new_hidden_state: New hidden state to insert.
            cache_position: Position index for insertion (optional).

        Returns:
            Updated HybridCacheView.

        Raises:
            ValueError: If called on a layer that doesn't support conv state.
        """
        if self.layer_type not in (LINEAR_ATTENTION, PARALLEL_HYBRID):
            raise ValueError(
                f"update_conv_state is only valid for linear_attention or parallel_hybrid layers, got {self.layer_type}"
            )

        # Roll and update
        conv_state = jnp.roll(self.conv_state, shift=-1, axis=-1)
        conv_state = conv_state.at[:, :, -1].set(new_hidden_state)

        return self.replace(conv_state=conv_state)

    def update_kda_states(
        self,
        new_q_conv_state: Float[Array, "batch key_dim d_conv"] | None = None,
        new_k_conv_state: Float[Array, "batch key_dim d_conv"] | None = None,
        new_v_conv_state: Float[Array, "batch value_dim d_conv"] | None = None,
        new_recurrent_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
    ) -> HybridCacheView:
        """Update KDA states for kda_linear_attention layers (Kimi Linear).

        This is a convenience wrapper around concatenate_to_cache for
        KDA linear attention layers.

        Args:
            new_q_conv_state: New Q convolution state.
            new_k_conv_state: New K convolution state.
            new_v_conv_state: New V convolution state.
            new_recurrent_state: New recurrent state.

        Returns:
            Updated HybridCacheView.
        """
        _, _, updated_view = self.concatenate_to_cache(
            q_conv_state=new_q_conv_state,
            k_conv_state=new_k_conv_state,
            v_conv_state=new_v_conv_state,
            recurrent_state=new_recurrent_state,
        )
        return updated_view

    def reset(self) -> HybridCacheView:
        """Reset all cache states to zeros.

        Returns:
            Reset HybridCacheView with zeroed states.
        """
        if self.layer_type == FULL_ATTENTION:
            return self.replace(
                key=jnp.zeros_like(self.key),
                value=jnp.zeros_like(self.value),
                positions=jnp.zeros_like(self.positions),
            )
        elif self.layer_type == LINEAR_ATTENTION:
            return self.replace(
                conv_state=jnp.zeros_like(self.conv_state),
                recurrent_state=jnp.zeros_like(self.recurrent_state),
                positions=jnp.zeros_like(self.positions),
            )
        elif self.layer_type == KDA_LINEAR_ATTENTION:
            return self.replace(
                q_conv_state=jnp.zeros_like(self.q_conv_state),
                k_conv_state=jnp.zeros_like(self.k_conv_state),
                v_conv_state=jnp.zeros_like(self.v_conv_state),
                recurrent_state=jnp.zeros_like(self.recurrent_state),
                positions=jnp.zeros_like(self.positions),
            )
        elif self.layer_type == PARALLEL_HYBRID:
            return self.replace(
                key=jnp.zeros_like(self.key),
                value=jnp.zeros_like(self.value),
                conv_state=jnp.zeros_like(self.conv_state),
                recurrent_state=jnp.zeros_like(self.recurrent_state),
                positions=jnp.zeros_like(self.positions),
            )
        else:
            raise ValueError(f"Unknown layer_type: {self.layer_type}")

    def __repr__(self) -> str:
        if self.layer_type == FULL_ATTENTION:
            return (
                f"{self.__class__.__name__}(layer_type={self.layer_type}, "
                f"key={self.key.shape if self.key is not None else None}, "
                f"value={self.value.shape if self.value is not None else None}, "
                f"layer_index={self.layer_index})"
            )
        elif self.layer_type == LINEAR_ATTENTION:
            return (
                f"{self.__class__.__name__}(layer_type={self.layer_type}, "
                f"conv_state={self.conv_state.shape if self.conv_state is not None else None}, "
                f"recurrent_state={self.recurrent_state.shape if self.recurrent_state is not None else None}, "
                f"layer_index={self.layer_index})"
            )
        elif self.layer_type == KDA_LINEAR_ATTENTION:
            return (
                f"{self.__class__.__name__}(layer_type={self.layer_type}, "
                f"q_conv={self.q_conv_state.shape if self.q_conv_state is not None else None}, "
                f"k_conv={self.k_conv_state.shape if self.k_conv_state is not None else None}, "
                f"v_conv={self.v_conv_state.shape if self.v_conv_state is not None else None}, "
                f"recurrent={self.recurrent_state.shape if self.recurrent_state is not None else None}, "
                f"layer_index={self.layer_index})"
            )
        elif self.layer_type == PARALLEL_HYBRID:
            return (
                f"{self.__class__.__name__}(layer_type={self.layer_type}, "
                f"key={self.key.shape if self.key is not None else None}, "
                f"value={self.value.shape if self.value is not None else None}, "
                f"conv_state={self.conv_state.shape if self.conv_state is not None else None}, "
                f"recurrent_state={self.recurrent_state.shape if self.recurrent_state is not None else None}, "
                f"layer_index={self.layer_index})"
            )
        else:
            return f"{self.__class__.__name__}(layer_type={self.layer_type}, layer_index={self.layer_index})"

    __str__ = __repr__


@auto_pytree(frozen=False)
class ParallelHybridCacheView(BaseCacheView):
    """Cache view for layers that need multiple caches in parallel.

    Some decoder layers run more than one cache-bearing block in the same layer
    (e.g., attention KV-cache + SSM/Mamba state in FalconH1). This view carries
    both a TransformerCacheView and a RecurrentCacheView while presenting a
    single object to model code.
    """

    transformer: TransformerCacheView
    recurrent: RecurrentCacheView

    # Mirror recurrent fields so dataclasses.replace(self, conv_state=...) works.
    conv_state: Float[Array, "batch conv_dim conv_kernel_size"] | ImplicitArray | None = None
    recurrent_state: Float[Array, "batch ..."] | ImplicitArray | None = None
    positions: Int[Array, "batch"] | None = None  # noqa: F821
    seqlen_offset: Int[Array, ""] | None = None

    layer_index: int | None = field(pytree_node=False, default=None)

    def __post_init__(self):
        if self.recurrent is None:
            return

        conv_state = self.conv_state if self.conv_state is not None else getattr(self.recurrent, "conv_state", None)
        recurrent_state = (
            self.recurrent_state
            if self.recurrent_state is not None
            else getattr(self.recurrent, "recurrent_state", None)
        )
        positions = self.positions if self.positions is not None else getattr(self.recurrent, "positions", None)
        seqlen_offset = (
            self.seqlen_offset if self.seqlen_offset is not None else getattr(self.recurrent, "seqlen_offset", None)
        )

        self.conv_state = conv_state
        self.recurrent_state = recurrent_state
        self.positions = positions
        self.seqlen_offset = seqlen_offset

        needs_sync = (
            getattr(self.recurrent, "conv_state", None) is not conv_state
            or getattr(self.recurrent, "recurrent_state", None) is not recurrent_state
            or getattr(self.recurrent, "positions", None) is not positions
            or getattr(self.recurrent, "seqlen_offset", None) is not seqlen_offset
        )
        if not needs_sync:
            return

        update_dict: dict[str, tp.Any] = {
            "conv_state": conv_state,
            "recurrent_state": recurrent_state,
            "positions": positions,
        }
        if seqlen_offset is not None:
            update_dict["seqlen_offset"] = seqlen_offset
        self.recurrent = self.recurrent.replace(**update_dict)

    @property
    def key(self):
        return self.transformer.key

    @property
    def value(self):
        return self.transformer.value

    @property
    def indexs(self):
        return self.transformer.indexs

    @property
    def starts(self):
        return self.transformer.starts

    @property
    def masking_details(self):
        return getattr(self.transformer, "masking_details", None)

    @property
    def metadata(self):
        return getattr(self.transformer, "metadata", None)

    @classmethod
    def init(cls, metadata: BaseCacheConfig, *args, **kwargs) -> "ParallelHybridCacheView":
        del metadata, args
        transformer = kwargs.pop("transformer", None)
        recurrent = kwargs.pop("recurrent", None)
        layer_index = kwargs.pop("layer_index", None)
        if transformer is None or recurrent is None:
            raise ValueError("ParallelHybridCacheView.init requires `transformer` and `recurrent`.")
        if kwargs:
            raise TypeError(f"Unexpected kwargs for ParallelHybridCacheView.init: {tuple(kwargs.keys())}")
        return cls(transformer=transformer, recurrent=recurrent, layer_index=layer_index)

    def concatenate_to_cache(self, *args, **kwargs):
        del args

        is_transformer_call = "query" in kwargs or ("key" in kwargs and "value" in kwargs and "mask_info" in kwargs)
        is_recurrent_call = "conv_state" in kwargs or "recurrent_state" in kwargs

        if is_transformer_call and is_recurrent_call:
            raise TypeError("ParallelHybridCacheView.concatenate_to_cache received mixed transformer + recurrent args.")

        if is_transformer_call:
            key_cache, value_cache, mask_info, new_transformer, masking_details = self.transformer.concatenate_to_cache(
                **kwargs
            )
            new_view = self.replace(transformer=new_transformer)
            return key_cache, value_cache, mask_info, new_view, masking_details

        if is_recurrent_call:
            conv_state, recurrent_state, new_recurrent = self.recurrent.concatenate_to_cache(**kwargs)
            new_view = self.replace(
                recurrent=new_recurrent,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
                positions=getattr(new_recurrent, "positions", self.positions),
                seqlen_offset=getattr(new_recurrent, "seqlen_offset", self.seqlen_offset),
            )
            return conv_state, recurrent_state, new_view

        raise TypeError(
            "ParallelHybridCacheView.concatenate_to_cache expected transformer-style args "
            "(query/key/value/...) or recurrent-style args (conv_state/recurrent_state)."
        )


@auto_pytree(max_print_length=int(1e6))
class HybridCache(BaseCache):
    """Multi-layer cache container for hybrid attention models.

    Orchestrates cache views across all layers, providing a unified
    interface for state management during inference. Each layer
    maintains the appropriate cache type based on its attention type.

    Attributes:
        views (list[BaseCacheView | None]): Ordered list of cache views,
            one per model layer. Views can be any cache type (TransformerCacheView,
            RecurrentCacheView, KDACacheView).
    """

    views: list[KDACacheView | RecurrentCacheView | TransformerCacheView | HybridCacheView | ParallelHybridCacheView]

    @classmethod
    def init_cache(
        cls,
        config: HybridCacheConfig,
        dtype: jnp.dtype | None = None,
        partition_specs: PartitionSpec | None = None,
    ) -> HybridCache:
        """Initialize a complete hybrid cache with views for all layers.

        Creates a fully initialized cache with appropriate storage for each
        layer based on its attention type.

        Args:
            config: Configuration defining cache dimensions and layer types.
            dtype: Data type for cache tensors. Defaults to bfloat16.
            partition_specs: Sharding specification for KV cache.

        Returns:
            HybridCache: Fully initialized cache ready for inference.
        """
        if dtype is None:
            dtype = jnp.bfloat16

        return cls(
            views=[
                HybridCacheView.init(
                    config=config,
                    layer_index=layer_idx,
                    dtype=dtype,
                    partition_specs=partition_specs,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    @classmethod
    def init_empty(cls, num_hidden_layers: int) -> HybridCache:
        """Initialize an empty hybrid cache without allocated storage.

        Creates a cache structure with None views that can be populated later.

        Args:
            num_hidden_layers: Number of layers to create placeholders for.

        Returns:
            HybridCache: Cache instance with None views.
        """
        return cls(views=[None] * num_hidden_layers)

    def update_kv_cache(
        self,
        layer_idx: int,
        key_states: Float[Array, "batch seq_len num_kv_heads head_dim"],
        value_states: Float[Array, "batch seq_len num_kv_heads head_dim"],
        cache_position: Int[Array, "batch"] | None = None,  # noqa
    ) -> tuple[
        Float[Array, "batch seq num_kv_heads head_dim"],
        Float[Array, "batch seq num_kv_heads head_dim"],
        HybridCache,
    ]:
        """Update KV cache for a full attention layer.

        Args:
            layer_idx: Index of the layer to update.
            key_states: New key states.
            value_states: New value states.
            cache_position: Position in cache to update.

        Returns:
            Tuple of (key_cache, value_cache, updated_cache).
        """
        if self.views[layer_idx] is None:
            raise ValueError(f"Cache view for layer {layer_idx} is None")

        key_cache, value_cache, updated_view = self.views[layer_idx].concatenate_to_cache(
            key_states=key_states,
            value_states=value_states,
            cache_position=cache_position,
        )

        new_views = list(self.views)
        new_views[layer_idx] = updated_view
        return key_cache, value_cache, self.replace(views=new_views)

    def update_recurrent_state(
        self,
        layer_idx: int,
        new_conv_state: Float[Array, "batch d_inner d_conv"] | None = None,
        new_recurrent_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
    ) -> HybridCache:
        """Update recurrent state for a linear attention layer.

        Args:
            layer_idx: Index of the layer to update.
            new_conv_state: New convolution state.
            new_recurrent_state: New recurrent state.

        Returns:
            Updated HybridCache.
        """
        if self.views[layer_idx] is None:
            raise ValueError(f"Cache view for layer {layer_idx} is None")

        updated_view = self.views[layer_idx].update_recurrent_state(
            new_conv_state=new_conv_state,
            new_recurrent_state=new_recurrent_state,
        )

        new_views = list(self.views)
        new_views[layer_idx] = updated_view
        return self.replace(views=new_views)

    def reset(self) -> HybridCache:
        """Reset all cache layers to zero states.

        Returns:
            New HybridCache instance with all states zeroed.
        """
        new_views = [view.reset() if view is not None else None for view in self.views]
        return self.replace(views=new_views)

    def get_layer_type(self, layer_idx: int) -> str | None:
        """Get the attention type for a specific layer.

        Args:
            layer_idx: Index of the layer.

        Returns:
            Layer type string or None if view is not initialized.
        """
        if self.views[layer_idx] is None:
            return None
        return self.views[layer_idx].layer_type

    def get_view(self, layer_idx: int) -> HybridCacheView | None:
        """Get the cache view for a specific layer.

        Args:
            layer_idx: Index of the layer.

        Returns:
            HybridCacheView for the layer or None if not initialized.
        """
        return self.views[layer_idx]

    def update_view(self, layer_idx: int, view: HybridCacheView) -> HybridCache:
        """Update the cache view for a specific layer.

        Args:
            layer_idx: Index of the layer.
            view: New HybridCacheView for this layer.

        Returns:
            New HybridCache instance with the updated view.
        """
        new_views = list(self.views)
        new_views[layer_idx] = view
        return self.replace(views=new_views)

    def get_cache_position(self) -> Array:
        """Get the current cache position (number of tokens cached).

        Returns:
            Array of shape [batch_size] with current position for each sequence.
        """
        for view in self.views:
            if view is not None and view.positions is not None:
                return view.positions
        return jnp.zeros((1,), dtype=jnp.int32)

    def to_pure(self) -> tuple[list[dict[str, tp.Any]], list[str]]:
        """Convert cache to pure Python data structure for serialization.

        Extracts raw tensors and metadata for checkpointing or transfer.
        Since HybridCache can contain different view types, each layer's
        data is stored as a dictionary with type information.

        Returns:
            tuple: Pair of (cache_data, layer_types) where:
                - cache_data: List of dicts with layer-specific data
                - layer_types: List of layer type strings for reconstruction
        """
        cache_data = []
        layer_types = []

        for view in self.views:
            if view is None:
                cache_data.append(None)
                layer_types.append("none")
            elif hasattr(view, "key") and view.key is not None:
                # TransformerCacheView-like
                cache_data.append(
                    {
                        "key": view.key,
                        "value": view.value,
                        "indexs": getattr(view, "indexs", None),
                        "starts": getattr(view, "starts", None),
                        "positions": getattr(view, "positions", None),
                    }
                )
                layer_types.append(FULL_ATTENTION)
            elif hasattr(view, "conv_state"):
                # RecurrentCacheView-like or HybridCacheView with linear attention
                cache_data.append(
                    {
                        "conv_state": view.conv_state,
                        "recurrent_state": view.recurrent_state,
                        "positions": getattr(view, "positions", None),
                        # KDA-specific fields
                        "q_conv_state": getattr(view, "q_conv_state", None),
                        "k_conv_state": getattr(view, "k_conv_state", None),
                        "v_conv_state": getattr(view, "v_conv_state", None),
                    }
                )
                layer_type = getattr(view, "layer_type", LINEAR_ATTENTION)
                layer_types.append(layer_type)
            else:
                cache_data.append(None)
                layer_types.append("unknown")

        return cache_data, layer_types

    @classmethod
    def from_pure(
        cls,
        cache_data: list[dict[str, tp.Any]],
        layer_types: list[str],
    ) -> HybridCache:
        """Reconstruct cache from pure Python data structure.

        Restores a cache from serialized tensors and type info,
        typically after loading from disk or receiving from transfer.

        Args:
            cache_data: List of dicts with layer-specific tensors.
            layer_types: List of layer type strings.

        Returns:
            HybridCache: Reconstructed cache instance.
        """
        views = []

        for idx, (data, layer_type) in enumerate(zip(cache_data, layer_types, strict=True)):
            if data is None or layer_type == "none":
                views.append(None)
            elif layer_type == FULL_ATTENTION:
                # Create TransformerCacheView-like structure
                # Note: This creates a minimal view; for full functionality,
                # use layer_configs to create proper views
                views.append(
                    TransformerCacheView(
                        key=data["key"],
                        value=data["value"],
                        indexs=data.get("indexs"),
                        starts=data.get("starts"),
                        metadata=None,  # Metadata needs to be provided separately
                        layer_index=idx,
                    )
                )
            elif layer_type == LINEAR_ATTENTION:
                from easydel.layers.caching.recurrent import RecurrentCacheView

                views.append(
                    RecurrentCacheView(
                        conv_state=data["conv_state"],
                        recurrent_state=data["recurrent_state"],
                        positions=data.get("positions", jnp.zeros((data["conv_state"].shape[0],), dtype=jnp.int32)),
                        metadata=None,
                        layer_index=idx,
                    )
                )
            elif layer_type == KDA_LINEAR_ATTENTION:
                views.append(
                    KDACacheView(
                        q_conv_state=data["q_conv_state"],
                        k_conv_state=data["k_conv_state"],
                        v_conv_state=data["v_conv_state"],
                        recurrent_state=data["recurrent_state"],
                        positions=data.get("positions", jnp.zeros((data["q_conv_state"].shape[0],), dtype=jnp.int32)),
                        metadata=None,
                        layer_index=idx,
                    )
                )
            else:
                views.append(None)

        return cls(views=views)

    def insert(
        self,
        other: HybridCache,
        slot: int,
        partition_manager: PartitionManager | None = None,
        quantizer: tp.Any | None = None,
    ) -> HybridCache:
        """Insert another cache's contents at specified batch slot.

        Copies states from another cache into this cache at the specified
        batch position. Works with all view types (Transformer, Recurrent, KDA).

        Args:
            other: Source cache to copy from.
            slot: Batch slot index to insert into.
            partition_manager: Optional sharding configuration (for transformer views).
            quantizer: Optional quantization configuration (for transformer views).

        Returns:
            HybridCache: Updated cache instance.
        """
        new_views = list(self.views)

        for idx in range(len(self.views)):
            view = self.views[idx]
            oview = other.views[idx]

            if view is None or oview is None:
                continue

            layer_type = getattr(view, "layer_type", FULL_ATTENTION)

            if layer_type == FULL_ATTENTION and hasattr(view, "key"):
                # TransformerCacheView-like
                new_key = lax.dynamic_update_slice(
                    view.key,
                    oview.key.astype(view.key.dtype),
                    (slot, 0, 0, 0),
                )
                new_value = lax.dynamic_update_slice(
                    view.value,
                    oview.value.astype(view.value.dtype),
                    (slot, 0, 0, 0),
                )

                update_dict = {"key": new_key, "value": new_value}

                if hasattr(view, "indexs") and view.indexs is not None:
                    update_dict["indexs"] = lax.dynamic_update_slice_in_dim(view.indexs, oview.indexs, slot, 0)
                if hasattr(view, "starts") and view.starts is not None:
                    update_dict["starts"] = lax.dynamic_update_slice_in_dim(view.starts, oview.starts, slot, 0)

                new_views[idx] = view.replace(**update_dict)

            elif layer_type == LINEAR_ATTENTION and hasattr(view, "conv_state"):
                # RecurrentCacheView-like
                new_conv = (
                    lax.dynamic_update_slice_in_dim(view.conv_state, oview.conv_state, slot, 0)
                    if view.conv_state is not None
                    else None
                )

                new_recurrent = (
                    lax.dynamic_update_slice_in_dim(view.recurrent_state, oview.recurrent_state, slot, 0)
                    if view.recurrent_state is not None
                    else None
                )

                update_dict = {
                    "conv_state": new_conv,
                    "recurrent_state": new_recurrent,
                }

                if hasattr(view, "positions") and view.positions is not None:
                    update_dict["positions"] = lax.dynamic_update_slice_in_dim(view.positions, oview.positions, slot, 0)

                new_views[idx] = view.replace(**update_dict)

            elif layer_type == KDA_LINEAR_ATTENTION:
                # KDACacheView-like
                update_dict = {}

                for field in ["q_conv_state", "k_conv_state", "v_conv_state", "recurrent_state"]:
                    view_val = getattr(view, field, None)
                    oview_val = getattr(oview, field, None)
                    if view_val is not None and oview_val is not None:
                        update_dict[field] = lax.dynamic_update_slice_in_dim(view_val, oview_val, slot, 0)

                if hasattr(view, "positions") and view.positions is not None:
                    update_dict["positions"] = lax.dynamic_update_slice_in_dim(view.positions, oview.positions, slot, 0)

                new_views[idx] = view.replace(**update_dict)

        return self.replace(views=new_views)

    def __repr__(self):
        return f"{self.__class__.__name__}(\n  " + "\n  ".join(str(view) for view in self.views) + "\n)"

    __str__ = __repr__


@auto_pytree
class HybridMetadata(BaseRunTimeMetadata):
    """Runtime metadata for hybrid cache operations.

    Stores dynamic information that varies during model execution
    but isn't part of the permanent cache state. Since HybridCache
    contains multiple view types, this metadata embeds the necessary
    fields for TransformerCacheView layers.

    Attributes:
        postpadded: Whether sequences are post-padded.
        starts: Starting positions for sequences.
        indexs: Current position indices.
    """

    postpadded: bool = False
    starts: tp.Any | None = None  # jnp.ndarray
    indexs: tp.Any | None = None  # jnp.ndarray


if __name__ == "__main__":
    from eformer.escale import PartitionAxis

    print("Testing HybridCache...")

    num_layers = 8
    layer_types = tuple("full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(num_layers))

    print(f"Layer types: {layer_types}")

    metadata = HybridCacheConfig.create(
        num_hidden_layers=num_layers,
        partition_axis=PartitionAxis(),
        batch_size=2,
        sequence_length=128,
        num_key_value_heads=8,
        head_dim=64,
        d_inner=1024,
        d_conv=4,
        d_state=64,
        layer_types=layer_types,
    )

    print(f"Created metadata with {metadata.num_hidden_layers} layers")

    cache = HybridCache.init_cache(metadata, dtype=jnp.float32)
    print(f"Initialized cache with {len(cache)} views")

    # Print layer types
    for i, view in enumerate(cache.views):
        print(f"  Layer {i}: {view.layer_type}")
        if view.layer_type == FULL_ATTENTION:
            print(f"    KV cache: key={view.key.shape}, value={view.value.shape}")
        else:
            print(f"    Recurrent state: conv={view.conv_state.shape}, recurrent={view.recurrent_state.shape}")

    # Test KV cache update on a full attention layer (layer 3)
    print("\nTesting KV cache update on layer 3 (full_attention)...")
    key_states = jnp.ones((2, 10, 8, 64))
    value_states = jnp.ones((2, 10, 8, 64))
    key_cache, value_cache, cache = cache.update_kv_cache(3, key_states, value_states)
    print(f"  Updated key cache shape: {key_cache.shape}")
    print(f"  Updated value cache shape: {value_cache.shape}")

    # Test recurrent state update on a linear attention layer (layer 0)
    print("\nTesting recurrent state update on layer 0 (linear_attention)...")
    new_conv_state = jnp.ones((2, 1024, 4))
    new_recurrent_state = jnp.ones((2, 8, 64, 64))
    cache = cache.update_recurrent_state(0, new_conv_state, new_recurrent_state)
    print(f"  Updated conv state: {cache.views[0].conv_state.shape}")
    print(f"  Updated recurrent state: {cache.views[0].recurrent_state.shape}")

    print("\nAll tests passed!")
