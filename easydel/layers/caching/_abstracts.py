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

"""Abstract base classes for caching systems in EasyDeL.

This module provides the foundational abstract classes that define the interface
for all caching implementations in EasyDeL. These abstractions enable different
caching strategies (transformer KV-cache, paged attention, state-space models, etc.)
to share a common interface while allowing for architecture-specific optimizations.

The caching system is built on a three-tier hierarchy:
1. Metadata classes that store configuration parameters
2. View classes that manage cache state for individual layers
3. Cache classes that orchestrate multiple views across all layers

Key Classes:
    BaseCacheMetadata: Abstract base for cache configuration metadata
    BaseRunTimeMetadata: Abstract base for runtime metadata during computation
    BaseCacheView: Abstract base for single-layer cache management
    BaseCache: Abstract base for multi-layer cache orchestration

Design Principles:
    - Functional updates: All cache modifications return new instances
    - PyTree compatibility: All classes use auto_pytree for JAX integration
    - Type safety: Strong typing with generics and protocols
    - Extensibility: Easy to add new caching strategies

Example:
    To implement a new caching strategy, extend the base classes:

    >>> class MyCustomMetadata(BaseCacheMetadata):
    ...     my_param: int
    ...
    ...     @classmethod
    ...     def create(cls, my_param: int) -> MyCustomMetadata:
    ...         if my_param <= 0:
    ...             raise ValueError("my_param must be positive")
    ...         return cls(my_param=my_param)

    >>> class MyCustomView(BaseCacheView):
    ...     # Implementation details
    ...     pass
"""

from __future__ import annotations

import typing as tp
from abc import ABC, abstractmethod

from eformer.pytree import auto_pytree
from jaxtyping import Array, Bool, Float, Int


@auto_pytree
class BaseCacheMetadata(ABC):
    """Abstract base class defining the interface for cache metadata.

    This class serves as the foundation for all cache metadata implementations.
    Metadata objects store static configuration that defines how a cache should
    be initialized and operated. They are immutable after creation and can be
    safely shared across multiple cache instances.

    The metadata pattern separates configuration from state, allowing:
    - Validation of parameters at creation time
    - Reuse of configurations across multiple caches
    - Serialization of cache configurations
    - Type-safe parameter passing

    Concrete implementations must:
    - Define all required configuration fields as class attributes
    - Implement the create() factory method with validation
    - Use the @auto_pytree decorator for JAX compatibility
    - Ensure all fields are immutable types or frozen dataclasses

    Attributes:
        All attributes are implementation-specific and should be documented
        in the concrete class.

    Note:
        The @auto_pytree decorator makes this class compatible with JAX's
        PyTree protocol, enabling it to be used in JAX transformations.
    """

    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs) -> BaseCacheMetadata:
        """Factory method to create and validate a metadata instance.

        This method serves as the primary constructor for metadata objects,
        providing a centralized location for parameter validation and
        initialization logic. It should be used instead of direct instantiation
        to ensure all metadata objects are properly validated.

        The factory pattern allows for:
        - Complex initialization logic beyond simple assignment
        - Parameter validation before object creation
        - Derived parameter calculation
        - Consistent error handling across implementations

        Args:
            *args: Positional arguments for metadata creation.
                Implementation-specific parameters.
            **kwargs: Keyword arguments for metadata creation.
                Implementation-specific parameters.

        Returns:
            BaseCacheMetadata: A validated instance of the concrete metadata
                implementation. The returned object is immutable and ready
                for use in cache initialization.

        Raises:
            ValueError: If any validation checks fail. Common validations include:
                - Positive integer checks for dimensions and sizes
                - Range checks for ratios and percentages
                - Consistency checks between related parameters
                - Resource availability checks
            TypeError: If required parameters are missing or have incorrect types.

        Example:
            >>> metadata = TransformerCacheMetaData.create(
            ...     batch_size=4,
            ...     sequence_length=1024,
            ...     num_hidden_layers=12,
            ...     num_heads=8,
            ...     head_dim=64
            ... )
        """
        pass


@auto_pytree
class BaseRunTimeMetadata:
    """Abstract base class for runtime metadata used during cache operations.

    Runtime metadata captures dynamic information that varies during model
    execution but isn't part of the permanent cache state. This includes
    temporary computation state, indices, masks, and other ephemeral data
    needed during cache updates.

    Unlike cache metadata (which is static configuration), runtime metadata:
    - Changes frequently during model execution
    - Is specific to individual forward passes
    - May not need to be preserved between calls
    - Often contains computation intermediates

    Common uses for runtime metadata:
    - Current sequence positions and offsets
    - Batch-specific indices and mappings
    - Temporary buffers for computation
    - Attention masks and patterns
    - Memory management information

    Concrete implementations should:
    - Keep fields lightweight and computation-focused
    - Avoid storing large tensors unless necessary
    - Use appropriate JAX dtypes for indices (int32)
    - Document which fields are required vs optional

    Note:
        This class intentionally has no abstract methods, as runtime
        metadata requirements vary significantly between cache types.
        Implementations should add fields and methods as needed.
    """


class BaseCacheView(ABC):
    """Abstract base class for single-layer cache management.

    A cache view represents the cache state for a single layer in a neural
    network. It encapsulates the layer-specific cached data (e.g., key/value
    pairs for attention, conv/SSM states for Mamba) and provides methods to
    update this data during inference.

    The view pattern allows:
    - Layer-specific optimization and sharding
    - Independent cache management per layer
    - Flexible cache formats for different layer types
    - Efficient memory layout for each layer's needs

    Key responsibilities:
    - Store cached states for one model layer
    - Track the current position/index in the cache
    - Update cache with new computed states
    - Apply quantization if configured
    - Manage memory layout and sharding

    Design principles:
    - Functional updates: Methods return new instances, not modify in-place
    - Layer isolation: Each view is independent of others
    - Type flexibility: Support both dense and quantized representations
    - Sharding aware: Integrate with JAX's sharding system

    Attributes:
        metadata (BaseCacheMetadata): Configuration metadata for this cache.
            Shared across all views in the same cache hierarchy.
        layer_index (int | None): The index of the layer this view represents.
            None for cache types that don't have layer structure.

    Note:
        While marked as ABC, this class doesn't use @auto_pytree because
        concrete implementations need to control their PyTree structure.
    """

    metadata: BaseCacheMetadata
    layer_index: int | None

    @classmethod
    @abstractmethod
    def init(cls, metadata: BaseCacheMetadata, *args, **kwargs) -> BaseCacheView:
        """Initialize a new cache view for a single layer.

        This factory method creates and initializes a cache view with the
        appropriate tensor shapes, dtypes, and sharding for a specific layer.
        It allocates the actual cache storage and sets up initial state.

        The initialization process typically:
        1. Calculates tensor shapes from metadata
        2. Determines sharding strategy for distributed execution
        3. Allocates cache tensors with appropriate dtype
        4. Applies quantization if configured
        5. Sets initial indices and positions

        Args:
            metadata (BaseCacheMetadata): Static configuration metadata that
                defines cache dimensions, dtypes, and behavior.
            *args: Additional positional arguments. Common args include:
                - mesh: JAX device mesh for sharding
                - dtype: JAX dtype for cache tensors
                - layer_index: Index of the layer
                - partition_manager: Sharding configuration
            **kwargs: Additional keyword arguments. Common kwargs include:
                - quantizer: Quantization configuration
                - initial_position: Starting cache position
                - device: Specific device placement

        Returns:
            BaseCacheView: An initialized cache view ready for use.
                The view contains allocated tensors and is configured
                for the specific layer's requirements.

        Raises:
            ValueError: If metadata parameters are invalid for this view type.
            MemoryError: If cache allocation fails due to insufficient memory.

        Example:
            >>> view = TransformerCacheView.init(
            ...     metadata=metadata,
            ...     mesh=mesh,
            ...     dtype=jnp.bfloat16,
            ...     layer_index=0
            ... )
        """
        pass

    @abstractmethod
    def concatenate_to_cache(self, *args, **kwargs) -> tp.Any:
        """Update the cache with new computed states.

        This is the primary method for cache updates during inference.
        It takes newly computed states (keys, values, hidden states, etc.)
        and incorporates them into the cache, returning updated tensors
        and any additional information needed for computation.

        The update process typically:
        1. Validates input shapes and dtypes
        2. Determines update position in cache
        3. Applies quantization if configured
        4. Updates cache tensors functionally
        5. Adjusts masks and indices
        6. Returns updated state for next computation

        Args:
            *args: Positional arguments vary by cache type but commonly include:
                - key: New key states (for attention caches)
                - value: New value states (for attention caches)
                - hidden_states: New hidden states (for SSM caches)
                - positions: Sequence positions for update
            **kwargs: Keyword arguments vary by cache type but commonly include:
                - attention_mask: Mask for valid positions
                - cache_metadata: Runtime metadata for update
                - quantizer: Quantization function
                - causal_mask: Causal attention pattern
                - mode: Prefill vs generation mode

        Returns:
            tp.Any: Return type varies by implementation but typically includes:
                - Updated cache tensors (functional return)
                - Modified attention masks
                - Updated view instance
                - Additional computation results

            Common return patterns:
            - Transformer: (key_cache, value_cache, mask, updated_view)
            - Mamba: (updated_view,)
            - Paged: (updated_view,)

        Note:
            This method should be functional, returning new tensors rather
            than modifying existing ones in-place. This ensures compatibility
            with JAX's functional programming model.

        Example:
            >>> key_cache, value_cache, mask, new_view = view.concatenate_to_cache(
            ...     query=query_states,
            ...     key=key_states,
            ...     value=value_states,
            ...     attention_mask=mask
            ... )
        """
        pass


class BaseCache(ABC):
    """Abstract base class for multi-layer cache orchestration.

    A cache container manages cache views across all layers of a model,
    providing a unified interface for cache initialization, access, and
    updates. It acts as the top-level cache object that users interact with.

    The cache container pattern enables:
    - Centralized cache management across layers
    - Batch operations on all cache views
    - Consistent initialization and configuration
    - Easy serialization and checkpointing

    Key responsibilities:
    - Maintain a collection of cache views (one per layer)
    - Provide factory methods for cache initialization
    - Enable indexed access to individual layer caches
    - Support batch operations across all layers
    - Handle cache serialization and restoration

    Design principles:
    - Composition: Aggregates multiple cache views
    - Consistency: Ensures all views share compatible configuration
    - Flexibility: Supports different view types per layer if needed
    - Convenience: Provides list-like interface for view access

    Attributes:
        views (tp.Sequence[BaseCacheView | None]): Ordered collection of
            cache views, one per model layer. None values indicate
            uninitialized or disabled cache for that layer.

    Note:
        The class provides default implementations for common operations
        like indexing and length, but concrete classes must implement
        the initialization methods.
    """

    views: tp.Sequence[BaseCacheView | None]

    @classmethod
    @abstractmethod
    def init_cache(
        cls,
        metadata: BaseCacheMetadata,
        *args,
        **kwargs,
    ) -> BaseCache:
        """Initialize a complete cache with views for all layers.

        This factory method creates a fully initialized cache with allocated
        storage for all layers. It's the primary way to create a cache for
        inference, setting up all necessary views with consistent configuration.

        The initialization process:
        1. Validates metadata configuration
        2. Determines resource allocation strategy
        3. Creates views for each layer
        4. Applies sharding and quantization
        5. Returns ready-to-use cache

        Args:
            metadata (BaseCacheMetadata): Configuration metadata defining
                cache dimensions, number of layers, and behavior.
            *args: Additional positional arguments. Common args include:
                - mesh: JAX device mesh for distributed execution
                - dtype: Default dtype for cache tensors
                - num_layers: Override for number of layers
            **kwargs: Additional keyword arguments. Common kwargs include:
                - partition_manager: Sharding configuration
                - quantizer: Quantization settings
                - device: Device placement preferences
                - initial_positions: Starting positions per layer

        Returns:
            BaseCache: A fully initialized cache with views for all layers.
                Ready for use in model inference.

        Raises:
            ValueError: If metadata is incompatible with cache type.
            MemoryError: If insufficient memory for allocation.
            RuntimeError: If device/sharding configuration fails.

        Example:
            >>> cache = TransformerCache.init_cache(
            ...     metadata=metadata,
            ...     mesh=mesh,
            ...     dtype=jnp.bfloat16,
            ...     partition_manager=pm
            ... )
            >>> print(f"Initialized cache with {len(cache)} layers")
        """
        pass

    @classmethod
    @abstractmethod
    def init_empty(cls, *args, **kwargs) -> BaseCache:
        """Initialize an empty cache container without allocated storage.

        Creates a cache structure with placeholder views that can be
        populated later. This is useful for:
        - Gradual cache building during training
        - Memory-efficient initialization
        - Dynamic cache allocation
        - Testing and debugging

        The empty cache has the correct structure but no allocated tensors,
        allowing the shape and configuration to be determined dynamically.

        Args:
            *args: Positional arguments. Common args include:
                - num_layers: Number of layers to create placeholders for
            **kwargs: Keyword arguments for future compatibility.

        Returns:
            BaseCache: A cache instance with uninitialized (None) views.
                Views must be populated before use.

        Example:
            >>> cache = TransformerCache.init_empty(num_hidden_layers=12)
            >>> # Populate views gradually
            >>> for i in range(12):
            ...     cache[i] = TransformerCacheView.init(...)
        """
        pass

    def __getitem__(self, index):
        """Access cache views by index using subscript notation.

        Provides a convenient list-like interface for accessing individual
        layer caches. Supports all standard Python indexing operations
        including negative indices and slicing.

        Args:
            index: Index of the cache view to retrieve. Can be:
                - int: Single layer index (e.g., cache[0])
                - slice: Range of layers (e.g., cache[1:3])
                - negative int: Index from end (e.g., cache[-1])

        Returns:
            BaseCacheView | None: The cache view at the specified index,
                or None if the view is uninitialized. For slice indices,
                returns a list of views.

        Raises:
            IndexError: If index is out of range.
            AttributeError: If views have not been initialized.

        Example:
            >>> first_layer = cache[0]
            >>> last_layer = cache[-1]
            >>> middle_layers = cache[4:8]
        """
        return self.views[index]

    def __setitem__(self, index, value):
        """Update cache views by index using subscript notation.

        Allows modification of individual cache views after initialization.
        This is useful for:
        - Gradual cache population
        - Replacing views with updated versions
        - Selective layer updates
        - Dynamic cache reconfiguration

        Args:
            index: Index of the cache view to update. Must be a valid
                integer index within the range of existing views.
            value (BaseCacheView | None): New cache view to assign at
                the index. Can be None to clear a cache view.

        Raises:
            IndexError: If index is out of range.
            AttributeError: If views have not been initialized.
            TypeError: If value is not a compatible cache view type.

        Example:
            >>> # Replace a specific layer's cache
            >>> cache[5] = TransformerCacheView.init(...)
            >>> # Clear a layer's cache
            >>> cache[5] = None
        """
        self.views[index] = value

    def __len__(self) -> int:
        """Return the number of cache views in this container.

        Provides the length of the cache, which typically corresponds to
        the number of layers in the model. This enables:
        - Iteration over cache views
        - Validation of layer counts
        - Cache size inspection

        Returns:
            int: The number of cache views (including None placeholders).
                Usually equals the number of model layers.

        Raises:
            AttributeError: If `self.views` has not been initialized by
                a subclass. This indicates improper cache initialization.

        Example:
            >>> cache = TransformerCache.init_cache(...)
            >>> print(f"Cache has {len(cache)} layers")
            >>> for i in range(len(cache)):
            ...     process_layer(cache[i])
        """
        if not hasattr(self, "views"):
            raise AttributeError(
                "The 'views' attribute has not been initialized. Ensure a concrete subclass initializes it."
            )
        return len(self.views)
