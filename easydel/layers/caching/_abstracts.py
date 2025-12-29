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
    BaseCacheConfig: Abstract base for cache configuration metadata
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

    >>> class MyCustomMetadata(BaseCacheConfig):
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


@auto_pytree
class BaseCacheConfig(ABC):
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
    def create(cls, *args, **kwargs) -> BaseCacheConfig:
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
            BaseCacheConfig: A validated instance of the concrete metadata
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
            >>> metadata = TransformerCacheConfig.create(
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


@auto_pytree
class OperationsMetadata(BaseRunTimeMetadata):
    """Unified runtime metadata for all cache types using composition.

    This class provides a single entry point for runtime metadata across all
    cache types. Instead of requiring users to know which specific metadata
    class to use, they can use OperationsMetadata which internally holds
    the appropriate type-specific metadata.

    The composition approach:
    - Keeps existing metadata classes unchanged (backward compatible)
    - Provides unified access to common fields
    - Allows type-specific access when needed
    - Works with the dynamic operation discovery system

    Only one of the type-specific metadata fields should be populated at a time,
    matching the cache type being used.

    Attributes:
        transformer: TransformerMetadata for standard KV-cache operations.
        hybrid: HybridMetadata for hybrid attention models.
        ragged: RaggedPagesMetadata for paged attention.
        recurrent: RecurrentMetadata for recurrent/linear attention models.

    Example:
        >>> # Create for transformer cache
        >>> metadata = OperationsMetadata.for_transformer(
        ...     starts=jnp.zeros((batch_size,), dtype=jnp.int32)
        ... )
        >>>
        >>> # Create for hybrid cache
        >>> metadata = OperationsMetadata.for_hybrid()
        >>>
        >>> # Access type-specific metadata
        >>> if metadata.transformer is not None:
        ...     print(metadata.transformer.starts)
    """

    transformer: tp.Any | None = None  # TransformerMetadata
    hybrid: tp.Any | None = None  # HybridMetadata
    ragged: tp.Any | None = None  # RaggedPagesMetadata
    recurrent: tp.Any | None = None  # RecurrentMetadata

    @classmethod
    def for_transformer(
        cls,
        postpadded: bool = False,
        starts: tp.Any | None = None,
        indexs: tp.Any | None = None,
    ) -> "OperationsMetadata":
        """Create OperationsMetadata for transformer cache.

        Args:
            postpadded: Whether sequences are post-padded.
            starts: Starting positions for sequences.
            indexs: Current position indices.

        Returns:
            OperationsMetadata with transformer field populated.
        """
        from easydel.layers.caching.transformer import TransformerMetadata

        return cls(transformer=TransformerMetadata(postpadded=postpadded, starts=starts, indexs=indexs))

    @classmethod
    def for_hybrid(
        cls,
        postpadded: bool = False,
        starts: tp.Any | None = None,
        indexs: tp.Any | None = None,
    ) -> "OperationsMetadata":
        """Create OperationsMetadata for hybrid cache.

        Since HybridCache contains multiple view types, the metadata includes
        fields needed by TransformerCacheView layers (postpadded, starts, indexs).
        Recurrent layers don't need additional metadata during inference.

        Args:
            postpadded: Whether sequences are post-padded.
            starts: Starting positions for sequences.
            indexs: Current position indices.

        Returns:
            OperationsMetadata with hybrid field populated.
        """
        from easydel.layers.caching.hybrid import HybridMetadata

        return cls(hybrid=HybridMetadata(postpadded=postpadded, starts=starts, indexs=indexs))

    @classmethod
    def for_ragged(
        cls,
        pages_tables: tp.Any,
        context_lens: tp.Any,
        query_start_loc: tp.Any,
        num_seqs: tp.Any,
        slot_mapping: tp.Any | None = None,
        position_ids: tp.Any | None = None,
        request_distribution: tp.Any | None = None,
        num_kv_update_slices: tp.Any | None = None,
        version: str = "v3",
        page_size: int = 128,
        prefill_chunk_size: int = 512,
    ) -> "OperationsMetadata":
        """Create OperationsMetadata for ragged pages cache.

        Args:
            pages_tables: Page tables mapping.
            context_lens: Context lengths per sequence.
            query_start_loc: Query start locations.
            num_seqs: Number of sequences.
            slot_mapping: Slot mapping for v2.
            position_ids: Position IDs.
            request_distribution: Request distribution for v3.
            num_kv_update_slices: KV update slices for v2.
            version: Version "v2" or "v3".
            page_size: Page size.
            prefill_chunk_size: Prefill chunk size.

        Returns:
            OperationsMetadata with ragged field populated.
        """
        from easydel.layers.caching.ragged_page import RaggedPagesMetadata

        return cls(
            ragged=RaggedPagesMetadata(
                pages_tables=pages_tables,
                context_lens=context_lens,
                query_start_loc=query_start_loc,
                num_seqs=num_seqs,
                slot_mapping=slot_mapping,
                position_ids=position_ids,
                request_distribution=request_distribution,
                num_kv_update_slices=num_kv_update_slices,
                version=version,
                page_size=page_size,
                prefill_chunk_size=prefill_chunk_size,
            )
        )

    @classmethod
    def for_recurrent(cls) -> "OperationsMetadata":
        """Create OperationsMetadata for recurrent cache.

        Returns:
            OperationsMetadata with recurrent field populated.
        """
        from easydel.layers.caching.recurrent import RecurrentMetadata

        return cls(recurrent=RecurrentMetadata())

    @property
    def cache_type(self) -> str:
        """Determine the cache type based on which field is populated.

        Returns:
            str: "transformer", "hybrid", "ragged", or "recurrent".
        """
        if self.transformer is not None:
            return "transformer"
        if self.hybrid is not None:
            return "hybrid"
        if self.ragged is not None:
            return "ragged"
        if self.recurrent is not None:
            return "recurrent"
        return "unknown"

    def get_inner(self) -> BaseRunTimeMetadata | None:
        """Get the inner type-specific metadata.

        Returns:
            The populated metadata instance, or None if none populated.
        """
        if self.transformer is not None:
            return self.transformer
        if self.hybrid is not None:
            return self.hybrid
        if self.ragged is not None:
            return self.ragged
        if self.recurrent is not None:
            return self.recurrent
        return None


def unwrap_metadata(metadata: tp.Any, expected_type: str | None = None) -> tp.Any:
    """Unwrap OperationsMetadata or HybridMetadata to the inner type-specific metadata.

    This helper function allows cache views and operations to accept:
    - Specific metadata types (e.g., TransformerMetadata)
    - OperationsMetadata (unified wrapper)
    - HybridMetadata (when using HybridCache)

    And automatically extract/convert to the appropriate type.

    Special handling for HybridMetadata:
    - When expected_type is "transformer", extracts TransformerMetadata from
      HybridMetadata's embedded fields (postpadded, starts, indexs).
    - This enables HybridCache to work seamlessly with TransformerCacheView layers.

    Args:
        metadata: Either a specific metadata type, OperationsMetadata, or HybridMetadata.
        expected_type: Optional expected cache type ("transformer", "hybrid",
            "ragged", "recurrent"). If provided, extracts/converts to that type.

    Returns:
        The unwrapped metadata. If metadata is already the expected type, returns
        it unchanged. If metadata is OperationsMetadata or HybridMetadata, returns
        the inner metadata converted to the expected type.

    Example:
        >>> # In TransformerCacheView.concatenate_to_cache:
        >>> cache_metadata = unwrap_metadata(cache_metadata, "transformer")
        >>> # cache_metadata is now TransformerMetadata (or None)
    """
    if metadata is None:
        return None

    # If it's OperationsMetadata, extract the inner metadata
    if isinstance(metadata, OperationsMetadata):
        if expected_type == "transformer":
            # First check if we have direct transformer metadata
            if metadata.transformer is not None:
                return metadata.transformer
            # If we have hybrid metadata, convert to transformer metadata
            if metadata.hybrid is not None:
                from easydel.layers.caching.transformer import TransformerMetadata

                return TransformerMetadata(
                    postpadded=getattr(metadata.hybrid, "postpadded", False),
                    starts=getattr(metadata.hybrid, "starts", None),
                    indexs=getattr(metadata.hybrid, "indexs", None),
                )
            return None
        elif expected_type == "hybrid":
            return metadata.hybrid
        elif expected_type == "ragged":
            return metadata.ragged
        elif expected_type == "recurrent":
            return metadata.recurrent
        else:
            # Return whatever is populated
            return metadata.get_inner()

    # Check if it's HybridMetadata and we need TransformerMetadata
    # Import here to avoid circular import
    from easydel.layers.caching.hybrid import HybridMetadata

    if isinstance(metadata, HybridMetadata) and expected_type == "transformer":
        from easydel.layers.caching.transformer import TransformerMetadata

        return TransformerMetadata(
            postpadded=getattr(metadata, "postpadded", False),
            starts=getattr(metadata, "starts", None),
            indexs=getattr(metadata, "indexs", None),
        )

    # Already a specific type, return as-is
    return metadata


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
        metadata (BaseCacheConfig): Configuration metadata for this cache.
            Shared across all views in the same cache hierarchy.
        layer_index (int | None): The index of the layer this view represents.
            None for cache types that don't have layer structure.

    Note:
        While marked as ABC, this class doesn't use @auto_pytree because
        concrete implementations need to control their PyTree structure.
    """

    metadata: BaseCacheConfig
    layer_index: int | None

    @classmethod
    @abstractmethod
    def init(cls, metadata: BaseCacheConfig, *args, **kwargs) -> BaseCacheView:
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
            metadata (BaseCacheConfig): Static configuration metadata that
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
            ...     config=metadata,
            ...     layer_index=0,
            ...     mesh=mesh,
            ...     dtype=jnp.bfloat16,
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
        metadata: BaseCacheConfig,
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
            metadata (BaseCacheConfig): Configuration metadata defining
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
