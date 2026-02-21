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

"""Specification classes for different caching strategies in EasyDeL.

This module defines specification dataclasses that describe the memory layout,
size requirements, and behavior of different cache types. These specifications
are used to:
- Calculate memory requirements before allocation
- Configure cache initialization parameters
- Optimize memory layout for specific attention patterns
- Enable hybrid caching strategies

The specifications follow a hierarchy:
- KVCacheSpec: Base specification for all KV cache types
- AttentionSpec: Base for attention-based caches
- FullAttentionSpec: Standard full attention caching
- SlidingWindowSpec: Sliding window attention caching
- ChunkedLocalAttentionSpec: Chunked local attention
- MambaSpec: State-space model caching

Key Concepts:
    - Page Size: Number of tokens per memory page
    - Type ID: Unique identifier for cache compatibility
    - Memory Budget: Maximum memory usage calculations
    - Hybrid Allocation: Mixing different cache types

Example:
    >>> spec = FullAttentionSpec(
    ...     page_size=128,
    ...     num_kv_heads=8,
    ...     head_size=64,
    ...     dtype=jnp.bfloat16,
    ...     use_mla=False
    ... )
    >>> memory_bytes = spec.max_memory_usage_bytes(max_model_len=2048)
"""

import copy
from dataclasses import dataclass
from math import prod
from typing import Self

import jax


def cdiv(a: int, b: int) -> int:
    """Ceiling division: divide a by b and round up.

    Computes the ceiling of a/b using integer arithmetic to avoid
    floating point operations. This is commonly used for calculating
    the number of pages needed for a given number of tokens.

    Args:
        a (int): Numerator (e.g., number of tokens)
        b (int): Denominator (e.g., page size)

    Returns:
        int: The ceiling of a/b

    Example:
        >>> cdiv(10, 3)  # 10 tokens, 3 per page
        4  # Need 4 pages
        >>> cdiv(9, 3)   # 9 tokens, 3 per page
        3  # Need 3 pages
    """
    return (a + b - 1) // b


@dataclass
class KVCacheSpec:
    """Base specification for key-value cache formats.

    This abstract base class defines the interface that all cache
    specifications must implement. It provides methods for calculating
    memory requirements and identifying cache types for compatibility.

    The specification pattern allows:
    - Pre-allocation memory budgeting
    - Cache type compatibility checking
    - Hybrid cache configuration
    - Memory optimization strategies

    Attributes:
        page_size (int): Number of tokens stored per cache page.
            Pages are the basic unit of cache allocation and help
            reduce memory fragmentation.

    Abstract Properties:
        type_id: Unique identifier for this cache type
        page_size_bytes: Size of one page in bytes

    Abstract Methods:
        max_memory_usage_bytes: Calculate maximum memory needed
        merge: Combine multiple specs of the same type
    """

    page_size: int

    @property
    def type_id(self) -> str:
        """Unique identifier for this cache specification type.

        The type ID is used to determine cache compatibility when mixing
        different cache types in a model. Caches with the same type_id
        can share memory pools and be managed together.

        Different type IDs should be returned for:
        - Different attention patterns (full vs sliding window)
        - Different cache sizes per token (varying head counts)
        - Different memory layouts (paged vs continuous)

        The ID typically encodes:
        - Cache strategy name
        - Key configuration parameters
        - Memory layout information

        Returns:
            str: A unique string identifier for this cache type.
                Format typically: "{strategy}_{params}_{size}"

        Example:
            "full_attention_128_16384" for full attention with
            page_size=128 and page_size_bytes=16384
        """
        raise NotImplementedError

    @property
    def page_size_bytes(self) -> int:
        """Calculate the memory size of a single cache page in bytes.

        This property computes the total memory required to store
        `page_size` tokens worth of cache data, accounting for:
        - Number of heads (key and value)
        - Head dimensions
        - Data type size
        - Any padding or alignment requirements

        The calculation typically follows:
        bytes = page_size * num_heads * head_dim * dtype_bytes * 2
        (where 2 accounts for both keys and values)

        Returns:
            int: Size of one cache page in bytes.

        Note:
            Implementations may include padding for memory alignment
            or hardware-specific optimizations.
        """
        raise NotImplementedError

    def max_memory_usage_bytes(self, *args, **kwargs) -> int:
        """Calculate maximum memory required for this cache configuration.

        Computes the worst-case memory usage for the cache based on
        the maximum sequence length and other parameters. This is used
        for memory budgeting and allocation planning.

        Args:
            *args: Implementation-specific arguments.
            **kwargs: Implementation-specific keyword arguments.
                Common kwargs include:
                - max_model_len: Maximum sequence length
                - max_num_batched_tokens: Max tokens per batch
                - max_num_reqs: Maximum concurrent requests

        Returns:
            int: Maximum memory usage in bytes.

        Note:
            Different cache types calculate this differently:
            - Full attention: O(max_length)
            - Sliding window: O(window_size)
            - Chunked: O(chunk_size + batch_size)
        """
        raise NotImplementedError

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """Merge multiple cache specifications into a single specification.

        Combines specifications from multiple layers that share the same
        cache type. This is used when multiple layers can share a cache
        pool for memory efficiency.

        The merge process:
        1. Validates all specs have compatible type_ids
        2. Combines configuration parameters
        3. Returns a unified specification

        Args:
            specs (list[Self]): List of specifications to merge.
                All must have the same type_id.

        Returns:
            Self: A merged specification representing the combined
                requirements of all input specifications.

        Raises:
            AssertionError: If specs have incompatible type_ids.

        Note:
            The default implementation returns a copy of the first spec.
            Subclasses may override to merge specific parameters.
        """
        assert all(spec.type_id == specs[0].type_id for spec in specs[1:]), (
            "All layers in the same KV cache group must share the same type_id."
        )
        return copy.deepcopy(specs[0])


@dataclass
class AttentionSpec(KVCacheSpec):
    """Base specification for attention-based cache formats.

    Extends KVCacheSpec with attention-specific parameters needed
    for transformer-based models. This includes head configuration,
    data types, and optimization flags.

    Attributes:
        num_kv_heads (int): Number of key-value attention heads.
            May differ from query heads in multi-query/grouped-query attention.
        head_size (int): Dimension of each attention head.
        dtype (jax.numpy.dtype): Data type for cache tensors.
            Common choices: bfloat16, float16, float32.
        use_mla (bool): Whether to use Multi-Level Attention optimization.
            MLA can reduce memory usage by sharing representations.
    """

    num_kv_heads: int
    head_size: int
    dtype: jax.numpy.dtype
    use_mla: bool

    @property
    def page_size_bytes(self) -> int:
        """Calculate page size for attention cache in bytes.

        Computes memory needed for one page of key-value pairs:
        - Without MLA: stores both keys and values (coef=2)
        - With MLA: stores combined representation (coef=1)

        Formula:
            bytes = coef * page_size * num_kv_heads * head_size * dtype_bytes

        Returns:
            int: Size of one attention cache page in bytes.
        """
        coef = 1 if self.use_mla else 2
        return coef * self.page_size * self.num_kv_heads * self.head_size * (jax.numpy.finfo(self.dtype).bits // 8)


@dataclass
class FullAttentionSpec(AttentionSpec):
    """Specification for full attention caching.

    Represents standard transformer attention where each token can
    attend to all previous tokens. This is the most common and
    memory-intensive cache type.

    When hybrid allocation is disabled, this spec can also represent
    sliding window or chunked attention layers by storing the window/chunk
    parameters while allocating full cache space. This simplifies memory
    management at the cost of over-allocation.

    Attributes:
        sliding_window (int | None): Optional sliding window size.
            When set, attention computation uses sliding window but
            cache allocation remains full-sized. None for standard full attention.
        attention_chunk_size (int | None): Optional chunk size for
            chunked attention. Similar to sliding_window but for
            chunked patterns. None for standard full attention.

    Note:
        Only one of sliding_window or attention_chunk_size should be set.
        Both being non-None is an error.
    """

    sliding_window: int | None = None
    attention_chunk_size: int | None = None

    @property
    def type_id(self) -> str:
        return f"full_attention_{self.page_size}_{self.page_size_bytes}"

    def max_memory_usage_bytes(self, max_model_len: int, **kwargs) -> int:
        """Calculate maximum memory for full attention cache.

        Memory scales linearly with maximum sequence length since
        all tokens need to be cached.

        Args:
            max_model_len (int): Maximum sequence length supported.
            **kwargs: Additional arguments (unused).

        Returns:
            int: Maximum memory in bytes.
                 Formula: ceil(max_model_len / page_size) * page_size_bytes
        """
        return cdiv(max_model_len, self.page_size) * self.page_size_bytes

    @classmethod
    def merge_window_sizes(cls, window_sizes: set[int]) -> int | None:
        """Merge sliding window sizes from multiple layers.

        Ensures all layers in a cache group use the same window size
        for consistent memory allocation.

        Args:
            window_sizes (set[int]): Set of window sizes from different layers.

        Returns:
            int | None: The single window size if consistent, None if no windows.

        Raises:
            ValueError: If layers have different window sizes.
        """
        if len(window_sizes) == 0:
            return None
        elif len(window_sizes) == 1:
            return window_sizes.pop()
        else:
            raise ValueError("All attention layers in the same KV cache group must have the same window size.")

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of FullAttentionSpec objects into a single
        FullAttentionSpec object.
        """
        merged_spec = super().merge(specs)
        sliding_window = set(spec.sliding_window for spec in specs if spec.sliding_window is not None)
        attention_chunk_size = set(spec.attention_chunk_size for spec in specs if spec.attention_chunk_size is not None)

        merged_spec.sliding_window = cls.merge_window_sizes(sliding_window)
        merged_spec.attention_chunk_size = cls.merge_window_sizes(attention_chunk_size)
        assert (merged_spec.sliding_window is not None) + (merged_spec.attention_chunk_size is not None) <= 1, (
            "Model with both sliding window layers and chunked local attention layers is not supported."
        )
        return merged_spec


@dataclass
class ChunkedLocalAttentionSpec(AttentionSpec):
    """Specification for chunked local attention caching.

    Optimizes memory usage for models that use local attention patterns
    where tokens only attend within fixed-size chunks. This significantly
    reduces memory requirements compared to full attention.

    Memory allocation is based on chunk size rather than full sequence
    length, making it suitable for very long sequences.

    Attributes:
        attention_chunk_size (int): Size of attention chunks.
            Tokens can only attend within their chunk boundaries.
    """

    attention_chunk_size: int

    @property
    def type_id(self) -> str:
        return f"local_attention_{self.attention_chunk_size}_{self.page_size}_{self.page_size_bytes}"

    def max_memory_usage_bytes(
        self,
        max_model_len: int,
        max_num_batched_tokens: int,
        **kwargs,
    ) -> int:
        """Calculate maximum memory for chunked attention cache.

        Memory is bounded by chunk size plus current batch size,
        not the full sequence length.

        Args:
            max_model_len (int): Maximum sequence length (upper bound).
            max_num_batched_tokens (int): Maximum tokens processed per batch.
            **kwargs: Additional arguments (unused).

        Returns:
            int: Maximum memory in bytes.
                 Based on min(chunk_size + batch_tokens, max_model_len).
        """
        num_tokens = min(self.attention_chunk_size + max_num_batched_tokens, max_model_len)
        return cdiv(num_tokens, self.page_size) * self.page_size_bytes


@dataclass
class SlidingWindowSpec(AttentionSpec):
    """Specification for sliding window attention caching.

    Implements a fixed-size sliding window where tokens can only
    attend to a limited number of previous tokens. This provides
    a good balance between memory efficiency and model capability.

    The cache maintains a rolling buffer of the most recent tokens,
    discarding older tokens beyond the window size.

    Attributes:
        sliding_window (int): Size of the sliding attention window.
            Each token attends to at most this many previous tokens.

    Constraints:
        - MLA optimization is not compatible with sliding windows
    """

    sliding_window: int

    def __post_init__(self):
        """Validate sliding window configuration.

        Raises:
            AssertionError: If MLA is enabled (not supported).
        """
        assert not self.use_mla, "MLA is not supported for sliding window"

    @property
    def type_id(self) -> str:
        return f"sliding_window_{self.sliding_window}_{self.page_size}_{self.page_size_bytes}"

    def max_memory_usage_bytes(
        self,
        max_model_len: int,
        max_num_batched_tokens: int,
        **kwargs,
    ) -> int:
        """Calculate maximum memory for sliding window cache.

        Memory is bounded by window size plus current batch, with
        an extra page for boundary conditions.

        Args:
            max_model_len (int): Maximum sequence length (upper bound).
            max_num_batched_tokens (int): Maximum tokens processed per batch.
            **kwargs: Additional arguments (unused).

        Returns:
            int: Maximum memory in bytes.
                 Includes extra page for window boundary handling.
        """
        num_tokens = min(self.sliding_window - 1 + max_num_batched_tokens, max_model_len)
        return (cdiv(num_tokens, self.page_size) + 1) * self.page_size_bytes


@dataclass
class MambaSpec(KVCacheSpec):
    """Specification for Mamba state-space model caching.

    Mamba models use state-space representations instead of attention,
    requiring different cache structures for hidden states and
    convolutional states.

    The cache stores multiple state tensors with different shapes,
    all packed into a single page-based allocation.

    Attributes:
        shapes (tuple[tuple[int, ...], ...]): Shapes of state tensors
            to cache. Each inner tuple defines one state tensor's shape.
        dtype (jax.numpy.dtype): Data type for state tensors.
        page_size_padded (int | None): Optional padded page size for
            alignment. If set, pages are padded to this size.
        num_elements (int): Total number of elements across all shapes.
            Calculated automatically in __post_init__.
    """

    shapes: tuple[tuple[int, ...], ...]
    dtype: jax.numpy.dtype
    page_size_padded: int | None = None

    def __post_init__(self):
        """Calculate total elements from shapes.

        Sets num_elements to the sum of products of each shape.
        """
        self.num_elements = sum(prod(shape) for shape in self.shapes)

    @property
    def type_id(self) -> str:
        return f"mamba_{self.shapes}_{self.dtype}"

    @property
    def page_size_bytes(self) -> int:
        """Calculate page size for Mamba state cache in bytes.

        Computes the memory needed to store all state tensors,
        optionally with padding for alignment.

        Returns:
            int: Size of one state cache page in bytes.
                 Uses page_size_padded if specified, otherwise
                 exact size based on num_elements * dtype_size.

        Raises:
            AssertionError: If page_size_padded is less than required size.
        """
        page_size = self.num_elements * (jax.numpy.finfo(self.dtype).bits // 8)
        if self.page_size_padded is not None:
            assert self.page_size_padded >= page_size
            return self.page_size_padded
        return page_size

    def max_memory_usage_bytes(self, *args, **kwargs) -> int:
        """Calculate maximum memory for Mamba state cache.

        Mamba caches have fixed size per layer regardless of sequence
        length, as they maintain a constant-size state representation.

        Args:
            *args: Unused (for compatibility).
            **kwargs: Unused (for compatibility).

        Returns:
            int: Maximum memory in bytes (equals page_size_bytes).
        """
        return self.page_size_bytes
