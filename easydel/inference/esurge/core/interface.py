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

"""Cache specification interfaces for KV-cache configuration.

This module defines the specification classes that describe how KV-cache
should be organized for different attention patterns. These specifications
are used by the cache manager to determine page sizes, memory requirements,
and caching behavior.

Classes:
    CacheSpec: Base class for all cache specifications.
    AttentionSpec: Base class for attention-based cache specifications.
    FullAttentionSpec: Specification for full/causal attention.
    SlidingWindowSpec: Specification for sliding window attention.
    ChunkedLocalAttentionSpec: Specification for chunked local attention.
    MambaSpec: Specification for Mamba state-space model caching.
    CacheGroupSpec: Groups layers sharing the same cache page table.
    CacheGroupsConfig: Complete KV-cache configuration for a model.

Functions:
    create_kv_cache_specs_from_config: Create cache specs from model config.

Example:
    >>> spec = FullAttentionSpec(
    ...     page_size=16,
    ...     num_kv_heads=8,
    ...     head_size=64,
    ...     dtype=jnp.float16,
    ...     use_mla=False
    ... )
    >>> print(spec.page_size_bytes)  # Bytes per page
"""

import copy
from collections import defaultdict
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Self

from jax import numpy as jnp

from ..utils import cdiv, get_dtype_size

if TYPE_CHECKING:
    from easydel.infra.base_config import EasyDeLBaseConfig


@dataclass
class CacheSpec:
    """Base class for specifying KV-cache format for a layer type.

    This abstract base class defines the interface for cache specifications
    that describe how KV-cache should be organized for different layer types.
    Subclasses implement specific behavior for different attention patterns
    or model architectures.

    Attributes:
        page_size: Number of tokens stored in each cache page.

    Note:
        Subclasses must implement type_id, page_size_bytes, and
        max_memory_usage_bytes properties/methods.
    """

    page_size: int

    @property
    def type_id(self) -> str:
        """Get the unique type identifier for this cache specification.

        Returns a string that uniquely identifies layers with the same
        cache characteristics. Layers with different type_ids use
        separate cache groups.

        Returns:
            String identifier encoding cache type and parameters.

        Raises:
            NotImplementedError: Must be implemented by subclasses.

        Note:
            Different type_ids are assigned for:
            - Different attention patterns (full vs sliding window)
            - Different KV cache sizes per token (different head counts)
            - Different page sizes
        """
        raise NotImplementedError

    @property
    def page_size_bytes(self) -> int:
        """Calculate the memory size of one cache page in bytes.

        Returns:
            Number of bytes required to store one page of KV cache.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def max_memory_usage_bytes(self, *args, **kwargs) -> int:
        """Calculate maximum possible memory usage for this cache.

        Returns:
            Maximum KV-cache memory in bytes for worst-case sequence length.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """Merge multiple cache specifications into one.

        Combines a list of CacheSpec objects from different layers that
        share the same type_id into a single unified specification.

        Args:
            specs: List of CacheSpec objects to merge.

        Returns:
            A merged CacheSpec representing all input specifications.

        Raises:
            AssertionError: If specs have different type_ids.
        """
        assert all(spec.type_id == specs[0].type_id for spec in specs[1:]), (
            "All layers in the same KV cache group must share the same type_id."
        )
        return copy.deepcopy(specs[0])


@dataclass
class AttentionSpec(CacheSpec):
    """Base specification for attention-based KV-cache.

    Extends CacheSpec with attention-specific parameters including
    head configuration and data types. Provides the foundation for
    full attention, sliding window, and chunked local attention specs.

    Attributes:
        num_kv_heads: Number of key-value attention heads.
        head_size: Dimension of each attention head.
        dtype: JAX dtype for cache tensors (e.g., jnp.float16, jnp.bfloat16).
        use_mla: Whether Multi-head Latent Attention is used (halves K/V storage).
    """

    num_kv_heads: int
    head_size: int
    dtype: jnp.dtype
    use_mla: bool

    @property
    def page_size_bytes(self) -> int:
        """Calculate page size in bytes for attention KV-cache.

        Computes the memory required for one page, accounting for both
        key and value tensors (unless MLA is used, which stores only
        one combined tensor).

        Returns:
            Number of bytes per page: tokens * heads * head_dim * dtype_size * 2
            (or * 1 for MLA).
        """
        coef = 1 if self.use_mla else 2
        return coef * self.page_size * self.num_kv_heads * self.head_size * get_dtype_size(self.dtype)


@dataclass
class FullAttentionSpec(AttentionSpec):
    """Cache specification for full (causal) attention layers.

    Describes KV-cache requirements for layers using standard full attention,
    where each token attends to all previous tokens. Optionally records
    sliding window or chunk size for hybrid models where the KV cache
    manager treats sliding window layers as full attention.

    Attributes:
        sliding_window: Optional sliding window size when hybrid allocator
            is disabled but the layer uses sliding window computation.
        attention_chunk_size: Optional chunk size for chunked attention
            variants treated as full attention for caching purposes.

    Note:
        When hybrid allocator is disabled, sliding window and chunked
        attention layers may be treated as full attention for memory
        allocation while still using their respective attention patterns
        during computation.

    Example:
        >>> spec = FullAttentionSpec(
        ...     page_size=16,
        ...     num_kv_heads=8,
        ...     head_size=64,
        ...     dtype=jnp.float16,
        ...     use_mla=False
        ... )
    """

    sliding_window: int | None = None
    attention_chunk_size: int | None = None

    @property
    def type_id(self) -> str:
        """Get the type identifier for full attention.

        Returns:
            String in format 'full_attention_{page_size}_{page_size_bytes}'.
        """
        return f"full_attention_{self.page_size}_{self.page_size_bytes}"

    def max_memory_usage_bytes(self, max_model_len: int, **kwargs) -> int:
        """Calculate maximum memory for full attention KV-cache.

        For full attention, maximum memory is determined by the maximum
        sequence length, as all tokens are cached.

        Args:
            max_model_len: Maximum sequence length supported.
            **kwargs: Additional arguments (ignored).

        Returns:
            Maximum memory in bytes.
        """
        return cdiv(max_model_len, self.page_size) * self.page_size_bytes

    @classmethod
    def merge_window_sizes(cls, window_sizes: set[int]) -> int | None:
        """Merge window sizes from multiple specifications.

        Args:
            window_sizes: Set of window sizes to merge.

        Returns:
            The single window size if all are the same, None if empty.

        Raises:
            ValueError: If multiple different window sizes are provided.
        """
        if len(window_sizes) == 0:
            return None
        elif len(window_sizes) == 1:
            return window_sizes.pop()
        else:
            raise ValueError("All attention layers in the same KV cache group must have the same window size.")

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """Merge multiple FullAttentionSpec objects into one.

        Combines specifications while validating that window sizes and
        chunk sizes are consistent across all specifications.

        Args:
            specs: List of FullAttentionSpec objects to merge.

        Returns:
            Merged specification with unified window/chunk sizes.

        Raises:
            AssertionError: If both sliding window and chunk size are set.
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
    """Cache specification for chunked local attention layers.

    Describes KV-cache requirements for layers using chunked local attention,
    where tokens only attend within fixed-size chunks. This reduces memory
    usage compared to full attention while maintaining reasonable context.

    Attributes:
        attention_chunk_size: Size of each attention chunk in tokens.

    Example:
        >>> spec = ChunkedLocalAttentionSpec(
        ...     page_size=16,
        ...     num_kv_heads=8,
        ...     head_size=64,
        ...     dtype=jnp.float16,
        ...     use_mla=False,
        ...     attention_chunk_size=256
        ... )
    """

    attention_chunk_size: int

    @property
    def type_id(self) -> str:
        """Get the type identifier for chunked local attention.

        Returns:
            String including chunk size, page size, and page bytes.
        """
        return f"local_attention_{self.attention_chunk_size}_{self.page_size}_{self.page_size_bytes}"

    def max_memory_usage_bytes(self, max_model_len: int, max_num_batched_tokens: int, **kwargs) -> int:
        """Calculate maximum memory for chunked local attention.

        Memory is bounded by chunk size plus batch tokens, capped at
        max sequence length.

        Args:
            max_model_len: Maximum sequence length supported.
            max_num_batched_tokens: Maximum tokens per batch.
            **kwargs: Additional arguments (ignored).

        Returns:
            Maximum memory in bytes.
        """
        num_tokens = min(self.attention_chunk_size + max_num_batched_tokens, max_model_len)

        return cdiv(num_tokens, self.page_size) * self.page_size_bytes


@dataclass
class SlidingWindowSpec(AttentionSpec):
    """Cache specification for sliding window attention layers.

    Describes KV-cache requirements for layers using sliding window attention,
    where each token attends only to the most recent `sliding_window` tokens.
    This significantly reduces memory usage for long sequences.

    Attributes:
        sliding_window: Number of tokens in the attention window.

    Note:
        MLA (Multi-head Latent Attention) is not supported for sliding window.

    Example:
        >>> spec = SlidingWindowSpec(
        ...     page_size=16,
        ...     num_kv_heads=8,
        ...     head_size=64,
        ...     dtype=jnp.float16,
        ...     use_mla=False,
        ...     sliding_window=4096
        ... )
    """

    sliding_window: int

    def __post_init__(self) -> None:
        """Validate sliding window configuration.

        Raises:
            AssertionError: If use_mla is True.
        """
        assert not self.use_mla, "MLA is not supported for sliding window"

    @property
    def type_id(self) -> str:
        """Get the type identifier for sliding window attention.

        Returns:
            String including window size, page size, and page bytes.
        """
        return f"sliding_window_{self.sliding_window}_{self.page_size}_{self.page_size_bytes}"

    def max_memory_usage_bytes(self, max_model_len: int, max_num_batched_tokens: int, **kwargs) -> int:
        """Calculate maximum memory for sliding window attention.

        Memory is bounded by window size plus batch tokens, capped at
        max sequence length.

        Args:
            max_model_len: Maximum sequence length supported.
            max_num_batched_tokens: Maximum tokens per batch.
            **kwargs: Additional arguments (ignored).

        Returns:
            Maximum memory in bytes (includes one extra page for boundary).
        """
        num_tokens = min(self.sliding_window - 1 + max_num_batched_tokens, max_model_len)

        return (cdiv(num_tokens, self.page_size) + 1) * self.page_size_bytes


@dataclass
class MambaSpec(CacheSpec):
    """Cache specification for Mamba state-space model layers.

    Describes state caching requirements for Mamba layers, which use
    recurrent state-space models instead of attention. Mamba layers
    maintain fixed-size state tensors regardless of sequence length.

    Attributes:
        shapes: Tuple of shapes for each state tensor component.
        dtype: JAX dtype for state tensors.
        page_size_padded: Optional padded page size for alignment.

    Example:
        >>> spec = MambaSpec(
        ...     page_size=1,
        ...     shapes=((16, 64), (16, 32)),
        ...     dtype=jnp.float16
        ... )
    """

    shapes: tuple[tuple[int, ...], ...]
    dtype: jnp.dtype
    page_size_padded: int | None = None

    def __post_init__(self) -> None:
        """Calculate total number of elements across all state shapes."""
        self.num_elements = sum(prod(shape) for shape in self.shapes)

    @property
    def type_id(self) -> str:
        """Get the type identifier for Mamba state caching.

        Returns:
            String including shapes and dtype.
        """
        return f"mamba_{self.shapes}_{self.dtype}"

    @property
    def page_size_bytes(self) -> int:
        """Calculate page size in bytes for Mamba state.

        Returns padded size if specified, otherwise computes from
        total elements and dtype.

        Returns:
            Number of bytes per page.
        """
        page_size = self.num_elements * get_dtype_size(self.dtype)
        if self.page_size_padded is not None:
            assert self.page_size_padded >= page_size
            return self.page_size_padded
        return page_size

    def max_memory_usage_bytes(self, **kwargs) -> int:
        """Calculate maximum memory for Mamba state (constant).

        Mamba uses fixed-size state, so max memory equals page size.

        Args:
            **kwargs: Ignored (sequence length doesn't affect Mamba memory).

        Returns:
            Maximum memory in bytes (equals page_size_bytes).
        """
        return self.page_size_bytes


@dataclass
class CacheGroupSpec:
    """Specification for a group of layers sharing a KV-cache page table.

    Layers within a group share the same cache configuration and page
    table, allowing them to be managed as a single unit by the cache
    manager. Typically groups layers with identical attention patterns.

    Attributes:
        kv_cache_spec: The cache specification for this group.
        layer_names: Optional list of layer names in this group for debugging.

    Example:
        >>> group = CacheGroupSpec(
        ...     kv_cache_spec=FullAttentionSpec(...),
        ...     layer_names=['layer.0', 'layer.1', 'layer.2']
        ... )
    """

    kv_cache_spec: CacheSpec

    layer_names: list[str] | None = None


@dataclass
class CacheGroupsConfig:
    """Complete KV-cache configuration for a model.

    Contains all information needed to set up the cache manager,
    including the page pool size and all cache group specifications.

    Attributes:
        num_pages: Total number of pages to allocate in the page pool.
        kv_cache_groups: List of cache group specifications.

    Example:
        >>> config = CacheGroupsConfig(
        ...     num_pages=1000,
        ...     kv_cache_groups=[full_attn_group, sliding_window_group]
        ... )
    """

    num_pages: int
    kv_cache_groups: list[CacheGroupSpec]


def create_kv_cache_specs_from_config(
    config: "EasyDeLBaseConfig",
    page_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: jnp.dtype,
    use_mla: bool = False,
) -> list[CacheGroupSpec]:
    """Convert model config's get_mask_details() to CacheGroupSpec list.

    This function reads the attention mask details from the model configuration
    and creates appropriate cache specifications for each attention type.
    Layers with the same attention type are grouped together.

    Args:
        config: Model configuration with get_mask_details() method.
        page_size: Number of tokens per cache page.
        num_kv_heads: Number of key-value attention heads.
        head_size: Dimension of each attention head.
        dtype: Data type for cache tensors.
        use_mla: Whether to use Multi-head Latent Attention.

    Returns:
        List of CacheGroupSpec, one per attention type found in the config.
        Falls back to a single FullAttentionSpec if no mask details available.
    """
    from easydel.infra.utils import AttnMaskType

    mask_details = config.get_mask_details() if hasattr(config, "get_mask_details") else None

    if not mask_details:
        return [
            CacheGroupSpec(
                kv_cache_spec=FullAttentionSpec(
                    page_size=page_size,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    dtype=dtype,
                    use_mla=use_mla,
                ),
                layer_names=None,
            )
        ]

    groups: dict[AttnMaskType, list[tuple[int, int | None, int | None]]] = defaultdict(list)
    for layer_idx, detail in mask_details.items():
        groups[detail.mask_type].append((layer_idx, detail.size, detail.chunks))

    specs: list[CacheGroupSpec] = []

    for mask_type, layers in groups.items():
        layer_names = [f"layer.{idx}" for idx, _, _ in layers]

        if mask_type == AttnMaskType.FULL:
            specs.append(
                CacheGroupSpec(
                    kv_cache_spec=FullAttentionSpec(
                        page_size=page_size,
                        num_kv_heads=num_kv_heads,
                        head_size=head_size,
                        dtype=dtype,
                        use_mla=use_mla,
                    ),
                    layer_names=layer_names,
                )
            )
        elif mask_type == AttnMaskType.SLIDING:
            sliding_window = layers[0][1]
            if sliding_window is None:
                raise ValueError(f"Sliding window size is required for sliding attention layers: {layer_names}")
            specs.append(
                CacheGroupSpec(
                    kv_cache_spec=SlidingWindowSpec(
                        page_size=page_size,
                        num_kv_heads=num_kv_heads,
                        head_size=head_size,
                        dtype=dtype,
                        use_mla=False,  # MLA is not supported for sliding window
                        sliding_window=sliding_window,
                    ),
                    layer_names=layer_names,
                )
            )
        elif mask_type == AttnMaskType.CHUNK:
            chunk_size = layers[0][2]
            if chunk_size is None:
                raise ValueError(f"Chunk size is required for chunked attention layers: {layer_names}")
            specs.append(
                CacheGroupSpec(
                    kv_cache_spec=ChunkedLocalAttentionSpec(
                        page_size=page_size,
                        num_kv_heads=num_kv_heads,
                        head_size=head_size,
                        dtype=dtype,
                        use_mla=use_mla,
                        attention_chunk_size=chunk_size,
                    ),
                    layer_names=layer_names,
                )
            )
        else:
            raise ValueError(f"Unknown attention mask type: {mask_type}")

    return (
        specs
        if specs
        else [
            CacheGroupSpec(
                kv_cache_spec=FullAttentionSpec(
                    page_size=page_size,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    dtype=dtype,
                    use_mla=use_mla,
                ),
                layer_names=None,
            )
        ]
    )
