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

"""Caching systems for efficient inference.

Provides various caching mechanisms for different model architectures
to optimize memory usage and computation during inference.

Cache Types:
    TransformerCache: Standard KV-cache for transformer models
    RaggedPagesCache: Paged cache for efficient memory management
    RecurrentCache/LinearCache: Unified cache for state-space and linear attention models
        (Mamba, Mamba2, GatedDeltaNet, RWKV, RetNet)
    HybridCache: Cache for hybrid models mixing attention types (e.g., Qwen3Next)
    LightningCache: Cache for Lightning attention
    KDACache: Cache for KDA (Key-Driven Attention) models

Cache Specifications:
    FullAttentionSpec: Specification for full attention caching
    SlidingWindowSpec: Specification for sliding window attention
    ChunkedLocalAttentionSpec: Specification for chunked local attention
    MambaSpec: Specification for Mamba model caching
    KVCacheSpec: Base specification for KV caches

Key Features:
    - Memory-efficient caching strategies
    - Support for various attention patterns
    - Page-based memory management
    - Sliding window and local attention support
    - Unified recurrent/linear cache for state-space models
    - Hybrid caching for mixed-attention architectures

Example:
    >>> from easydel.layers.caching import TransformerCache
    >>> # Create a transformer cache
    >>> cache = TransformerCache.init(
    ...     batch_size=2,
    ...     max_length=1024,
    ...     num_heads=16,
    ...     head_dim=64,
    ...     dtype=jnp.bfloat16
    ... )
    >>> # Update cache with new key-value pairs
    >>> cache = cache.update(keys, values, positions)

    >>> from easydel.layers.caching import RecurrentCache, RecurrentCacheConfig
    >>> # Create a recurrent cache for Mamba-style models
    >>> metadata = RecurrentCacheConfig.create_for_mamba(
    ...     num_hidden_layers=24,
    ...     partition_axis=partition_axis,
    ...     batch_size=2,
    ...     intermediate_size=2048,
    ...     ssm_state_size=16,
    ...     conv_kernel_size=4
    ... )
    >>> cache = RecurrentCache.init_cache(config=metadata, dtype=jnp.float32)

Note:
    Different cache types are optimized for specific model
    architectures and attention patterns. Choose the appropriate
    cache based on your model's requirements.
"""

from ._abstracts import OperationsMetadata, unwrap_metadata
from ._metadatabuilder import AttentionMetadataBuilder
from ._specs import ChunkedLocalAttentionSpec, FullAttentionSpec, KVCacheSpec, MambaSpec, SlidingWindowSpec
from .hybrid import (
    HybridCache,
    HybridCacheConfig,
    HybridCacheView,
    HybridMetadata,
    ParallelHybridCacheView,
)
from .kda import KDACache, KDACacheConfig, KDACacheView, KDAMetadata
from .lightning import LightningCache, LightningCacheConfig, LightningCacheView, LightningMetadata
from .ragged_page import RaggedPagesCache, RaggedPagesCacheConfig, RaggedPagesCacheView, RaggedPagesMetadata
from .recurrent import (
    LinearCache,
    LinearCacheConfig,
    LinearCacheView,
    LinearMetadata,
    RecurrentCache,
    RecurrentCacheConfig,
    RecurrentCacheView,
    RecurrentMetadata,
)
from .transformer import TransformerCache, TransformerCacheConfig, TransformerCacheView, TransformerMetadata

__all__ = (
    "AttentionMetadataBuilder",
    "ChunkedLocalAttentionSpec",
    "FullAttentionSpec",
    "HybridCache",
    "HybridCacheConfig",
    "HybridCacheView",
    "HybridMetadata",
    "KDACache",
    "KDACacheConfig",
    "KDACacheView",
    "KDAMetadata",
    "KVCacheSpec",
    "LightningCache",
    "LightningCacheConfig",
    "LightningCacheView",
    "LightningMetadata",
    "LinearCache",
    "LinearCacheConfig",
    "LinearCacheView",
    "LinearMetadata",
    "MambaSpec",
    "OperationsMetadata",
    "ParallelHybridCacheView",
    "RaggedPagesCache",
    "RaggedPagesCacheConfig",
    "RaggedPagesCacheView",
    "RaggedPagesMetadata",
    "RecurrentCache",
    "RecurrentCacheConfig",
    "RecurrentCacheView",
    "RecurrentMetadata",
    "SlidingWindowSpec",
    "TransformerCache",
    "TransformerCacheConfig",
    "TransformerCacheView",
    "TransformerMetadata",
    "unwrap_metadata",
)
