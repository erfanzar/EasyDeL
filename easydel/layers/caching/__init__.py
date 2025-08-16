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
    PagesCache: Paged cache for efficient memory management
    MambaCache: Cache for Mamba state-space models
    Mamba2Cache: Enhanced cache for Mamba2 models
    LightningCache: Cache for Lightning attention

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
    - State-space model caching

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

Note:
    Different cache types are optimized for specific model
    architectures and attention patterns. Choose the appropriate
    cache based on your model's requirements.
"""

from ._specs import ChunkedLocalAttentionSpec, FullAttentionSpec, KVCacheSpec, MambaSpec, SlidingWindowSpec
from .lightning import LightningCache, LightningCacheMetaData, LightningCacheView, LightningMetadata
from .mamba import MambaCache, MambaCacheMetaData, MambaCacheView, MambaMetadata
from .mamba2 import Mamba2Cache, Mamba2CacheMetaData, Mamba2CacheView, Mamba2Metadata
from .page import PagesCache, PagesCacheMetaData, PagesCacheView, PagesMetadata
from .transformer import TransformerCache, TransformerCacheMetaData, TransformerCacheView, TransformerMetadata

__all__ = (
    "ChunkedLocalAttentionSpec",
    "FullAttentionSpec",
    "KVCacheSpec",
    "LightningCache",
    "LightningCacheMetaData",
    "LightningCacheView",
    "LightningMetadata",
    "Mamba2Cache",
    "Mamba2CacheMetaData",
    "Mamba2CacheView",
    "Mamba2Metadata",
    "MambaCache",
    "MambaCacheMetaData",
    "MambaCacheView",
    "MambaMetadata",
    "MambaSpec",
    "PagesCache",
    "PagesCacheMetaData",
    "PagesCacheView",
    "PagesMetadata",
    "SlidingWindowSpec",
    "TransformerCache",
    "TransformerCacheMetaData",
    "TransformerCacheView",
    "TransformerMetadata",
)
