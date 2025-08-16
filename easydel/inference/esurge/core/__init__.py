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

"""Core cache management components for the eSurge engine.

This module provides the core infrastructure for managing KV-cache and
attention mechanisms in the eSurge engine. It includes cache coordinators,
page pools, and various attention pattern managers.

Classes:
    CacheCoordinator: Main cache coordination
    CacheManager: Cache allocation and management
    PagePool: Pool of available cache pages
    FullAttentionManager: Standard attention cache
    SlidingWindowManager: Sliding window attention cache
    ChunkedLocalAttentionManager: Chunked local attention
    MambaManager: Mamba model cache management

Specifications:
    AttentionSpec: Base attention specification
    FullAttentionSpec: Full attention pattern
    SlidingWindowSpec: Sliding window pattern
    ChunkedLocalAttentionSpec: Chunked local pattern
    MambaSpec: Mamba model specification

Example:
    >>> from easydel.inference.esurge.core import CacheCoordinator, CacheGroupsConfig
    >>>
    >>> config = CacheGroupsConfig(groups=[...])
    >>> coordinator = CacheCoordinator(
    ...     config=config,
    ...     max_num_reqs=32,
    ...     page_pool=page_pool
    ... )
    >>> pages = coordinator.allocate(num_tokens=100)
"""

from .coordinator import CacheCoordinator, CacheCoordinatorNoPrefixCache, HybridCacheCoordinator, UnitaryCacheCoordinator
from .interface import (
    AttentionSpec,
    CacheGroupsConfig,
    CacheGroupSpec,
    CacheSpec,
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from .manager import CacheManager, CachePages
from .page_pool import PagePool
from .single_type_cache_manager import (
    ChunkedLocalAttentionManager,
    FullAttentionManager,
    MambaManager,
    SingleTypeCacheManager,
    SlidingWindowManager,
)

__all__ = (
    "AttentionSpec",
    "CacheCoordinator",
    "CacheCoordinatorNoPrefixCache",
    "CacheGroupSpec",
    "CacheGroupsConfig",
    "CacheManager",
    "CachePages",
    "CacheSpec",
    "ChunkedLocalAttentionManager",
    "ChunkedLocalAttentionSpec",
    "FullAttentionManager",
    "FullAttentionSpec",
    "HybridCacheCoordinator",
    "MambaManager",
    "MambaSpec",
    "PagePool",
    "SingleTypeCacheManager",
    "SlidingWindowManager",
    "SlidingWindowSpec",
    "UnitaryCacheCoordinator",
)
