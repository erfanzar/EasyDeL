# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

The KV core (page pool, prefix cache, cache-group managers and coordinators) lives in surgecore's Rust crate;
:class:`CacheManager` is its binding. Cache specifications (:class:`FullAttentionSpec`, :class:`SlidingWindowSpec`,
:class:`ChunkedLocalAttentionSpec`) and :func:`create_kv_cache_specs_from_config` stay in Python.

The former Python KV-core classes (``PagePool``, the cache coordinators and the single-type managers) are kept as
names only: using one raises :class:`RuntimeError` that points to :class:`CacheManager`.
"""

from .interface import (
    AttentionSpec,
    CacheGroupsConfig,
    CacheGroupSpec,
    CacheSpec,
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    SlidingWindowSpec,
    create_kv_cache_specs_from_config,
)
from .manager import CacheManager, CachePages


def _removed(name: str) -> type:
    """A placeholder class for a KV-core name that moved into surgecore's Rust KV core."""

    def __new__(cls, *args, **kwargs):
        raise RuntimeError(
            f"easydel.inference.esurge.core.{name} was removed: the KV core is now surgecore's Rust KvCore. "
            "Use easydel.inference.esurge.core.CacheManager (same methods as before) instead."
        )

    return type(name, (), {"__new__": __new__, "__doc__": f"Removed; use :class:`CacheManager`. ({name})"})


CacheCoordinator = _removed("CacheCoordinator")
CacheCoordinatorNoPrefixCache = _removed("CacheCoordinatorNoPrefixCache")
UnitaryCacheCoordinator = _removed("UnitaryCacheCoordinator")
HybridCacheCoordinator = _removed("HybridCacheCoordinator")
PagePool = _removed("PagePool")
SingleTypeCacheManager = _removed("SingleTypeCacheManager")
FullAttentionManager = _removed("FullAttentionManager")
SlidingWindowManager = _removed("SlidingWindowManager")
ChunkedLocalAttentionManager = _removed("ChunkedLocalAttentionManager")

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
    "PagePool",
    "SingleTypeCacheManager",
    "SlidingWindowManager",
    "SlidingWindowSpec",
    "UnitaryCacheCoordinator",
    "create_kv_cache_specs_from_config",
)
