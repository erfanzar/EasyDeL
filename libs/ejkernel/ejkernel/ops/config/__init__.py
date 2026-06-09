# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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


"""Configuration management and caching for ejkernel.ops.

This module provides a comprehensive configuration selection and caching system
with multiple fallback layers and automatic performance tuning capabilities.

Classes:
    ConfigCache: In-memory configuration cache keyed by (device, op_id, call_key).
    PersistentCache: Disk-based configuration persistence using JSON files.
    ConfigSelectorChain: Multi-tier configuration selection with a fallback chain
        (override → overlay → memory cache → persistent cache → autotune → heuristics).
    AutotunePolicy: Dataclass controlling autotuning behavior (allow_autotune,
        allow_heuristics, cache_miss_fallback, validate_backward).
    Tuner: Performance benchmarking engine used by ConfigSelectorChain to rank
        candidate configurations.
    overlay_cache: Context manager for temporary in-process cache overrides.
    policy_override: Context manager for temporary AutotunePolicy changes.
    forward_autotune_only: Context manager that suppresses backward-pass timing
        during autotuning even when AutotunePolicy.validate_backward is True.
    log_autotune_progress: Context manager that enables tqdm progress bars during
        autotuning (also controllable via EJKERNEL_AUTOTUNE_PROGRESS=1).
    set_autotune_progress: Imperative toggle for autotune progress bars.
"""

from .cache import ConfigCache, overlay_cache
from .persistent import PersistentCache
from .selection import (
    AutotunePolicy,
    ConfigSelectorChain,
    Tuner,
    forward_autotune_only,
    log_autotune_progress,
    policy_override,
    set_autotune_progress,
)

__all__ = (
    "AutotunePolicy",
    "ConfigCache",
    "ConfigSelectorChain",
    "PersistentCache",
    "Tuner",
    "forward_autotune_only",
    "log_autotune_progress",
    "overlay_cache",
    "policy_override",
    "set_autotune_progress",
)
