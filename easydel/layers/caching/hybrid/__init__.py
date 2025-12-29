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

"""Hybrid cache for models with mixed attention types.

This module provides caching for hybrid models that combine different
attention mechanisms (e.g., standard attention with linear attention).
Examples include Qwen3Next which alternates between full attention
layers (using KV cache) and linear attention layers (using recurrent state).

Key Components:
    - HybridCacheConfig: Configuration for hybrid cache dimensions
    - HybridCacheView: Per-layer cache supporting both KV and recurrent state
    - HybridCache: Multi-layer hybrid cache orchestration
    - HybridMetadata: Runtime metadata for hybrid cache operations

Features:
    - Support for both KV cache (full attention) and recurrent state (linear attention)
    - Per-layer attention type specification
    - Memory-efficient storage based on layer type
    - Functional cache updates for JAX compatibility

Example:
    >>> metadata = HybridCacheConfig.create(
    ...     num_hidden_layers=48,
    ...     partition_axis=partition_axis,
    ...     batch_size=2,
    ...     sequence_length=2048,
    ...     num_key_value_heads=8,
    ...     head_dim=128,
    ...     d_inner=2048,
    ...     d_conv=4,
    ...     d_state=64,
    ...     layer_types=("linear_attention", "linear_attention", "linear_attention", "full_attention", ...)
    ... )
    >>> cache = HybridCache.init_cache(
    ...     metadata=metadata,
    ...     dtype=jnp.float32
    ... )
"""

from .cache import (
    FULL_ATTENTION,
    KDA_LINEAR_ATTENTION,
    LINEAR_ATTENTION,
    PARALLEL_HYBRID,
    HybridCache,
    HybridCacheConfig,
    HybridCacheView,
    HybridMetadata,
    ParallelHybridCacheView,
)

__all__ = (
    "FULL_ATTENTION",
    "KDA_LINEAR_ATTENTION",
    "LINEAR_ATTENTION",
    "PARALLEL_HYBRID",
    "HybridCache",
    "HybridCacheConfig",
    "HybridCacheView",
    "HybridMetadata",
    "ParallelHybridCacheView",
)
