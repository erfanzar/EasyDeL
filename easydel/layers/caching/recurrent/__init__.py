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

"""Unified recurrent/linear cache for state-space and linear attention models.

This module provides a unified caching interface for models that use recurrent
state (conv_state + ssm_state) rather than KV-cache. This includes:
- Mamba (SSM with selective state spaces)
- Mamba2 (Multi-head SSM)
- GatedDeltaNet (Linear attention with gated delta rule)
- RWKV (Linear attention with time-mixing)
- RetNet (Retentive networks)

The RecurrentCache unifies MambaCache and Mamba2Cache into a single flexible
implementation that can handle different state shapes and organizations.

Key Components:
    - RecurrentCacheConfig: Flexible configuration for cache dimensions
    - RecurrentCacheView: Per-layer state storage for conv and recurrent state
    - RecurrentCache: Multi-layer cache orchestration
    - RecurrentMetadata: Runtime metadata for cache operations

Features:
    - Unified interface for both single-head and multi-head SSM states
    - Support for arbitrary recurrent state shapes
    - Rolling buffer for convolutional states
    - Functional cache updates for JAX compatibility
    - Backward compatible with MambaCache and Mamba2Cache APIs

Example:
    >>> # For Mamba-style models
    >>> metadata = RecurrentCacheConfig.create(
    ...     num_hidden_layers=24,
    ...     partition_axis=partition_axis,
    ...     batch_size=2,
    ...     conv_dim=2048,
    ...     conv_kernel_size=4,
    ...     recurrent_state_shape=(2048, 16),  # [intermediate_size, state_size]
    ... )
    >>> cache = RecurrentCache.init_cache(config=metadata, dtype=jnp.float32)

    >>> # For Mamba2-style models (multi-head)
    >>> metadata = RecurrentCacheConfig.create(
    ...     num_hidden_layers=32,
    ...     partition_axis=partition_axis,
    ...     batch_size=2,
    ...     conv_dim=2816,
    ...     conv_kernel_size=4,
    ...     recurrent_state_shape=(16, 64, 128),  # [num_heads, head_dim, state_size]
    ... )
    >>> cache = RecurrentCache.init_cache(config=metadata, dtype=jnp.float32)
"""

from .cache import (
    LinearCache,
    LinearCacheConfig,
    LinearCacheView,
    LinearMetadata,
    RecurrentCache,
    RecurrentCacheConfig,
    RecurrentCacheView,
    RecurrentMetadata,
)

__all__ = (
    "LinearCache",
    "LinearCacheConfig",
    "LinearCacheView",
    "LinearMetadata",
    "RecurrentCache",
    "RecurrentCacheConfig",
    "RecurrentCacheView",
    "RecurrentMetadata",
)
