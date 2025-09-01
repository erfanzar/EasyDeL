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

"""Low-level attention operator implementations.

Provides concrete implementations of various attention mechanisms
with hardware-specific optimizations. These operators are used by
the higher-level FlexibleAttentionModule.

Classes:
    AttentionImpl: Base class for attention implementations
    AttentionMetadata: Metadata for attention computation
    AttentionOutput: Output container for attention results
    AttentionRegistry: Registry for attention implementations

Implementations:
    AutoRegressiveDecodeAttn: Optimized for autoregressive decoding
    FlashAttn: FlashAttention implementation
    RaggedPageAttn: Paged attention for efficient inference
    RingAttn: Ring attention for sequence parallelism
    ScaledDotProductAttn: Standard scaled dot-product attention
    SplashAttn: Splash attention for TPUs
    VanillaAttn: Basic dot-product attention

Example:
    >>> from easydel.layers.attention_operator import FlashAttn
    >>> flash_attn = FlashAttn(
    ...     num_heads=16,
    ...     head_dim=64,
    ...     dtype=jnp.float16
    ... )
    >>> output = flash_attn.compute(
    ...     query, key, value,
    ...     mask=attention_mask
    ... )
"""

from ._attention_impl import AttentionImpl, AttentionMetadata, AttentionOutput, AttentionRegistry
from .modules import (
    AutoRegressiveDecodeAttn,
    FlashAttn,
    RaggedPageAttn,
    RingAttn,
    ScaledDotProductAttn,
    SplashAttn,
    VanillaAttn,
)

__all__ = (
    "AttentionImpl",
    "AttentionMetadata",
    "AttentionOutput",
    "AttentionRegistry",
    "AutoRegressiveDecodeAttn",
    "FlashAttn",
    "RaggedPageAttn",
    "RingAttn",
    "ScaledDotProductAttn",
    "SplashAttn",
    "VanillaAttn",
)
