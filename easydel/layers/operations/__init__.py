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
    OperationImpl: Base class for attention implementations
    OperationMetadata: Metadata for attention computation
    AttentionOutput: Output container for attention results
    OperationRegistry: Registry for attention implementations

Implementations:
    AutoRegressiveDecodeAttn: Optimized for autoregressive decoding
    FlashAttn: FlashAttention implementation
    RaggedPageAttnV3: Paged attention for efficient inference v3
    RaggedPageAttnV2: Paged attention for efficient inference v2
    RingAttn: Ring attention for sequence parallelism
    ScaledDotProductAttn: Standard scaled dot-product attention
    BlockSparseAttn: Splash attention for TPUs
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

from ejkernel.modules.operations.configs import (
    AttentionConfig,
    BaseOperationConfig,
    BlockSparseAttentionConfig,
    FlashAttentionConfig,
    RaggedPageAttentionv2Config,
    RaggedPageAttentionv3Config,
    RingAttentionConfig,
    ScaledDotProductAttentionConfig,
)

from ._attention_outputs import AttentionOutput
from ._operation_impl import OperationImpl, OperationMetadata, OperationRegistry
from .executor import OperationExecutor
from .modules import (
    AutoRegressiveDecodeAttn,
    BlockSparseAttn,
    FlashAttn,
    RaggedPageAttnV2,
    RaggedPageAttnV3,
    RingAttn,
    ScaledDotProductAttn,
    VanillaAttn,
)
from .requirements import (
    CacheRequirements,
    CacheType,
    ExecutionMode,
    MetadataField,
    MetadataRequirements,
    ModeSpecificBuilder,
    ModeSpecificRequirements,
    OperationRequirements,
    RequirementsBuilder,
    RequirementsValidator,
    ValidationResult,
)

__all__ = (
    "AttentionConfig",
    "AttentionOutput",
    "AutoRegressiveDecodeAttn",
    "BaseOperationConfig",
    "BlockSparseAttentionConfig",
    "BlockSparseAttn",
    "CacheRequirements",
    "CacheType",
    "ExecutionMode",
    "FlashAttentionConfig",
    "FlashAttn",
    "MetadataField",
    "MetadataRequirements",
    "ModeSpecificBuilder",
    "ModeSpecificRequirements",
    "OperationExecutor",
    "OperationImpl",
    "OperationMetadata",
    "OperationRegistry",
    "OperationRequirements",
    "RaggedPageAttentionv2Config",
    "RaggedPageAttentionv3Config",
    "RaggedPageAttnV2",
    "RaggedPageAttnV3",
    "RequirementsBuilder",
    "RequirementsValidator",
    "RingAttentionConfig",
    "RingAttn",
    "ScaledDotProductAttentionConfig",
    "ScaledDotProductAttn",
    "ValidationResult",
    "VanillaAttn",
)
