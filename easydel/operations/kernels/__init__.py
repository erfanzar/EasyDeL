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

"""Attention and sequence modeling operation implementations.

This subpackage contains the concrete implementations of various attention
mechanisms and sequence modeling operations used in EasyDeL. Each implementation
is registered with the OperationRegistry and can be instantiated by name.

Attention Operations:
    BlockSparseAttn: Splash Attention for TPU using Pallas kernels.
        Optimized block-sparse attention for sequences divisible by 128.

    FlashAttn: Flash Attention implementation for GPU and TPU.
        Memory-efficient attention with O(N) memory complexity.

    RingAttn: Ring Attention for sequence parallelism.
        Distributes attention computation across multiple devices.

    ScaledDotProductAttn: Standard SDPA using JAX primitives.
        Leverages jax.nn.dot_product_attention with automatic backend selection.

    VanillaAttn: Reference attention implementation.
        Standard scaled dot-product attention with optional weights.

    AutoRegressiveDecodeAttn: Optimized single-token decoding attention.
        Specialized for autoregressive generation with KV-cache.

    UnifiedAttn: Unified attention for continuous batching.
        Combines prefill and decode in a single kernel for vLLM-style serving.

    RaggedPageAttnV2, RaggedPageAttnV3: Paged attention variants.
        Memory-efficient attention with page-based KV-cache management.

State Space Model Operations:
    SSM1Op: Mamba/S4 style selective state space layer.
        Linear-time sequence modeling with selective gating.

    SSM2Op: Mamba-2 style state space layer with SSD kernel.
        Improved state space model with structured state decay.

    GatedDeltaRuleOp: Gated Delta Rule linear attention.
        Recurrent linear attention for hybrid transformer architectures.

    KernelDeltaAttnOp: Kernel Delta Attention (KDA).
        Linear attention variant used in Kimi Linear models.

Example:
    >>> from easydel.operations import OperationRegistry, OperationMetadata
    >>> from easydel.operations.kernels import FlashAttn, VanillaAttn
    >>>
    >>> # Create attention by name through registry
    >>> metadata = OperationMetadata(runtime_dtype=jnp.float16)
    >>> attn = OperationRegistry.create("flash", metadata)
    >>>
    >>> # Or instantiate directly
    >>> flash_attn = FlashAttn(metadata)
    >>> vanilla_attn = VanillaAttn(metadata)
"""

from .blocksparse_attention import BlockSparseAttn
from .decode_attention import AutoRegressiveDecodeAttn
from .flash_attention import FlashAttn
from .gated_delta_rule import GatedDeltaRuleOp, GatedDeltaRuleOutput
from .glm_moe_dsa_indexer import GlmMoeDsaIndexerOp, GlmMoeDsaIndexerOutput
from .kda import KDAOutput, KernelDeltaAttnOp, fused_kda_gate
from .paged_flash_attention import PagedFlashAttn
from .ragged_page_attention import RaggedPageAttnV2, RaggedPageAttnV3
from .ring_attention import RingAttn
from .scaled_dot_product_attention import ScaledDotProductAttn
from .ssm1 import SSM1Op, SSM1Output
from .ssm2 import SSM2Op, SSM2Output
from .unified_attention import UnifiedAttn
from .vanilla_attention import VanillaAttn

__all__ = (
    "AutoRegressiveDecodeAttn",
    "BlockSparseAttn",
    "FlashAttn",
    "GatedDeltaRuleOp",
    "GatedDeltaRuleOutput",
    "GlmMoeDsaIndexerOp",
    "GlmMoeDsaIndexerOutput",
    "KDAOutput",
    "KernelDeltaAttnOp",
    "PagedFlashAttn",
    "RaggedPageAttnV2",
    "RaggedPageAttnV3",
    "RingAttn",
    "SSM1Op",
    "SSM1Output",
    "SSM2Op",
    "SSM2Output",
    "ScaledDotProductAttn",
    "UnifiedAttn",
    "VanillaAttn",
    "fused_kda_gate",
)
