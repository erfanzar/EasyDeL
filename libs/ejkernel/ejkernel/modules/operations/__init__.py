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


"""Registered kernel operations: public API for ejkernel attention and matmul.

Each operation in this package is a thin dispatch layer that:

1. Wraps a ``Kernel`` subclass (e.g. ``FlashAttention``) responsible for
   ``run()``, ``heuristic_cfg()``, and ``candidate_cfgs()`` methods.
2. Exposes a module-level functional alias (e.g. ``flash_attention``) backed by
   a pre-configured ``Executor`` with config caching and optional autotuning.
3. Selects the fastest registered backend automatically using ``detect_platform``
   from ``ejkernel.modules.base`` — no manual platform specification required.

All operations accept an optional ``cfg`` keyword argument (a typed
``BaseOperationConfig`` subclass from ``configs.py``) to override block sizes,
platform, or backend selection. Pass ``platform=`` directly to the functional
interface as a shorthand for a one-off platform override.

Available Operations:
    Attention:
        - Attention / attention: Standard MHA (XLA); returns (output, weights).
        - FlashAttention / flash_attention: Memory-efficient O(N) attention.
        - ScaledDotProductAttention / scaled_dot_product_attention: XLA SDPA.
        - BlockSparseAttention / blocksparse_attention: Block-sparse with mask builder.
        - NativeSparseAttention / native_sparse_attention: Explicit block patterns.
        - RingAttention / ring_attention: Ring-topology distributed attention.
        - DeepSeekAttention / deepseek_attn: MLA + Lightning Indexer sparse attn.
        - FlashMLA / flash_mla: Flash multi-head latent (low-rank KV) attention.

    Paged / Serving Attention:
        - PageAttention / page_attention: Single-query paged KV decode.
        - DecodeAttention / decode_attention: vLLM paged decode returning output+LSE.
        - UnifiedAttention / unified_attention: vLLM unified prefill+decode attention.
        - PrefillPageAttention / prefill_page_attention: Chunked prefill with paged KV.
        - ChunkedPrefillPagedDecode / chunked_prefill_paged_decode: Fused KV write+decode.
        - RaggedDecodeAttention / ragged_decode_attention: Ragged sequence decode.
        - RaggedPageAttentionv2 / ragged_page_attention_v2: Ragged paged v2 (read-only).
        - RaggedPageAttentionv2TurboQuant / ragged_page_attention_v2_turboquant:
            TurboQuant compressed ragged paged v2.
        - RaggedPageAttentionv3 / ragged_page_attention_v3: Ragged paged v3 with
            chunked-prefill support.
        - RaggedPageAttentionv3TurboQuant / ragged_page_attention_v3_turboquant:
            TurboQuant compressed ragged paged v3.
        - MultiLatentRaggedPageAttention / multi_latent_ragged_page_attention: MLA+ragged.
        - MultiLatentRaggedPageAttentionV2 / multi_latent_ragged_page_attention_v2:
            MLA ragged v2 with per-phase block tuning.

    Linear / Recurrent Attention:
        - GLAttention / gla_attention: Gated linear attention (O(N)).
        - LightningAttention / lightning_attention: Lightning attention with decay.
        - KernelDeltaAttention / kernel_delta_attention / kda_attention: Delta-rule LA.
        - RecurrentAttention / recurrent_attention: Stateful recurrent attention.
        - GatedDeltaRule / gated_delta_rule / gdr_attention: Gated delta rule (O(N)).
        - RaggedGatedDeltaRule / ragged_gated_delta_rule: GDR for ragged sequences.

    Linear Recurrent Models:
        - RWKV4 / rwkv4: WKV recurrence with fixed time-decay.
        - RWKV6 / rwkv6: WKV recurrence with token-dependent decay.
        - RWKV7 / rwkv7: Enhanced RWKV-7 state-update recurrence.
        - RWKV7Mul / rwkv7_mul: RWKV-7 with multiplicative state updates.

    State Space Models:
        - StateSpaceV1 / state_space_v1: Mamba-1 style selective SSM.
        - StateSpaceV2 / state_space_v2: Mamba-2 style SSD.

    Matrix Operations:
        - GroupedMatmul / grouped_matmul: Variable-group-size matmul for MoE.
        - AllGatherMatmul / all_gather_matmul: Fused all-gather + matmul.
        - ReduceScatterMatmul / reduce_scatter_matmul: Fused matmul + reduce-scatter.
        - QuantizedMatmul / quantized_matmul: Packed uint32 symmetric quantized matmul.

    Miscellaneous:
        - MeanPooling / mean_pooling: Sequence mean pooling.

Configuration classes (imported from ``configs``):
    Each operation has a corresponding ``*Config`` dataclass documented in
    ``ejkernel.modules.operations.configs``.  All configs inherit ``platform``
    and ``backend`` from ``BaseOperationConfig``.

Aliases:
    ``gdr_attention`` is an alias for ``gated_delta_rule`` (module-level
    assignment at the bottom of this file).

Example:
    >>> from ejkernel.modules.operations import flash_attention, FlashAttentionConfig
    >>>
    >>> # Minimal causal attention (auto platform selection)
    >>> output = flash_attention(query, key, value, causal=True)
    >>>
    >>> # Explicit Triton platform with custom block sizes
    >>> from ejkernel.ops import FwdParams
    >>> cfg = FlashAttentionConfig(
    ...     fwd_params=FwdParams(q_blocksize=128, kv_blocksize=64),
    ...     platform="triton",
    ... )
    >>> output = flash_attention(query, key, value, causal=True, cfg=cfg)
    >>>
    >>> # MLA for memory-efficient inference (low-rank KV compression)
    >>> from ejkernel.modules.operations import flash_mla
    >>> output = flash_mla(query, key_value, w_kc, w_vc, causal=True)

Note:
    All functional interfaces are backed by persistent-cache-aware ``Executor``
    instances that amortise autotuning cost across repeated calls with the same
    tensor shapes.  Set the ``EJKERNEL_AUTOTUNE_POLICY`` environment variable
    to ``"heuristics"`` to skip autotuning and always use heuristic defaults.
"""

from .all_gather_matmul import AllGatherMatmul, all_gather_matmul
from .attention import Attention, attention
from .blocksparse_attention import BlockSparseAttention, blocksparse_attention
from .chunked_prefill_paged_decode import ChunkedPrefillPagedDecode, chunked_prefill_paged_decode
from .configs import (
    AllGatherMatmulConfig,
    AttentionConfig,
    BlockSparseAttentionConfig,
    ChunkedPrefillPagedDecodeConfig,
    DecodeAttentionConfig,
    DeepSeekAttentionConfig,
    FlashAttentionConfig,
    FlashMLAConfig,
    FusedCrossEntropyConfig,
    FusedKLDivergenceConfig,
    GatedDeltaRuleConfig,
    GLAttentionConfig,
    GroupedMatmulConfig,
    KernelDeltaAttentionConfig,
    LightningAttentionConfig,
    MeanPoolingConfig,
    MultiLatentRaggedPageAttentionConfig,
    MultiLatentRaggedPageAttentionV2Config,
    NativeSparseAttentionConfig,
    PageAttentionConfig,
    PrefillPageAttentionConfig,
    QuantizedMatmulConfig,
    RaggedDecodeAttentionConfig,
    RaggedGatedDeltaRuleConfig,
    RaggedPageAttentionv2Config,
    RaggedPageAttentionv2TurboQuantConfig,
    RaggedPageAttentionv3Config,
    RaggedPageAttentionv3TurboQuantConfig,
    RecurrentAttentionConfig,
    ReduceScatterMatmulConfig,
    RingAttentionConfig,
    RWKV4Config,
    RWKV6Config,
    RWKV7Config,
    RWKV7MulConfig,
    ScaledDotProductAttentionConfig,
    StateSpaceV1Config,
    StateSpaceV2Config,
    UnifiedAttentionConfig,
)
from .decode_attention import DecodeAttention, decode_attention
from .deepseek_attn import DeepSeekAttention, deepseek_attn
from .flash_attention import FlashAttention, flash_attention
from .fused_cross_entropy import CrossEntropyOutput, FusedCrossEntropy, fused_cross_entropy
from .fused_kl_divergence import FusedKLDivergence, KLDivergenceOutput, fused_kl_divergence
from .gated_delta_rule import GatedDeltaRule, gated_delta_rule
from .gated_linear_attention import GLAttention, gla_attention
from .grouped_matmul import GroupedMatmul, grouped_matmul
from .kernel_delta_attention import KernelDeltaAttention, kda_attention, kernel_delta_attention
from .lightning_attention import LightningAttention, lightning_attention
from .multi_head_latent_attention import FlashMLA, flash_mla
from .multi_latent_ragged_page_attention import (
    MultiLatentRaggedPageAttention,
    multi_latent_ragged_page_attention,
)
from .multi_latent_ragged_page_attention_v2 import (
    MultiLatentRaggedPageAttentionV2,
    multi_latent_ragged_page_attention_v2,
)
from .native_sparse_attention import NativeSparseAttention, native_sparse_attention
from .page_attention import PageAttention, page_attention
from .pooling import MeanPooling, mean_pooling
from .prefill_page_attention import PrefillPageAttention, prefill_page_attention
from .quantized_matmul import QuantizedMatmul, quantized_matmul
from .ragged_decode_attention import RaggedDecodeAttention, ragged_decode_attention
from .ragged_gated_delta_rule import RaggedGatedDeltaRule, ragged_gated_delta_rule
from .ragged_page_attention_v2 import RaggedPageAttentionv2, ragged_page_attention_v2
from .ragged_page_attention_v2_turboquant import (
    RaggedPageAttentionv2TurboQuant,
    ragged_page_attention_v2_turboquant,
)
from .ragged_page_attention_v3 import RaggedPageAttentionv3, ragged_page_attention_v3
from .ragged_page_attention_v3_turboquant import (
    RaggedPageAttentionv3TurboQuant,
    ragged_page_attention_v3_turboquant,
)
from .recurrent import RecurrentAttention, recurrent_attention
from .reduce_scatter_matmul import ReduceScatterMatmul, reduce_scatter_matmul
from .ring_attention import RingAttention, ring_attention
from .rwkv4 import RWKV4, rwkv4
from .rwkv6 import RWKV6, rwkv6
from .rwkv7 import RWKV7, RWKV7Mul, rwkv7, rwkv7_mul
from .scaled_dot_product_attention import ScaledDotProductAttention, scaled_dot_product_attention
from .state_space_v1 import StateSpaceV1, state_space_v1
from .state_space_v2 import StateSpaceV2, state_space_v2
from .unified_attention import UnifiedAttention, unified_attention

gdr_attention = gated_delta_rule

__all__ = (
    "RWKV4",
    "RWKV6",
    "RWKV7",
    "AllGatherMatmul",
    "AllGatherMatmulConfig",
    "Attention",
    "AttentionConfig",
    "BlockSparseAttention",
    "BlockSparseAttentionConfig",
    "ChunkedPrefillPagedDecode",
    "ChunkedPrefillPagedDecodeConfig",
    "CrossEntropyOutput",
    "DecodeAttention",
    "DecodeAttentionConfig",
    "DeepSeekAttention",
    "DeepSeekAttentionConfig",
    "FlashAttention",
    "FlashAttentionConfig",
    "FlashMLA",
    "FlashMLAConfig",
    "FusedCrossEntropy",
    "FusedCrossEntropyConfig",
    "FusedKLDivergence",
    "FusedKLDivergenceConfig",
    "GLAttention",
    "GLAttentionConfig",
    "GatedDeltaRule",
    "GatedDeltaRuleConfig",
    "GroupedMatmul",
    "GroupedMatmulConfig",
    "KLDivergenceOutput",
    "KernelDeltaAttention",
    "KernelDeltaAttentionConfig",
    "LightningAttention",
    "LightningAttentionConfig",
    "MeanPooling",
    "MeanPoolingConfig",
    "MultiLatentRaggedPageAttention",
    "MultiLatentRaggedPageAttentionConfig",
    "MultiLatentRaggedPageAttentionV2",
    "MultiLatentRaggedPageAttentionV2Config",
    "NativeSparseAttention",
    "NativeSparseAttentionConfig",
    "PageAttention",
    "PageAttentionConfig",
    "PrefillPageAttention",
    "PrefillPageAttentionConfig",
    "QuantizedMatmul",
    "QuantizedMatmulConfig",
    "RWKV4Config",
    "RWKV6Config",
    "RWKV7Config",
    "RWKV7Mul",
    "RWKV7MulConfig",
    "RaggedDecodeAttention",
    "RaggedDecodeAttentionConfig",
    "RaggedGatedDeltaRule",
    "RaggedGatedDeltaRuleConfig",
    "RaggedPageAttentionv2",
    "RaggedPageAttentionv2Config",
    "RaggedPageAttentionv2TurboQuant",
    "RaggedPageAttentionv2TurboQuantConfig",
    "RaggedPageAttentionv3",
    "RaggedPageAttentionv3Config",
    "RaggedPageAttentionv3TurboQuant",
    "RaggedPageAttentionv3TurboQuantConfig",
    "RecurrentAttention",
    "RecurrentAttentionConfig",
    "ReduceScatterMatmul",
    "ReduceScatterMatmulConfig",
    "RingAttention",
    "RingAttentionConfig",
    "ScaledDotProductAttention",
    "ScaledDotProductAttentionConfig",
    "StateSpaceV1",
    "StateSpaceV1Config",
    "StateSpaceV2",
    "StateSpaceV2Config",
    "UnifiedAttention",
    "UnifiedAttentionConfig",
    "all_gather_matmul",
    "attention",
    "blocksparse_attention",
    "chunked_prefill_paged_decode",
    "decode_attention",
    "deepseek_attn",
    "flash_attention",
    "flash_mla",
    "fused_cross_entropy",
    "fused_kl_divergence",
    "gated_delta_rule",
    "gdr_attention",
    "gla_attention",
    "grouped_matmul",
    "kda_attention",
    "kernel_delta_attention",
    "lightning_attention",
    "mean_pooling",
    "multi_latent_ragged_page_attention",
    "multi_latent_ragged_page_attention_v2",
    "native_sparse_attention",
    "page_attention",
    "prefill_page_attention",
    "quantized_matmul",
    "ragged_decode_attention",
    "ragged_gated_delta_rule",
    "ragged_page_attention_v2",
    "ragged_page_attention_v2_turboquant",
    "ragged_page_attention_v3",
    "ragged_page_attention_v3_turboquant",
    "recurrent_attention",
    "reduce_scatter_matmul",
    "ring_attention",
    "rwkv4",
    "rwkv6",
    "rwkv7",
    "rwkv7_mul",
    "scaled_dot_product_attention",
    "state_space_v1",
    "state_space_v2",
    "unified_attention",
)
