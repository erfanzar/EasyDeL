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


"""High-level kernel modules with automatic optimization.

This is the top-level public namespace for ejkernel. It re-exports every class
and functional alias from ``ejkernel.modules.operations`` so that callers can
import directly from here without traversing the internal package layout.

All operations follow the same two-layer pattern:

1. **Class interface** — a ``Kernel`` subclass (e.g. ``FlashAttention``) that
   exposes ``run()``, ``heuristic_cfg()``, and ``candidate_cfgs()`` methods and
   is intended to be driven by an ``Executor``.
2. **Functional interface** — a module-level function (e.g. ``flash_attention``)
   that wraps a pre-configured ``Executor`` for one-line call-site usage.

Kernel selection is automatic by default: ``detect_platform`` inspects the
active JAX backend (CPU/GPU/TPU) and the kernel registry to pick the best
available implementation (XLA, Triton, Pallas GPU, Pallas TPU, CUDA, CUTE).

Available Attention Modules:
    Standard Attention:
        - Attention: Standard multi-head attention (XLA); returns output + weights.
        - FlashAttention: Memory-efficient O(N) attention via tiling.
        - ScaledDotProductAttention: Thin wrapper over XLA dot-product attention.

    Paged/Serving Attention:
        - PageAttention: Single-query paged decode attention.
        - DecodeAttention: vLLM-style paged decode attention returning output + LSE.
        - UnifiedAttention: vLLM-style unified (prefill + decode) paged attention.
        - PrefillPageAttention: Chunked prefill with paged KV cache.
        - ChunkedPrefillPagedDecode: Fused KV-cache write + paged decode in one pass.
        - RaggedDecodeAttention: Ragged sequence decode attention.
        - RaggedPageAttentionv2: Ragged paged attention (read-only cache).
        - RaggedPageAttentionv2TurboQuant: TurboQuant-compressed ragged paged attention v2.
        - RaggedPageAttentionv3: Ragged paged attention with chunked prefill support.
        - RaggedPageAttentionv3TurboQuant: TurboQuant-compressed ragged paged attention v3.

    Sparse Attention:
        - BlockSparseAttention: Block-sparse attention with custom mask builder.
        - NativeSparseAttention: Sparse attention with explicit block patterns.

    Linear/Recurrent Attention:
        - GatedDeltaRule: Gated delta rule linear attention (O(N)).
        - RaggedGatedDeltaRule: GDR for ragged (variable-length packed) sequences.
        - GLAttention: Gated linear attention with token-level and layer-level gates.
        - LightningAttention: Lightning attention with per-layer decay factors.
        - KernelDeltaAttention: Linear attention with delta rule updates.
        - RecurrentAttention: Stateful recurrent attention.

    Distributed Attention:
        - RingAttention: Ring-topology distributed attention.

    MLA / Compressed Attention:
        - DeepSeekAttention: DeepSeek Sparse Attention (MLA + Lightning Indexer).
        - FlashMLA: Flash multi-head latent attention with low-rank KV compression.
        - MultiLatentRaggedPageAttention: MLA ragged paged attention.
        - MultiLatentRaggedPageAttentionV2: MLA ragged paged attention v2 with
          per-phase block-size tuning.

Linear Recurrent Models:
    - RWKV4: WKV recurrence with fixed time-decay (RWKV v4).
    - RWKV6: WKV recurrence with token-dependent decay (RWKV v6).
    - RWKV7: Enhanced state-update recurrence (RWKV v7).
    - RWKV7Mul: RWKV-7 with multiplicative state updates.

State Space Models:
    - StateSpaceV1: Mamba-1 style selective state space model.
    - StateSpaceV2: Mamba-2 style selective state space model (SSD).

Matrix Operations:
    - GroupedMatmul: Grouped matmul for Mixture-of-Experts routing.
    - AllGatherMatmul: Fused all-gather + matmul for tensor-parallel layers.
    - ReduceScatterMatmul: Fused matmul + reduce-scatter for tensor-parallel layers.
    - QuantizedMatmul: Packed uint32 symmetric quantized matmul.

Miscellaneous:
    - MeanPooling: Sequence mean pooling with optional attention mask.

Aliases:
    ``gdr_attention`` is an alias for ``gated_delta_rule``.
    ``kda_attention`` and ``kernel_delta_attention`` both refer to the same op.

Example:
    >>> from ejkernel.modules import FlashAttention, flash_attention
    >>>
    >>> # Class interface (for use with a custom Executor)
    >>> attn = FlashAttention()
    >>> output = attn.run(q, k, v, causal=True, cfg=attn.heuristic_cfg(None))
    >>>
    >>> # Functional interface (uses the built-in Executor with autotuning)
    >>> output = flash_attention(q, k, v, causal=True)
"""

from .operations import (
    RWKV4,
    RWKV6,
    RWKV7,
    AllGatherMatmul,
    AllGatherMatmulConfig,
    Attention,
    AttentionConfig,
    BlockSparseAttention,
    BlockSparseAttentionConfig,
    ChunkedPrefillPagedDecode,
    ChunkedPrefillPagedDecodeConfig,
    DecodeAttention,
    DecodeAttentionConfig,
    DeepSeekAttention,
    DeepSeekAttentionConfig,
    FlashAttention,
    FlashAttentionConfig,
    FlashMLA,
    FlashMLAConfig,
    GatedDeltaRule,
    GatedDeltaRuleConfig,
    GLAttention,
    GLAttentionConfig,
    GroupedMatmul,
    GroupedMatmulConfig,
    KernelDeltaAttention,
    KernelDeltaAttentionConfig,
    LightningAttention,
    LightningAttentionConfig,
    MeanPooling,
    MeanPoolingConfig,
    MultiLatentRaggedPageAttention,
    MultiLatentRaggedPageAttentionConfig,
    MultiLatentRaggedPageAttentionV2,
    MultiLatentRaggedPageAttentionV2Config,
    NativeSparseAttention,
    NativeSparseAttentionConfig,
    PageAttention,
    PageAttentionConfig,
    PrefillPageAttention,
    PrefillPageAttentionConfig,
    QuantizedMatmul,
    QuantizedMatmulConfig,
    RaggedDecodeAttention,
    RaggedDecodeAttentionConfig,
    RaggedGatedDeltaRule,
    RaggedGatedDeltaRuleConfig,
    RaggedPageAttentionv2,
    RaggedPageAttentionv2Config,
    RaggedPageAttentionv2TurboQuant,
    RaggedPageAttentionv2TurboQuantConfig,
    RaggedPageAttentionv3,
    RaggedPageAttentionv3Config,
    RaggedPageAttentionv3TurboQuant,
    RaggedPageAttentionv3TurboQuantConfig,
    RecurrentAttention,
    RecurrentAttentionConfig,
    ReduceScatterMatmul,
    ReduceScatterMatmulConfig,
    RingAttention,
    RingAttentionConfig,
    RWKV4Config,
    RWKV6Config,
    RWKV7Config,
    RWKV7Mul,
    RWKV7MulConfig,
    ScaledDotProductAttention,
    ScaledDotProductAttentionConfig,
    StateSpaceV1,
    StateSpaceV1Config,
    StateSpaceV2,
    StateSpaceV2Config,
    UnifiedAttention,
    UnifiedAttentionConfig,
    all_gather_matmul,
    attention,
    blocksparse_attention,
    chunked_prefill_paged_decode,
    decode_attention,
    deepseek_attn,
    flash_attention,
    flash_mla,
    gated_delta_rule,
    gdr_attention,
    gla_attention,
    grouped_matmul,
    kda_attention,
    kernel_delta_attention,
    lightning_attention,
    mean_pooling,
    multi_latent_ragged_page_attention,
    multi_latent_ragged_page_attention_v2,
    native_sparse_attention,
    page_attention,
    prefill_page_attention,
    quantized_matmul,
    ragged_decode_attention,
    ragged_gated_delta_rule,
    ragged_page_attention_v2,
    ragged_page_attention_v2_turboquant,
    ragged_page_attention_v3,
    ragged_page_attention_v3_turboquant,
    recurrent_attention,
    reduce_scatter_matmul,
    ring_attention,
    rwkv4,
    rwkv6,
    rwkv7,
    rwkv7_mul,
    scaled_dot_product_attention,
    state_space_v1,
    state_space_v2,
    unified_attention,
)

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
    "DecodeAttention",
    "DecodeAttentionConfig",
    "DeepSeekAttention",
    "DeepSeekAttentionConfig",
    "FlashAttention",
    "FlashAttentionConfig",
    "FlashMLA",
    "FlashMLAConfig",
    "GLAttention",
    "GLAttentionConfig",
    "GatedDeltaRule",
    "GatedDeltaRuleConfig",
    "GroupedMatmul",
    "GroupedMatmulConfig",
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
