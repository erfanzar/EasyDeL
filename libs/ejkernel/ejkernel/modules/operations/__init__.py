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
        - GatedDeltaRuleGroupedDecode / gated_delta_rule_grouped_decode: Grouped-head
            single-step GDR decode (XLA reference + TPU Pallas for head sharding);
            ``gated_delta_rule_grouped_decode_op`` is a back-compat alias of the functional op.
        - RaggedGatedDeltaRule / ragged_gated_delta_rule: GDR for ragged sequences.
        - RaggedGatedDeltaRuleV2 / ragged_gated_delta_rule_v2: Packed continuous-batch
            GDN v2 (mixed prefill+decode); consumes a packed mixed-QKV buffer plus
            per-token gates and a recurrent state pool, returns the updated state pool
            and packed output.  ``ragged_gated_delta_rule_decode_only`` and
            ``ragged_gated_delta_rule_mixed_prefill`` are the specialised TPU Pallas
            entry points; ``ragged_gated_delta_rule_v2_op`` is a back-compat alias
            of the functional op.

    GDN / Conv-State Inference Helpers:
        - GDNComputeScheduleV2 / gdn_compute_schedule_v2: Build the GDN v2 schedule
            table splitting a packed batch into decode work, regular prefill chunks,
            and transition rows; ``compute_schedule_table_v2`` is the underlying
            host/XLA reference builder.
        - RaggedCausalConv1D / ragged_causal_conv1d: Short causal depthwise conv1d over
            a packed token buffer with per-request rolling state
            (``query_start_loc``/``state_indices``); ``ragged_causal_conv1d_head_sharded``
            adds the feature-axis ``shard_map`` path and ``ragged_causal_conv1d_op``
            is a back-compat alias of the functional op.
        - FusedConvDecode / fused_conv_decode: Single-token conv-state shift + depthwise
            convolution + activation for GDR/GDN decode; ``fused_conv_decode_op`` is a
            back-compat alias of the functional op.

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

    Fused Training Losses:
        - FusedCrossEntropy / fused_cross_entropy: Per-row cross-entropy fused with its
            analytic gradient, never materialising the ``[..., V]`` softmax in HBM;
            supports label smoothing, z-loss, and dense soft targets, with optional
            ``jax.shard_map`` vocab-parallel reduction.  Returns a ``CrossEntropyOutput``.
        - FusedKLDivergence / fused_kl_divergence: Per-row forward KL
            ``KL(softmax(teacher) || softmax(student))`` fused with the analytic student
            gradient (teacher detached); same HBM-free + vocab-parallel behaviour.
            Returns a ``KLDivergenceOutput``.

    Miscellaneous:
        - MeanPooling / mean_pooling: Sequence mean pooling.

Configuration classes (imported from ``configs``):
    Each operation has a corresponding ``*Config`` dataclass documented in
    ``ejkernel.modules.operations.configs``.  All configs inherit ``platform``
    and ``backend`` from ``BaseOperationConfig``.

GDN Tile-Policy Controls:
    Re-exported from the TPU ragged GDN v2 backend for tuning the packed-token
    block size used by ``ragged_gated_delta_rule_v2``:

    - ``KernelTilePolicy``: literal type alias of valid policy names.
    - ``KERNEL_TILE_POLICIES``: frozenset of valid policy strings
      (``"auto"``, ``"b16"``, ``"b8"``, ``"b4"``, ``"b2"``, ``"b1"``).
    - ``normalize_kernel_tile_policy``: validate/lower-case a policy string
      (``None`` -> ``"auto"``); raises ``ValueError`` on an unknown value.
    - ``set_gdn_kernel_tile_policy``: install the active global tile policy.

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
from .compressed_window_attention import CompressedWindowAttention, compressed_window_attention
from .compressed_window_decode import CompressedWindowDecode, compressed_window_decode
from .configs import (
    AllGatherMatmulConfig,
    AttentionConfig,
    BlockSparseAttentionConfig,
    ChunkedPrefillPagedDecodeConfig,
    CompressedWindowAttentionConfig,
    CompressedWindowDecodeConfig,
    DecodeAttentionConfig,
    DeepSeekAttentionConfig,
    FlashAttentionConfig,
    FlashMLAConfig,
    FusedConvDecodeConfig,
    FusedCrossEntropyConfig,
    FusedKLDivergenceConfig,
    GatedDeltaRuleConfig,
    GatedDeltaRuleGroupedDecodeConfig,
    GDNComputeScheduleV2Config,
    GdnSpecWindowStatesConfig,
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
    RaggedCausalConv1DConfig,
    RaggedDecodeAttentionConfig,
    RaggedGatedDeltaRuleConfig,
    RaggedGatedDeltaRuleV2Config,
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
from .fused_conv_decode import FusedConvDecode, fused_conv_decode, fused_conv_decode_op
from .fused_cross_entropy import CrossEntropyOutput, FusedCrossEntropy, fused_cross_entropy
from .fused_kl_divergence import FusedKLDivergence, KLDivergenceOutput, fused_kl_divergence
from .gated_delta_rule import GatedDeltaRule, gated_delta_rule
from .gated_delta_rule_grouped_decode import (
    GatedDeltaRuleGroupedDecode,
    gated_delta_rule_grouped_decode,
    gated_delta_rule_grouped_decode_op,
)
from .gated_linear_attention import GLAttention, gla_attention
from .gdn_compute_schedule_v2 import GDNComputeScheduleV2, compute_schedule_table_v2, gdn_compute_schedule_v2
from .gdn_spec_window import (
    GdnSpecWindowStates,
    gdn_spec_window_states,
    gdn_spec_window_states_op,
)
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
from .ragged_causal_conv1d import (
    RaggedCausalConv1D,
    ragged_causal_conv1d,
    ragged_causal_conv1d_head_sharded,
    ragged_causal_conv1d_op,
)
from .ragged_decode_attention import RaggedDecodeAttention, ragged_decode_attention
from .ragged_gated_delta_rule import RaggedGatedDeltaRule, ragged_gated_delta_rule
from .ragged_gated_delta_rule_v2 import (
    KERNEL_TILE_POLICIES,
    KernelTilePolicy,
    RaggedGatedDeltaRuleV2,
    normalize_kernel_tile_policy,
    ragged_gated_delta_rule_decode_only,
    ragged_gated_delta_rule_mixed_prefill,
    ragged_gated_delta_rule_v2,
    ragged_gated_delta_rule_v2_op,
    set_gdn_kernel_tile_policy,
)
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
    "KERNEL_TILE_POLICIES",
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
    "CompressedWindowAttention",
    "CompressedWindowAttentionConfig",
    "CompressedWindowDecode",
    "CompressedWindowDecodeConfig",
    "CrossEntropyOutput",
    "DecodeAttention",
    "DecodeAttentionConfig",
    "DeepSeekAttention",
    "DeepSeekAttentionConfig",
    "FlashAttention",
    "FlashAttentionConfig",
    "FlashMLA",
    "FlashMLAConfig",
    "FusedConvDecode",
    "FusedConvDecodeConfig",
    "FusedCrossEntropy",
    "FusedCrossEntropyConfig",
    "FusedKLDivergence",
    "FusedKLDivergenceConfig",
    "GDNComputeScheduleV2",
    "GDNComputeScheduleV2Config",
    "GLAttention",
    "GLAttentionConfig",
    "GatedDeltaRule",
    "GatedDeltaRuleConfig",
    "GatedDeltaRuleGroupedDecode",
    "GatedDeltaRuleGroupedDecodeConfig",
    "GdnSpecWindowStates",
    "GdnSpecWindowStatesConfig",
    "GroupedMatmul",
    "GroupedMatmulConfig",
    "KLDivergenceOutput",
    "KernelDeltaAttention",
    "KernelDeltaAttentionConfig",
    "KernelTilePolicy",
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
    "RaggedCausalConv1D",
    "RaggedCausalConv1DConfig",
    "RaggedDecodeAttention",
    "RaggedDecodeAttentionConfig",
    "RaggedGatedDeltaRule",
    "RaggedGatedDeltaRuleConfig",
    "RaggedGatedDeltaRuleV2",
    "RaggedGatedDeltaRuleV2Config",
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
    "compressed_window_attention",
    "compressed_window_decode",
    "compute_schedule_table_v2",
    "decode_attention",
    "deepseek_attn",
    "flash_attention",
    "flash_mla",
    "fused_conv_decode",
    "fused_conv_decode_op",
    "fused_cross_entropy",
    "fused_kl_divergence",
    "gated_delta_rule",
    "gated_delta_rule_grouped_decode",
    "gated_delta_rule_grouped_decode_op",
    "gdn_compute_schedule_v2",
    "gdn_spec_window_states",
    "gdn_spec_window_states_op",
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
    "normalize_kernel_tile_policy",
    "page_attention",
    "prefill_page_attention",
    "quantized_matmul",
    "ragged_causal_conv1d",
    "ragged_causal_conv1d_head_sharded",
    "ragged_causal_conv1d_op",
    "ragged_decode_attention",
    "ragged_gated_delta_rule",
    "ragged_gated_delta_rule_decode_only",
    "ragged_gated_delta_rule_mixed_prefill",
    "ragged_gated_delta_rule_v2",
    "ragged_gated_delta_rule_v2_op",
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
    "set_gdn_kernel_tile_policy",
    "state_space_v1",
    "state_space_v2",
    "unified_attention",
)
