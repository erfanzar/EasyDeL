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

"""EasyDeL Layers Module — pure NN building blocks.

Top-level aggregator for every reusable neural-network primitive that does
not depend on a specific model architecture. The package is structured so
that model implementations under ``easydel/modules/`` can compose attention,
MLP, norm, embedding, MoE, and quantization pieces directly from this
namespace without reaching into private modules.

Submodules:
    :mod:`attention`: :class:`FlexibleAttentionModule`, :class:`UnifiedAttention`,
        :class:`AttentionMechanisms`, and decoder layer helpers.
    :mod:`layouts`: Fused projection layouts and TP-interleaved checkpoint
        reform rules (:class:`FusedColumnLayout`, :class:`FusedExpertLayout`,
        :func:`build_fused_qkv_projection`, :func:`build_fused_gate_up_projection`).
    :mod:`linears`: :class:`ParallelLinear`, :class:`ColumnParallelLinear`,
        :class:`RowParallelLinear`, plus quantized and MoE variants.
    :mod:`norms`: :class:`RMSNorm`, :class:`RMSNormGated`.
    :mod:`embeddings`: :class:`Embed`.
    :mod:`moe`: :class:`BaseMoeModule` and routing/load-balancing strategies.
    :mod:`quantization`: :class:`QuantizationConfig`, :class:`EasyQuantizer`,
        straight-through estimators.
    :mod:`rotary`: Rotary embedding variants (Llama3, YaRN, DeepSeek, Phi3,
        dynamic-NTK, linear scaling) and frequency computation helpers.
"""

from .embeddings import Embed
from .layouts import (
    FusedColumnLayout,
    FusedExpertLayout,
    FusedSegment,
    build_fused_gate_up_projection,
    build_fused_qkv_projection,
    dense_gate_up_layout,
    dense_qkv_layout,
    gate_up_fusion_reform_param,
    hf_per_expert_swiglu_reform_param,
    interleaved_fusion_reform_param,
    keep_interleaved_segments_last_axis,
    moe_down_projection_reform_param,
    moe_fused_gate_up_reform_param,
    moe_gate_up_fusion_reform_param,
    normalize_segment_sizes,
    qkv_fusion_reform_param,
    split_contiguous_segments_last_axis,
    split_fused_gate_up_projection,
    split_fused_qkv_projection,
    split_interleaved_pair_last_axis,
    split_interleaved_segments_last_axis,
    tensor_parallel_axis,
    tensor_parallel_size,
    torch_deinterleave_axis_segments_for_tp,
    torch_deinterleave_segments_for_tp,
    torch_interleave_axis_segments_for_tp,
    torch_interleave_segments_for_tp,
    with_tp_last_axis_sharding,
)
from .linears import (
    ColumnParallelLinear,
    ColumnParallelLinearQuantized,
    ColumnParallelMoELinear,
    ParallelLinear,
    ParallelLinearQuantized,
    ParallelMoELinear,
    RowParallelLinear,
    RowParallelLinearQuantized,
    RowParallelMoELinear,
    eLoRA,
)
from .moe import (
    BaseMoeModule,
    MoeFusedHooks,
    MoeLoadBalancingStrategy,
    MoEMethods,
    MoeMetrics,
    MoeRoutingStrategy,
    get_moe_partition_spec,
)
from .norms import RMSNorm, RMSNormGated
from .quantization import (
    EasyDeLQuantizationConfig,
    EasyQuantizer,
    QuantizationConfig,
    QuantizationType,
    quantize,
    straight_through,
    straight_through_1bit,
    straight_through_8bit,
    straight_through_mxfp4,
    straight_through_mxfp8,
    straight_through_nf4,
    straight_through_nvfp8,
)
from .rotary import (
    DeepseekScalingRotaryEmbedding,
    DynamicNTKScalingRotaryEmbedding,
    LinearScalingRotaryEmbedding,
    Llama3RotaryEmbedding,
    MultiModalRotaryEmbedding,
    Phi3LongRoPEScaledRotaryEmbedding,
    RopeConfig,
    RotaryEmbedding,
    YaRNScalingRotaryEmbedding,
    compute_basic_frequencies,
    compute_basic_inv_frequencies,
    compute_deepseek_frequencies,
    compute_dynamic_frequencies,
    compute_linear_frequencies,
    compute_llama3_frequencies,
    compute_llama3_inv_frequencies,
    compute_phi3_frequencies,
    compute_yarn_frequencies,
    compute_yarn_inv_frequencies,
    get_frequencies,
    get_inv_frequencies,
    get_rope,
)

__all__ = [
    "BaseMoeModule",
    "ColumnParallelLinear",
    "ColumnParallelLinearQuantized",
    "ColumnParallelMoELinear",
    "DeepseekScalingRotaryEmbedding",
    "DynamicNTKScalingRotaryEmbedding",
    "EasyDeLQuantizationConfig",
    "EasyQuantizer",
    "Embed",
    "FusedColumnLayout",
    "FusedExpertLayout",
    "FusedSegment",
    "LinearScalingRotaryEmbedding",
    "Llama3RotaryEmbedding",
    "MoEMethods",
    "MoeFusedHooks",
    "MoeLoadBalancingStrategy",
    "MoeMetrics",
    "MoeRoutingStrategy",
    "MultiModalRotaryEmbedding",
    "ParallelLinear",
    "ParallelLinearQuantized",
    "ParallelMoELinear",
    "Phi3LongRoPEScaledRotaryEmbedding",
    "QuantizationConfig",
    "QuantizationType",
    "RMSNorm",
    "RMSNormGated",
    "RopeConfig",
    "RotaryEmbedding",
    "RowParallelLinear",
    "RowParallelLinearQuantized",
    "RowParallelMoELinear",
    "YaRNScalingRotaryEmbedding",
    "build_fused_gate_up_projection",
    "build_fused_qkv_projection",
    "compute_basic_frequencies",
    "compute_basic_inv_frequencies",
    "compute_deepseek_frequencies",
    "compute_dynamic_frequencies",
    "compute_linear_frequencies",
    "compute_llama3_frequencies",
    "compute_llama3_inv_frequencies",
    "compute_phi3_frequencies",
    "compute_yarn_frequencies",
    "compute_yarn_inv_frequencies",
    "dense_gate_up_layout",
    "dense_qkv_layout",
    "eLoRA",
    "gate_up_fusion_reform_param",
    "get_frequencies",
    "get_inv_frequencies",
    "get_moe_partition_spec",
    "get_rope",
    "hf_per_expert_swiglu_reform_param",
    "interleaved_fusion_reform_param",
    "keep_interleaved_segments_last_axis",
    "moe_down_projection_reform_param",
    "moe_fused_gate_up_reform_param",
    "moe_gate_up_fusion_reform_param",
    "normalize_segment_sizes",
    "qkv_fusion_reform_param",
    "quantize",
    "split_contiguous_segments_last_axis",
    "split_fused_gate_up_projection",
    "split_fused_qkv_projection",
    "split_interleaved_pair_last_axis",
    "split_interleaved_segments_last_axis",
    "straight_through",
    "straight_through_1bit",
    "straight_through_8bit",
    "straight_through_mxfp4",
    "straight_through_mxfp8",
    "straight_through_nf4",
    "straight_through_nvfp8",
    "tensor_parallel_axis",
    "tensor_parallel_size",
    "torch_deinterleave_axis_segments_for_tp",
    "torch_deinterleave_segments_for_tp",
    "torch_interleave_axis_segments_for_tp",
    "torch_interleave_segments_for_tp",
    "with_tp_last_axis_sharding",
]
