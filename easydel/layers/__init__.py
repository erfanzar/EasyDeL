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

This module provides optimized neural network layers and components for building
deep learning models in JAX/Flax.

Submodules:
    attention: FlexibleAttentionModule, UnifiedAttention, AttentionMechanisms
    linears: ParallelLinear, ColumnParallelLinear, RowParallelLinear, quantized/MoE variants
    norms: RMSNorm, LayerNorm
    embeddings: Embed
    moe: BaseMoeModule, routing strategies
    quantization: QuantizationConfig, EasyQuantizer
    rotary: RotaryEmbedding and variants, RopeConfig
"""

from .embeddings import Embed
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
from .norms import RMSNorm
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
