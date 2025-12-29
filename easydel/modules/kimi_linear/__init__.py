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

"""Kimi-Linear: Hybrid MLA + KDA Attention with MoE.

This module implements the Kimi-Linear architecture from Moonshot AI, featuring a hybrid
attention mechanism that combines Multi-head Latent Attention (MLA) and Kernel Delta
Attention (KDA) linear attention, along with Mixture of Experts (MoE) for efficient scaling.

Architecture Overview:
    Kimi-Linear uses a hybrid attention architecture:

    1. Hybrid Attention Mechanism:
       - Full Attention Layers (MLA): Multi-head Latent Attention from DeepSeek-V3
         * Compressed KV cache via low-rank projections (kv_lora_rank)
         * Query absorbs KV pattern for efficiency
         * Separate nope (non-positional) and rope (rotational) components

       - Linear Attention Layers (KDA): Kernel Delta Attention
         * O(N) complexity for long sequences (vs O(N^2) for full attention)
         * Kernel-based approximation of attention
         * Delta mechanism for expressiveness
         * Ideal for processing very long contexts

       - Layer Assignment: Configurable via layer_types or linear_attn_config
         * Typically alternates between full and linear attention
         * Early layers often use full attention for strong modeling
         * Deep layers may use linear attention for long-range dependencies

    2. Mixture of Experts (MoE):
       - Hybrid dense-sparse architecture like GLM-4-MoE
       - First `first_k_dense_replace` layers use dense FFN
       - MoE layers appear every `moe_layer_freq` layers
       - Sigmoid router activation (vs softmax in most MoE models)
       - Shared experts: Always active for common knowledge
       - Routed experts: Conditionally selected via top-k routing
       - Grouped top-k: Hierarchical routing through expert groups

    3. MLA (Multi-head Latent Attention):
       - Compresses KV cache to kv_lora_rank dimension (typically 512)
       - Query uses q_lora_rank for compression (if specified)
       - Separate qk_nope_head_dim and qk_rope_head_dim components
       - v_head_dim for value projection dimension
       - Dramatically reduces memory footprint for long contexts

    4. KDA (Kernel Delta Attention):
       - Linear complexity O(N) attention approximation
       - Kernel function projects Q and K to feature space
       - Delta mechanism adds expressiveness to kernel attention
       - Enables efficient processing of very long sequences
       - Configured via linear_attn_config dictionary

Key Features:
    - Hybrid Full + Linear Attention: MLA for strong modeling, KDA for efficiency
    - Multi-head Latent Attention (MLA): Compressed KV cache for memory efficiency
    - Kernel Delta Attention (KDA): Linear complexity for long sequences
    - MoE with Sigmoid Routing: Different from standard softmax routing
    - Shared + Routed Experts: Hybrid expert architecture
    - Grouped Top-K: Hierarchical expert selection

Attention Layer Configuration:
    - layer_types: List specifying attention type per layer (FULL_ATTENTION or KDA_LINEAR_ATTENTION)
    - num_nextn_predict_layers: Number of layers using next-n prediction
    - linear_attn_config: Detailed configuration for KDA layers

MoE Configuration:
    - num_experts: Total number of routed experts
    - num_experts_per_token: Active experts per token
    - num_shared_experts: Always-active shared experts
    - moe_router_activation_func: "sigmoid" (default) or "softmax"
    - routed_scaling_factor: Scale factor for routed expert outputs

Usage Example:
    ```python
    from easydel import KimiLinearForCausalLM, KimiLinearConfig
    from easydel.layers.caching.hybrid import FULL_ATTENTION, KDA_LINEAR_ATTENTION
    import jax.numpy as jnp

    # Configure hybrid attention + MoE model
    config = KimiLinearConfig(
        hidden_size=4096,
        num_hidden_layers=48,
        num_attention_heads=32,
        # MLA configuration
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        # MoE configuration
        num_experts=64,
        num_experts_per_token=8,
        num_shared_experts=2,
        first_k_dense_replace=1,
        # Hybrid attention: alternate between full and linear
        layer_types=[FULL_ATTENTION if i % 2 == 0 else KDA_LINEAR_ATTENTION
                     for i in range(48)],
    )

    model = KimiLinearForCausalLM(config, rngs=nnx.Rngs(0))

    # Forward pass
    input_ids = jnp.array([[1, 2, 3, 4, 5]])
    outputs = model(input_ids=input_ids, output_router_logits=True)

    logits = outputs.logits  # (batch, seq_len, vocab_size)
    router_logits = outputs.router_logits  # For MoE load balancing
    ```

Configuration:
    - KimiLinearConfig: Main configuration with MLA, KDA, and MoE parameters

Models:
    - KimiLinearModel: Base transformer with hybrid attention and MoE
    - KimiLinearForCausalLM: Causal language modeling with LM head

References:
    - Kimi Linear: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct
    - DeepSeek-V3 (MLA): https://huggingface.co/deepseek-ai/DeepSeek-V3
"""

from .kimi_linear_configuration import KimiLinearConfig
from .modeling_kimi_linear import KimiLinearForCausalLM, KimiLinearModel

__all__ = ("KimiLinearConfig", "KimiLinearForCausalLM", "KimiLinearModel")
