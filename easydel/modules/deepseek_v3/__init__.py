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

"""DeepSeek V3 model implementation for EasyDeL.

This module provides the DeepSeek V3 model architecture with advanced
Multi-head Latent Attention and mixture-of-experts capabilities.

DeepSeek V3 is an advanced large language model developed by DeepSeek AI, featuring
innovative Multi-head Latent Attention (MLA) and an efficient mixture-of-experts (MoE)
architecture. It achieves state-of-the-art performance while maintaining computational
efficiency through novel attention mechanisms and expert routing strategies.

Key Architectural Features:
    - Multi-head Latent Attention (MLA): Revolutionary attention mechanism that
      compresses key-value cache into low-rank latent representations, dramatically
      reducing KV cache memory requirements (up to 16x compression) while maintaining
      model quality. Uses separate latent dimensions for keys and values.
    - Fine-grained Mixture of Experts (MoE): Employs a large number of experts
      (typically 256+) with top-k routing (default k=8) and auxiliary load balancing
      losses. Includes shared experts that are always activated alongside routed experts.
    - Multi-Token Prediction (MTP): Optional auxiliary training objective that predicts
      multiple future tokens simultaneously, improving sample efficiency and model
      quality through richer training signals.
    - RoPE with Yarn Scaling: Uses YaRN-based RoPE scaling for extended context
      windows beyond pre-training length, supporting very long sequences.
    - QK Normalization: Applies RMSNorm to query and key projections before attention
      computation for improved training stability.
    - Auxiliary Loss Balancing: Implements expert load balancing and routing losses
      to ensure efficient expert utilization during MoE training.

Model Variants:
    - DeepseekV3Model: Base decoder-only transformer with MLA and MoE.
    - DeepseekV3ForCausalLM: Language modeling head for next-token prediction.

Usage Example:
    ```python
    from easydel import AutoEasyDeLModelForCausalLM
    import jax.numpy as jnp

    # Load pretrained model
    model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-V3",
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
    )

    # Prepare input tokens
    input_ids = jnp.array([[1, 2, 3, 4, 5]])

    # Generate text with efficient KV caching via MLA
    outputs = model.generate(
        input_ids=input_ids,
        params=params,
        max_length=100,
        temperature=0.7,
    )
    ```

Configuration Example:
    ```python
    from easydel.modules.deepseek_v3 import DeepseekV3Config

    config = DeepseekV3Config(
        vocab_size=102400,
        hidden_size=7168,
        intermediate_size=18432,
        num_hidden_layers=61,
        num_attention_heads=128,
        num_key_value_heads=128,
        # MLA configuration
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        # MoE configuration
        n_routed_experts=256,
        num_experts_per_tok=8,
        n_shared_experts=2,
        # Advanced features
        qk_nope_head_dim=128,
        rope_theta=10000.0,
        max_position_embeddings=163840,
    )
    ```

For more information on DeepSeek V3, see:
    - Model Card: https://huggingface.co/deepseek-ai/DeepSeek-V3
    - Technical Report: https://arxiv.org/abs/2412.19437
    - MLA Paper: Multi-head Latent Attention
"""

from .deepseek_configuration import DeepseekV3Config
from .modeling_deepseek import DeepseekV3ForCausalLM, DeepseekV3Model

__all__ = ("DeepseekV3Config", "DeepseekV3ForCausalLM", "DeepseekV3Model")
