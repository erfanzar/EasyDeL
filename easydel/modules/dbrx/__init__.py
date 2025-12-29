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

"""DBRX model implementation for EasyDeL.

This module provides the DBRX (Databricks) model architecture, a state-of-the-art
fine-grained Mixture-of-Experts (MoE) transformer developed by Databricks. DBRX
features an innovative expert architecture with 16 experts where each token is
routed to the top-4 most relevant experts.

Architecture Highlights:
    - Fine-grained MoE: 16 experts per layer with top-4 expert selection per token
    - Multi-Query Attention (MQA): Single key-value head (kv_n_heads=1) for efficiency
    - Grouped Query Attention support: Configurable number of KV heads
    - QKV Clipping: Optional clipping of query, key, value tensors (clip_qkv=8.0)
    - Expert normalization: Configurable expert weight normalization for stability
    - Rotary Position Embeddings (RoPE): Standard RoPE with theta=10000

Key Features:
    - Total experts: 16 per MoE layer (moe_num_experts=16)
    - Active experts: 4 experts per token (moe_top_k=4)
    - Expert specialization: Fine-grained expert assignment for better specialization
    - Vocabulary size: 32,000 tokens (configurable)
    - Maximum sequence length: 2,048 tokens default
    - Router auxiliary loss: Weighted auxiliary loss for load balancing (moe_loss_weight=0.01)

MoE Configuration:
    - FFN hidden size: 3,584 dimensions per expert (ffn_hidden_size)
    - Activation: SiLU activation function
    - Jitter noise: Optional jitter for expert routing (moe_jitter_eps)
    - Uniform assignment: Optional uniform expert assignment for debugging
    - Weight normalization: Expert output weight normalization (moe_normalize_expert_weights)

Attention Configuration:
    - Attention dropout: Configurable dropout on attention outputs (attn_pdrop)
    - Clip QKV: Gradient clipping for attention tensors
    - KV heads: Typically 1 for MQA, configurable for GQA
    - RoPE theta: Base frequency for rotary embeddings

Usage Example:
    ```python
    from easydel import DbrxConfig, DbrxForCausalLM, DbrxAttentionConfig, DbrxFFNConfig
    from jax import random
    import jax.numpy as jnp

    # Configure attention
    attn_config = DbrxAttentionConfig(
        attn_pdrop=0.0,
        clip_qkv=8.0,
        kv_n_heads=1,  # MQA
        rope_theta=10000.0
    )

    # Configure MoE FFN
    ffn_config = DbrxFFNConfig(
        ffn_hidden_size=3584,
        moe_num_experts=16,  # Total experts
        moe_top_k=4,  # Active experts per token
        moe_loss_weight=0.01,
        moe_normalize_expert_weights=1.0
    )

    # Initialize full configuration
    config = DbrxConfig(
        d_model=2048,  # Hidden dimension
        n_heads=16,  # Attention heads
        n_layers=24,  # Transformer layers
        max_seq_len=2048,
        vocab_size=32000,
        attn_config=attn_config,
        ffn_config=ffn_config,
        output_router_logits=True,  # For auxiliary loss
        router_aux_loss_coef=0.05
    )

    # Create model for causal LM
    model = DbrxForCausalLM(
        config=config,
        rngs=random.PRNGKey(0)
    )

    # Prepare inputs
    input_ids = jnp.ones((1, 256), dtype=jnp.int32)

    # Forward pass
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # Shape: (1, 256, 32000)
    router_logits = outputs.router_logits  # MoE routing information
    ```

Available Classes:
    - DbrxConfig: Main configuration class for DBRX models
    - DbrxAttentionConfig: Attention-specific configuration
    - DbrxFFNConfig: MoE FFN-specific configuration
    - DbrxModel: Base transformer model without task-specific heads
    - DbrxForCausalLM: DBRX model with causal language modeling head
    - DbrxForSequenceClassification: DBRX model with classification head
"""

from .dbrx_configuration import DbrxAttentionConfig, DbrxConfig, DbrxFFNConfig
from .modeling_dbrx import DbrxForCausalLM, DbrxForSequenceClassification, DbrxModel

__all__ = (
    "DbrxAttentionConfig",
    "DbrxConfig",
    "DbrxFFNConfig",
    "DbrxForCausalLM",
    "DbrxForSequenceClassification",
    "DbrxModel",
)
