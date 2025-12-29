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

"""Grok-1 Model Implementation for EasyDeL.

This module provides the Grok-1 architecture from xAI, a massive 314 billion parameter
Mixture-of-Experts (MoE) decoder-only transformer model. Grok-1 uses sparse activation
with 8 experts per layer, routing each token to the top-2 most relevant experts for
efficient inference while maintaining high model capacity.

Architecture Details:
    Grok-1 is a decoder-only MoE language model with the following components:

    - Token Embeddings: Learned token embeddings with configurable scaling
    - **Mixture-of-Experts Decoder Layers**: Each layer contains:
        * Multi-head self-attention with RoPE (Rotary Position Embeddings)
        * Grouped Query Attention (GQA) for efficient KV caching
        * **8 Expert Feed-Forward Networks**: Each token is routed to top-2 experts
        * Gating Network: Learns to route tokens to the most relevant experts
        * Router with auxiliary load-balancing loss
        * RMSNorm for layer normalization
        * Residual connections with configurable scaling
    - Final RMSNorm: Layer normalization after the last decoder layer
    - Language Model Head: Linear projection to vocabulary with output scaling

Key Features:
    - **Mixture-of-Experts**: 8 experts per layer with top-2 routing for sparse activation
    - **Massive Scale**: 314B total parameters with ~86B active per forward pass
    - Router Auxiliary Loss: Load balancing across experts to prevent collapse
    - Attention Output Multiplier: Configurable scaling for attention outputs
    - Embedding/Output Scaling: Separate multiplier scales for embeddings and outputs
    - Max Attention Value Clipping: Prevents attention weight explosion
    - Grouped Query Attention: Reduces KV cache memory requirements
    - Rotary Position Embeddings: RoPE for position encoding
    - Flash Attention Support: Optimized attention computation
    - Tensor Parallelism: Expert and model parallelism support
    - Gradient Checkpointing: Memory-efficient training

Default Configuration (Grok-1-314B):
    - Vocabulary size: 32,000
    - Hidden size: 4096 (can be scaled up to 6144+ for full model)
    - Intermediate size: 32,768 per expert
    - Number of layers: 32 (64 in full model)
    - Attention heads: 32 (48 in full model)
    - Key-value heads: 32 (8 in full model with GQA)
    - Max position embeddings: 4096
    - Number of experts: 8
    - Experts per token: 2
    - RoPE theta: 10000.0
    - Router auxiliary loss coefficient: 0.001

Usage Example:
    ```python
    from easydel import Grok1Config, Grok1ForCausalLM
    import jax.numpy as jnp
    from flax import nnx as nn

    # Initialize configuration (smaller variant for demonstration)
    config = Grok1Config(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=32768,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,  # Using GQA
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=4096,
        output_router_logits=True,  # For monitoring expert usage
        router_aux_loss_coef=0.001,
    )

    # Create causal language model
    rngs = nn.Rngs(0)
    model = Grok1ForCausalLM(
        config=config,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        rngs=rngs,
    )

    # Generate text with MoE
    input_ids = jnp.array([[1, 2, 3, 4, 5]])
    outputs = model(
        input_ids=input_ids,
        output_router_logits=True,  # Get expert routing info
    )

    logits = outputs.logits  # Shape: (batch, seq_len, vocab_size)
    router_logits = outputs.router_logits  # Expert routing information
    ```

Available Classes:
    - Grok1Config: Configuration class for Grok-1 MoE models
    - Grok1Model: Base Grok-1 transformer with Mixture-of-Experts
    - Grok1ForCausalLM: Grok-1 with causal language modeling head

Mixture-of-Experts Architecture:
    Grok-1 uses a sparse MoE architecture where:
    - Each layer has 8 expert feed-forward networks
    - A learned gating network routes each token to the top-2 experts
    - Only ~25% of parameters are active per forward pass (2 out of 8 experts)
    - Router auxiliary loss encourages balanced expert utilization
    - This enables massive model capacity (314B params) while maintaining
      computational efficiency (~86B active parameters)

Model Variants:
    - Grok-1: 314B parameters (64 layers, 48 heads, 6144 hidden size)
    - Smaller research variants with reduced layers/hidden size

Reference:
    Grok-1 by xAI.
    See: https://github.com/xai-org/grok-1

Note:
    Grok-1 is one of the largest open-weights models available. The MoE architecture
    allows it to achieve strong performance while being more computationally efficient
    than dense models of equivalent quality. The router auxiliary loss and expert
    balancing are critical for stable training.
"""

from .grok_1_configuration import Grok1Config
from .modeling_grok_1 import Grok1ForCausalLM, Grok1Model

__all__ = "Grok1Config", "Grok1ForCausalLM", "Grok1Model"
